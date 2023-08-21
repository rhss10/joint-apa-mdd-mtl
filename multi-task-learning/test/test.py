import argparse
import math
import os
import sys

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    HubertConfig,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    WavLMConfig,
    get_scheduler,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

sys.path.append("/PATH_TO_REPO/multi-task-learning")
from models.joint_apa_mdd import JointAPAMDD, JointAPAMDDHubert, JointAPAMDDWavLM

DEVICE = "cuda"


def prepare_cfg(raw_args=None):
    parser = argparse.ArgumentParser(description=("Test the model."))
    parser.add_argument("--batch_size", type=int, default=8, help="same batch size for train and test")
    parser.add_argument("--model_name_or_path", type=str, help="path to the model provided by bash script")
    parser.add_argument("--num_classes", type=int, default=11, help="speechocean762 dataset has 0-10 range of scores")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--group_by_length", type=bool, default=False)

    args = parser.parse_args(raw_args)
    args.exp_name = str(args.model_name_or_path).split("/")[-1]
    os.makedirs(os.path.join("./", args.exp_name), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args


# collator for huggingface.datasets
def collate_fn(batch):
    with processor.as_target_processor():
        ctc_labels = [processor(x["ans"]).input_ids for x in batch]

    return {
        "input_values": processor(
            [np.float32(x["audio"]["array"]) for x in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_values,
        "input_lengths": torch.LongTensor([len(x["audio"]["array"]) for x in batch]),
        "cls_labels": torch.LongTensor([[x["total"], x["prosodic"], x["fluency"], x["accuracy"]] for x in batch]),
        "ctc_labels": torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(c) for c in ctc_labels],
            batch_first=True,
            padding_value=-100,
        ),
        "text": [x["ans"] for x in batch],
        "real": [x["real"] for x in batch],
        "canon": [x["canon"] for x in batch],
    }


# sample by length
def length_sampler(cfg, split, processor, ds):
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    if cfg.group_by_length and split == "train":
        print("- Using Length Sampler")
        model_input_name = processor.tokenizer.model_input_names[0]  # will output 'input_ids'
        return LengthGroupedSampler(
            cfg.batch_size,
            dataset=ds,
            lengths=[len(x["audio"]["array"]) for x in ds],
            model_input_name=model_input_name,
            generator=generator,
        )
    else:
        return RandomSampler(ds, generator=generator)


def test(model, processor, ds):
    num_classes = list(range(cfg.num_classes))
    losses, cls_losses, ctc_losses = [], [], []

    tot_preds_sum, tot_labels_sum = [], []
    pros_preds_sum, pros_labels_sum = [], []
    flu_preds_sum, flu_labels_sum = [], []
    acc_preds_sum, acc_labels_sum = [], []

    PREDS = open(f"{cfg.exp_name}/PREDS_{cfg.exp_name}", "w")
    ANNOT = open(f"{cfg.exp_name}/ANNOT_{cfg.exp_name}", "w")
    CANON = open(f"{cfg.exp_name}/CANON_{cfg.exp_name}", "w")
    num = 1

    model.eval()
    for step, x in tqdm(enumerate(ds)):
        with torch.no_grad():
            loss, cls_loss, ctc_loss, _, cls_logits, ctc_logits = model(
                **{k: v.to(model.device) for k, v in x.items() if k not in ["text", "canon", "real"]}
            )
        losses.append(loss.item())
        cls_losses.append(cls_loss.item())
        ctc_losses.append(ctc_loss.item())

        cls_preds = torch.argmax(cls_logits, dim=1)
        ctc_preds = processor.batch_decode(torch.argmax(ctc_logits, dim=-1))

        if step < 2:
            for i in range(3):
                print(torch.argmax(ctc_logits[i], dim=-1))
                print(f"PRD: [{ctc_preds[i]}]")
                print(f'ANS: [{x["text"][i]}]')
                print(f'PRD SCORE: {cls_preds[i]}, ANS SCORE: {x["cls_labels"][i]}')
                print("PER:", each_per.compute(predictions=[ctc_preds[i]], references=[x["text"][i]]))

        per_metric.add_batch(predictions=ctc_preds, references=x["text"])
        tot_metric.add_batch(predictions=cls_preds.T[0], references=x["cls_labels"].T[0])
        pros_metric.add_batch(predictions=cls_preds.T[1], references=x["cls_labels"].T[1])
        flu_metric.add_batch(predictions=cls_preds.T[2], references=x["cls_labels"].T[2])
        acc_metric.add_batch(predictions=cls_preds.T[3], references=x["cls_labels"].T[3])

        tot_preds_sum.extend(cls_preds.T[0].tolist())
        tot_labels_sum.extend(x["cls_labels"].T[0].tolist())
        pros_preds_sum.extend(cls_preds.T[1].tolist())
        pros_labels_sum.extend(x["cls_labels"].T[1].tolist())
        flu_preds_sum.extend(cls_preds.T[2].tolist())
        flu_labels_sum.extend(x["cls_labels"].T[2].tolist())
        acc_preds_sum.extend(cls_preds.T[3].tolist())
        acc_labels_sum.extend(x["cls_labels"].T[3].tolist())

        for i in range(len(x["text"])):
            print(num, ctc_preds[i], file=PREDS, sep="\t")
            print(num, x["real"][i], file=ANNOT, sep="\t")
            print(num, x["canon"][i], file=CANON, sep="\t")
            num += 1

    per_res = {"per": per_metric.compute()}
    tot_res = tot_metric.compute()
    pros_res = pros_metric.compute()
    flu_res = flu_metric.compute()
    acc_res = acc_metric.compute()

    tot_res["total"] = tot_res.pop("pearsonr")
    pros_res["prosodic"] = pros_res.pop("pearsonr")
    flu_res["fluency"] = flu_res.pop("pearsonr")
    acc_res["accuracy"] = acc_res.pop("pearsonr")

    all_res = {}
    all_res.update(**acc_res, **flu_res, **pros_res, **tot_res)
    total = 0
    for aspect in all_res:
        if math.isnan(all_res[aspect]):
            total += 0
        else:
            total += all_res[aspect]
    all_res.update(**{"all": total / 4})

    loss_res = {
        "average_loss": np.array(losses).mean(),
        "average_cls_loss": np.array(cls_losses).mean(),
        "average_ctc_loss": np.array(ctc_losses).mean(),
    }
    all_res.update(**per_res, **loss_res)

    tot_cm = confusion_matrix(tot_labels_sum, tot_preds_sum, labels=num_classes)
    pros_cm = confusion_matrix(pros_labels_sum, pros_preds_sum, labels=num_classes)
    flu_cm = confusion_matrix(flu_labels_sum, flu_preds_sum, labels=num_classes)
    acc_cm = confusion_matrix(acc_labels_sum, acc_preds_sum, labels=num_classes)

    df = pd.DataFrame([acc_preds_sum, flu_preds_sum, pros_preds_sum, tot_preds_sum]).transpose()
    df.columns = ["acc_preds", "flu_preds", "pros_preds", "tot_preds"]
    df.to_csv(f"{cfg.exp_name}/prediction.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    heatmap(
        tot_cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidth=0.2,
        cbar=False,
        xticklabels=num_classes,
        yticklabels=num_classes,
        annot_kws={"size": 5},
        ax=ax[0, 0],
    )
    ax[0, 0].set(xlabel=f'Total Predictions (PCC:{tot_res["total"]:.3f})', ylabel="Total Labels")

    heatmap(
        pros_cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidth=0.2,
        cbar=False,
        xticklabels=num_classes,
        yticklabels=num_classes,
        annot_kws={"size": 5},
        ax=ax[0, 1],
    )
    ax[0, 1].set(xlabel=f'Prosodic Predictions (PCC:{pros_res["prosodic"]:.3f})', ylabel="Prosodic Labels")

    heatmap(
        flu_cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidth=0.2,
        cbar=False,
        xticklabels=num_classes,
        yticklabels=num_classes,
        annot_kws={"size": 5},
        ax=ax[1, 0],
    )
    ax[1, 0].set(xlabel=f'Fluency Predictions (PCC:{flu_res["fluency"]:.3f})', ylabel="Fluency Labels")

    heatmap(
        acc_cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidth=0.2,
        cbar=False,
        xticklabels=num_classes,
        yticklabels=num_classes,
        annot_kws={"size": 5},
        ax=ax[1, 1],
    )
    ax[1, 1].set(xlabel=f'Accuracy Predictions (PCC:{acc_res["accuracy"]:.3f})', ylabel="Accuracy Labels")
    fig.savefig(f"{cfg.exp_name}/cm_{cfg.exp_name}_4scores.pdf", bbox_inches="tight")
    fig.savefig(f"{cfg.exp_name}/cm_{cfg.exp_name}_4scores.png", bbox_inches="tight")

    PREDS.close()
    ANNOT.close()
    CANON.close()

    return all_res


if __name__ == "__main__":
    cfg = prepare_cfg()
    print(cfg)

    processor = Wav2Vec2Processor.from_pretrained(cfg.model_name_or_path)
    TEST = "/PATH_TO_DATA/speechocean_test_ds"
    print("- Test Data:", TEST)
    print("- Model:", cfg.model_name_or_path)
    test_ds = load_from_disk(TEST)
    if cfg.group_by_length:
        test_sampler = length_sampler(cfg, "test", processor, test_ds)
    else:
        test_sampler = length_sampler(cfg, "test", processor, test_ds)

    # NOTE: sampler option is mutually exclusive with shuffle
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    print("***** Running test *****")
    print(f"  Num examples = {len(test_ds)}")
    print(f"  Batch size = {cfg.batch_size}")
    print(f"  Dataloader size = {len(test_dataloader)}")

    per_metric = evaluate.load("wer")
    each_per = evaluate.load("wer")
    tot_metric = evaluate.load("pearsonr")
    pros_metric = evaluate.load("pearsonr")
    flu_metric = evaluate.load("pearsonr")
    acc_metric = evaluate.load("pearsonr")

    # Test on the best model
    config = AutoConfig.from_pretrained(os.path.join(cfg.model_name_or_path, "last-model-ckpt"))
    if config.model_type == "wav2vec2":
        print("- Using Wav2Vec2 model")
        best_model = JointAPAMDD.from_pretrained(os.path.join(cfg.model_name_or_path, "last-model-ckpt")).to(DEVICE)
    elif config.model_type == "hubert":
        print("- Using Hubert model")
        best_model = JointAPAMDDHubert.from_pretrained(os.path.join(cfg.model_name_or_path, "last-model-ckpt")).to(DEVICE)
    elif config.model_type == "wavlm":
        print("- Using WavLM model")
        best_model = JointAPAMDDWavLM.from_pretrained(os.path.join(cfg.model_name_or_path, "last-model-ckpt")).to(DEVICE)

    best_model.eval()
    test_results = test(best_model, processor, test_dataloader)

    mdd_file = open("./apa_results.csv", "a")
    print(
        cfg.model_name_or_path,
        test_results["accuracy"],
        test_results["fluency"],
        test_results["prosodic"],
        test_results["total"],
        test_results["all"],
        test_results["per"],
        file=mdd_file,
        sep=",",
    )
    mdd_file.close()

    print("- Test finished.")
