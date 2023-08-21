# train the joint APA and MDD model with wav2vec2, hubert, and wavlm
# the train.py used https://arxiv.org/abs/2210.15387 as backbone code, such as MTL start epoch.
# the models are tested with the results of the last epoch, to equally compare for APA and MDD.

import argparse
import json
import math
import os
import sys

import evaluate
import numpy as np
import torch
import torchinfo
from datasets import load_from_disk
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    get_scheduler,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

sys.path.append("./")
from models.joint_apa_mdd import JointAPAMDD, JointAPAMDDHubert, JointAPAMDDWavLM

DEVICE = "cuda"


def prepare_cfg(raw_args=None):
    parser = argparse.ArgumentParser(description=("Train and evaluate the model."))
    parser.add_argument("--batch_size", type=int, default=8, help="same batch size for train and test")
    parser.add_argument("--model_name_or_path", type=str, default="/PATH_TO_DATA/trainer/TIM_robust_lr0.0001_warm0.1_type-linear")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=11, help="speechocean762 dataset has 0-10 range of scores")
    parser.add_argument("--ctc_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=0.25)
    parser.add_argument("--enable_cls_epochs", type=int, default=0)
    parser.add_argument("--metric_for_best_model", type=str, default="per")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--exp_prefix", type=str, default="")
    parser.add_argument("--save_all_epochs", action="store_true", help="Save all the epoch-wise models during training.")
    parser.add_argument("--freeze_feature_extractor", type=bool, default=True, help="Freeze convolution models in wav2vec2.")
    parser.add_argument("--ctc_loss_reduction", type=str, default="mean")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--no_phone_recognition", help="The model does not leverage auxiliary phone recognition", action="store_true")
    args = parser.parse_args(raw_args)

    args.exp_name = f"{args.exp_prefix}_bat{args.batch_size}_ctcW{args.ctc_weight}_clsW{args.cls_weight}_clsSt{args.enable_cls_epochs}"
    args.model_dir = f"/PATH_TO_MODEL/finetuned_models/{args.exp_name}"
    os.makedirs(args.model_dir, exist_ok=False)
    args.log_dir = f"./tb_tracker/{args.exp_name}"
    os.makedirs(args.log_dir, exist_ok=False)

    if args.ctc_weight == 0.0:
        args.metric_for_best_model = "all"
        args.greater_is_better = True
        print("- Using avg scores for metric")
    else:
        print(f"- Using {args.metric_for_best_model} for metric")

    return args


# collator
def collate_fn(batch):
    with processor.as_target_processor():
        ctc_labels = [processor(x["ans"]).input_ids for x in batch]

    return {
        "input_values": processor(
            [np.float32(x["audio"]["array"]) for x in batch], sampling_rate=16000, return_tensors="pt", padding=True
        ).input_values,
        "input_lengths": torch.LongTensor([len(x["audio"]["array"]) for x in batch]),
        "cls_labels": torch.LongTensor([[x["total"], x["prosodic"], x["fluency"], x["accuracy"]] for x in batch]),
        "ctc_labels": torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(c) for c in ctc_labels],
            batch_first=True,
            padding_value=-100,
        ),
        "text": [x["ans"] for x in batch],
    }


# train sampler to use when args.groups_by_length is True
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


def prepare_model(cfg, config, num_training_steps):
    if config.model_type == "wav2vec2":
        print("- Using Wav2Vec2 model")
        model = JointAPAMDD.from_pretrained(cfg.model_name_or_path, config=config).to(DEVICE)
    elif config.model_type == "hubert":
        print("- Using Hubert model")
        model = JointAPAMDDHubert.from_pretrained(cfg.model_name_or_path, config=config).to(DEVICE)
    elif config.model_type == "wavlm":
        print("- Using WavLM model")
        model = JointAPAMDDWavLM.from_pretrained(cfg.model_name_or_path, config=config).to(DEVICE)

    if cfg.freeze_feature_extractor:
        print("- Freezing feature encoder.")
        model.freeze_feature_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    return model, optimizer, scheduler


def test(model, processor, ds):
    losses, cls_losses, ctc_losses = [], [], []
    model.eval()
    for step, x in tqdm(enumerate(ds)):
        with torch.no_grad():
            loss, cls_loss, ctc_loss, _, cls_logits, ctc_logits = model(**{k: v.to(model.device) for k, v in x.items() if k != "text"})
        losses.append(loss.item())
        cls_losses.append(cls_loss.item())
        ctc_losses.append(ctc_loss.item())
        cls_preds = torch.argmax(cls_logits, dim=1)
        ctc_preds = processor.batch_decode(torch.argmax(ctc_logits, dim=-1))

        per_metric.add_batch(predictions=ctc_preds, references=x["text"])
        tot_metric.add_batch(predictions=cls_preds.T[0], references=x["cls_labels"].T[0])
        pros_metric.add_batch(predictions=cls_preds.T[1], references=x["cls_labels"].T[1])
        flu_metric.add_batch(predictions=cls_preds.T[2], references=x["cls_labels"].T[2])
        acc_metric.add_batch(predictions=cls_preds.T[3], references=x["cls_labels"].T[3])

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

    return all_res


def train(cfg, model, processor, train_ds, valid_ds, optimizer, scheduler, best_ckpt_path, last_ckpt_path, all_ckpt_path, logger):
    if cfg.greater_is_better:
        eval_target = 0
    else:
        eval_target = float("inf")

    steps = 0
    start_epoch = 0
    early_stopping_cnt = 0

    progress_bar = tqdm(range((cfg.num_train_epochs - start_epoch) * len(train_ds)))
    for epoch in range(start_epoch, cfg.num_train_epochs):
        # Train
        model.enable_cls = epoch >= cfg.enable_cls_epochs
        model.train()
        for step, x in enumerate(train_ds):
            optimizer.zero_grad()
            x = {k: v.to(model.device) for k, v in x.items() if k != "text"}
            loss, cls_loss, ctc_loss, *_ = model(**x)
            loss.backward()

            losses = {"loss": loss.item(), "ctc_loss": ctc_loss.item(), "cls_loss": cls_loss.item()}
            progress_bar.set_description_str(" | ".join([f"Epoch [{epoch}] "] + [f"{k} {v:.4f}" for k, v in losses.items()]))
            for k, v in losses.items():
                logger(f"train/{k}", v, steps)
            logger(f"train/learning_rate", scheduler.optimizer.param_groups[0]["lr"], steps)
            steps += 1

            optimizer.step()
            scheduler.step()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        eval_results = test(model, processor, valid_ds)
        print(f"\nEpoch [{epoch}] EVAL:", eval_results)
        for k, v in eval_results.items():
            logger(f"eval/{k}", v, epoch)

        # Bestkeeping
        if (cfg.greater_is_better and eval_target < eval_results[cfg.metric_for_best_model]) or (
            not cfg.greater_is_better and eval_target > eval_results[cfg.metric_for_best_model]
        ):
            print(
                f"Updating the model with better {cfg.metric_for_best_model} value.\n"
                f"Prev: {eval_target:.4f}, Curr (epoch={epoch}): {eval_results[cfg.metric_for_best_model]:.4f}\n"
                f"Removing the previous checkpoint.\n"
            )
            eval_target = eval_results[cfg.metric_for_best_model]
            model.eval()
            model.save_pretrained(best_ckpt_path)
            # reset early stopping count
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
            print(f"Counting early stopping. Curr cnt: {early_stopping_cnt}.")
            if early_stopping_cnt >= cfg.early_stopping_patience:
                print(f"Early stopping patience: {cfg.early_stopping_patience}. Stopping training.")
                break

        # Saving everything
        if cfg.save_all_epochs:
            model.eval()
            model.save_pretrained(os.path.join(all_ckpt_path, f"e-{epoch:04d}"))

        # Save last model
        model.eval()
        model.save_pretrained(last_ckpt_path)
        torch.save(optimizer.state_dict(), os.path.join(last_ckpt_path, "optimizer.pt"))
        torch.save({"last_epoch": cfg.num_train_epochs}, os.path.join(last_ckpt_path, "scheduler.pt"))


def get_logger(tb_path):
    writer = SummaryWriter(log_dir=tb_path)

    def _log(name, value, step=0):
        writer.add_scalar(name, value, step)

    return _log


if __name__ == "__main__":
    cfg = prepare_cfg()
    print(cfg)
    logger = get_logger(cfg.log_dir)

    # initialize config and processor
    if cfg.no_phone_recognition:
        print(f"- You are not using auxiliary fine-tuning. Check {cfg.model_name_or_path} is not fine-tuned.")
        VOCAB = "../vocab/vocab.json"
        tokenizer = Wav2Vec2PhonemeCTCTokenizer(VOCAB, unk_token="[UNK]", pad_token="[PAD]", phone_delimiter_token=" ", do_phonemize=False)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name_or_path)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        config = AutoConfig.from_pretrained(
            cfg.model_name_or_path,
            vocab_size=processor.tokenizer.vocab_size,
            ctc_loss_reduction=cfg.ctc_loss_reduction,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    else:
        processor = Wav2Vec2Processor.from_pretrained(cfg.model_name_or_path)
        config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    config.task_specific_params = {
        "num_classes": cfg.num_classes,
        "ctc_weight": cfg.ctc_weight,
        "cls_weight": cfg.cls_weight,
    }
    processor.save_pretrained(cfg.model_dir)

    TRAIN = "/PATH_TO_DATA/speechocean_train_ds"
    TEST = "/PATH_TO_DATA/speechocean_test_ds"
    print("- Train Data:", TRAIN)
    print("- Test Data:", TEST)
    train_ds = load_from_disk(TRAIN)
    test_ds = load_from_disk(TEST)
    train_sampler = length_sampler(cfg, "train", processor, train_ds)
    test_sampler = length_sampler(cfg, "test", processor, test_ds)
    # NOTE: sampler option is mutually exclusive with shuffle
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    model, optimizer, scheduler = prepare_model(cfg, config, num_training_steps=len(train_dataloader) * cfg.num_train_epochs)
    print(torchinfo.summary(model))

    per_metric = evaluate.load("wer")
    tot_metric = evaluate.load("pearsonr")
    pros_metric = evaluate.load("pearsonr")
    flu_metric = evaluate.load("pearsonr")
    acc_metric = evaluate.load("pearsonr")
    best_ckpt_path = os.path.join(cfg.model_dir, "best-model-ckpt")
    last_ckpt_path = os.path.join(cfg.model_dir, "last-model-ckpt")
    all_ckpt_path = os.path.join(cfg.model_dir, "model-ckpts")

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {cfg.num_train_epochs}")
    print(f"  Batch size = {cfg.batch_size}")

    # Train & Validation loop
    train(
        cfg,
        model,
        processor,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        best_ckpt_path=best_ckpt_path,
        last_ckpt_path=last_ckpt_path,
        all_ckpt_path=all_ckpt_path,
        logger=logger,
    )
    print("- Training finished.")

    # Test on the best model
    if config.model_type == "wav2vec2":
        print("- Using Wav2Vec2 model")
        best_model = JointAPAMDD.from_pretrained(os.path.join(cfg.model_dir, "last-model-ckpt")).to(DEVICE)
    elif config.model_type == "hubert":
        print("- Using Hubert model")
        best_model = JointAPAMDDHubert.from_pretrained(os.path.join(cfg.model_dir, "last-model-ckpt")).to(DEVICE)
    elif config.model_type == "wavlm":
        print("- Using WavLM model")
        best_model = JointAPAMDDWavLM.from_pretrained(os.path.join(cfg.model_dir, "last-model-ckpt")).to(DEVICE)

    best_model.eval()
    test_results = test(best_model, processor, test_dataloader)
    print("TEST:", test_results)
    for k, v in test_results.items():
        logger(f"test/{k}", v, 1)

    json.dump(test_results, open(os.path.join(cfg.log_dir, "test_metric_results.json"), "w"))
    print("- Test finished.")
