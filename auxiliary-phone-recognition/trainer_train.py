# fine-tune SSL models on phone recognitoin to use for transfer learning

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk, load_metric
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    EarlyStoppingCallback,
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    WavLMForCTC,
)


def prepare_arguments():
    parser = argparse.ArgumentParser(description=("Train the SSL on phone recognition."))
    parser.add_argument("--batch_size", type=int, default=4)  # The paper was trained using 2 GPUs. 8 batches in total.
    parser.add_argument("--model_name_or_path", type=str, default="facebook/wav2vec2-large-robust")
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--ctc_loss_reduction", type=str, default="mean")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_wer")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--exp_prefix", type=str, default="")
    parser.add_argument("--freeze_feature_extractor", type=bool, default=True)

    args = parser.parse_args()
    args.exp_name = f"{args.exp_prefix}_lr{args.learning_rate}_warm{args.warmup_ratio}_type-{args.lr_scheduler_type}"
    args.save_dir_path = "/SAVE_PATH/trainer/" + args.exp_name
    args.save_log_path = "./logs/trainer_tracker/" + args.exp_name
    os.makedirs(args.save_dir_path, exist_ok=False)
    os.makedirs(args.save_log_path, exist_ok=False)

    with open(os.path.join(args.save_dir_path, "args.json"), "w") as args_file:
        json.dump(vars(args), args_file, ensure_ascii=False)

    return args


def prepare_dataset(batch):
    array = batch["audio"]["array"]

    # batched output is "un-batched"
    batch["input_values"] = processor(array, sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["ans"]).input_ids

    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


def compute_wer(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def prepare_trainer(args, processor, train_ds, test_ds):
    if "wav2vec2" in args.model_name_or_path:
        model = Wav2Vec2ForCTC.from_pretrained(
            args.model_name_or_path,
            ctc_loss_reduction=args.ctc_loss_reduction,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    elif "hubert" in args.model_name_or_path:
        model = HubertForCTC.from_pretrained(
            args.model_name_or_path,
            ctc_loss_reduction=args.ctc_loss_reduction,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    elif "wavlm" in args.model_name_or_path:
        model = WavLMForCTC.from_pretrained(
            args.model_name_or_path,
            ctc_loss_reduction=args.ctc_loss_reduction,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        )

    if args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=args.save_dir_path,
        logging_dir=args.save_log_path,
        report_to=["tensorboard"],
        group_by_length=args.group_by_length,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        log_level="info",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=3,
        push_to_hub=False,
        load_best_model_at_end=True,
        greater_is_better=args.greater_is_better,
        metric_for_best_model=args.metric_for_best_model,
    )

    with open(os.path.join(args.save_dir_path, "trainer_args.json"), "w") as args_file:
        json.dump(training_args.to_dict(), args_file, ensure_ascii=False)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_wer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=processor.feature_extractor,
    )

    return trainer


if __name__ == "__main__":
    args = prepare_arguments()
    print(args)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=args.save_log_path + "/logging.log",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    DS1 = "/PATH_TO_DATA/timit_train/"
    DS2 = "/PATH_TO_DATA/timit_test/"
    DS3 = "/PATH_TO_DATA/l2-arctic_train/"
    DS4 = "/PATH_TO_DATA/l2-arctic_test/"
    VOCAB = "../vocab/vocab.json"
    print(DS1, DS2, DS3, DS4, VOCAB, sep="\n")
    timit_train = load_from_disk(DS1)
    timit_test = load_from_disk(DS2)
    arctic_train = load_from_disk(DS3)
    arctic_test = load_from_disk(DS4)
    # ds_train = concatenate_datasets([timit_train, arctic_train])
    # ds_test = concatenate_datasets([timit_test, arctic_test])
    ds_train = timit_train
    ds_test = timit_test

    tokenizer = Wav2Vec2PhonemeCTCTokenizer(VOCAB, unk_token="[UNK]", pad_token="[PAD]", phone_delimiter_token=" ", do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.save_dir_path)

    ds_train = ds_train.map(prepare_dataset)
    ds_test = ds_test.map(prepare_dataset)

    wer_metric = evaluate.load("wer")

    trainer = prepare_trainer(args, processor, train_ds=ds_train, test_ds=ds_test)

    train_res = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_res.metrics
    metrics["train_samples"] = len(ds_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(ds_test)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    with open(args.save_log_path + ".log", "w") as f:
        for obj in trainer.state.log_history:
            f.write(str(obj))
            f.write("\n")
    print("- Training complete.")
