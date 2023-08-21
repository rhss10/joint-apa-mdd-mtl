import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    load_metric,
)
from transformers import (
    AutoConfig,
    HubertForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    WavLMForCTC,
)

# load data, processor and model
DS1 = "/PATH_TO_DATA/timit_test/"
DS2 = "/PATH_TO_DATA/l2-arctic_test/"
PTM = "/PATH_TO_MODEL/trainer/" + sys.argv[1]
print("- MODEL:", PTM)
timit_test = load_from_disk(DS1)
arctic_test = load_from_disk(DS2)
processor = Wav2Vec2Processor.from_pretrained(PTM)
config = AutoConfig.from_pretrained(PTM)
if config.model_type == "hubert":
    print("-Using Hubert")
    model = HubertForCTC.from_pretrained(PTM)
elif config.model_type == "wavlm":
    print("-Using WavLM")
    model = WavLMForCTC.from_pretrained(PTM)
else:
    print("-Using Wav2Vec2")
    model = Wav2Vec2ForCTC.from_pretrained(PTM)


# evaluation
def prepare_dataset(batch):
    array = batch["audio"]["array"]
    if array.dtype != "float32":
        array = np.float32(array)

    # batched output is "un-batched"
    batch["input_values"] = processor(array, sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["ans"], padding=True).input_ids

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


timit_test = timit_test.map(prepare_dataset)
arctic_test = arctic_test.map(prepare_dataset)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")


def compute_wer(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = TrainingArguments(
    output_dir=".",
    group_by_length=True,
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_wer,
    eval_dataset=timit_test,
    tokenizer=processor.feature_extractor,
)

print("- DATA:", DS1)
metrics = trainer.evaluate()
metrics["timit_samples"] = len(timit_test)
trainer.log_metrics("timit", metrics)
