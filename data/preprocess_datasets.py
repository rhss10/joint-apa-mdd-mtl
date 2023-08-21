# make csv files into huggingface datasets formant and make vocabulary

import ast
import json
import os
import re

import evaluate
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tgt
from datasets import (
    Audio,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    load_metric,
)

SPEECH_OCEAN_PATH = "/PATH_TO_DATA/INTERSPEECH/speechocean762/"

# make TIMIT utterance list to huggingface datasets
timit_train_ds = load_dataset("csv", data_files="./timit_train.csv", delimiter=",", column_names=["path", "phone", "text"], split="train")
timit_test_ds = load_dataset("csv", data_files="./timit_test.csv", delimiter=",", column_names=["path", "phone", "text"], split="train")
timit_phone_set = set()


def load_timit(batch):
    with open(batch["phone"], "r") as f:
        phone = []
        for line in f:
            phone.append(line.split()[-1])

    with open(batch["text"], "r") as f:
        text = f.read().strip()

    batch["phone"] = phone
    batch["text"] = text

    return batch


def preprocess_timit_phones(batch):
    phones = []
    for p in batch["phone"]:
        if p in ["h#", "epi", "pau", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl", "q"]:
            pass
        else:
            p = p.replace("em", "m")
            p = p.replace("el", "l")
            p = p.replace("en", "n")
            p = p.replace("nx", "n")
            p = p.replace("eng", "ng")
            p = p.replace("dx", "t")
            p = p.replace("ux", "uw")
            p = p.replace("axr", "er")
            p = p.replace("ix", "ih")
            p = p.replace("ax-h", "ah")
            p = p.replace("ax", "ah")
            p = p.replace("hv", "hh")
            phones.append(p.upper())

    batch["phone"] = " ".join(phones)
    timit_phone_set.update(phones)

    return batch


timit_train_ds = timit_train_ds.map(load_timit)
timit_train_ds = timit_train_ds.map(preprocess_timit_phones)
timit_train_ds = timit_train_ds.map(lambda x: {"audio": x["path"]})
timit_train_ds = timit_train_ds.cast_column("audio", Audio(sampling_rate=16000))
timit_train_ds = timit_train_ds.rename_column("phone", "ans")

timit_test_ds = timit_test_ds.map(load_timit)
timit_test_ds = timit_test_ds.map(preprocess_timit_phones)
timit_test_ds = timit_test_ds.map(lambda x: {"audio": x["path"]})
timit_test_ds = timit_test_ds.cast_column("audio", Audio(sampling_rate=16000))
timit_test_ds = timit_test_ds.rename_column("phone", "ans")

timit_test_ds.save_to_disk("/PATH_TO_DATA/timit_test")
timit_train_ds.save_to_disk("/PATH_TO_DATA/timit_train")


# make L2-ARCTIC utterance list to huggingface datasets
# 'real' refers to non-native realized phone sequences that include SID. Used for forced-alignment
train_arctic_ds = load_dataset("csv", data_files="./l2arctic_train.txt", delimiter="\t", column_names=["audio", "phone", "text"], split="train")
test_arctic_ds = load_dataset("csv", data_files="./l2arctic_test.txt", delimiter="\t", column_names=["audio", "phone", "text"], split="train")
arctic_phone_set = set()


def load_l2arctic(batch):
    tg = tgt.read_textgrid(batch["phone"])
    phone_annotations = tg.get_tier_by_name("phones").annotations
    batch["phone"] = [annotations.text for annotations in phone_annotations]

    with open(batch["text"], "r") as f:
        batch["text"] = f.read().strip()

    return batch


def preprocess_arctic_phones(batch):
    canon = []
    real = []
    ans = []
    for p in batch["phone"]:
        p = p.upper()
        p = p.replace("AX", "AH")
        p = re.sub("[0-9 *_)`]", "", p)

        # SID cases
        if "," in p:
            c, r, code = p.split(",")
            canon.append(c)
            real.append(r)
            if not ("SP" in r or "SIL" in r):
                ans.append(r)
        # C cases
        elif not ("SP" in p or "SIL" in p):
            canon.append(p)
            real.append(p)
            ans.append(p)

    arctic_phone_set.update(canon)
    arctic_phone_set.update(ans)

    batch["canon"] = " ".join(canon)
    batch["real"] = " ".join(real)
    batch["ans"] = " ".join(ans)

    return batch


train_arctic_ds = train_arctic_ds.map(load_l2arctic)
train_arctic_ds = train_arctic_ds.map(preprocess_arctic_phones)
train_arctic_ds = train_arctic_ds.map(lambda x: {"path": x["audio"]})
train_arctic_ds = train_arctic_ds.cast_column("audio", Audio(sampling_rate=16000))
train_arctic_ds = train_arctic_ds.remove_columns(["phone"])

test_arctic_ds = test_arctic_ds.map(load_l2arctic)
test_arctic_ds = test_arctic_ds.map(preprocess_arctic_phones)
test_arctic_ds = test_arctic_ds.map(lambda x: {"path": x["audio"]})
test_arctic_ds = test_arctic_ds.cast_column("audio", Audio(sampling_rate=16000))
test_arctic_ds = test_arctic_ds.remove_columns(["phone"])

train_arctic_ds.save_to_disk("/PATH_TO_DATA/l2-arctic_train")
test_arctic_ds.save_to_disk("/PATH_TO_DATA/l2-arctic_test")


# make SPEECHOCEAN762 utterance list to huggingface datasets
# 'real' refers to non-native realized phone sequences that include SID. Used for forced-alignment
train_ocean_ds = load_dataset("csv", data_files="./speechocean_train.csv", delimiter="|", split="train")
test_ocean_ds = load_dataset("csv", data_files="./speechocean_test.csv", delimiter="|", split="train")
ocean_phone_set = set()

w_stress = set()
w_acc = set()
w_tot = set()
p_acc = set()

tot = set()
comp = set()
pros = set()
flu = set()
acc = set()


def load_ocean(batch):
    batch["completeness"] = int(round(batch["completeness"]))

    batch["w_total"] = ast.literal_eval(batch["w_total"])
    batch["w_accuracy"] = ast.literal_eval(batch["w_accuracy"])
    batch["w_stress"] = ast.literal_eval(batch["w_stress"])
    batch["p_accuracy"] = ast.literal_eval(batch["p_accuracy"])
    batch["phone"] = ast.literal_eval(batch["phone"])
    batch["canon"] = ast.literal_eval(batch["canon"])
    batch["real"] = ast.literal_eval(batch["real"])
    batch["mispronunciations"] = ast.literal_eval(batch["mispronunciations"])

    batch["path"] = SPEECH_OCEAN_PATH + batch["path"]
    for i in range(len(batch["p_accuracy"])):
        batch["p_accuracy"][i] = int(batch["p_accuracy"][i] * 5)

    w_tot.update(batch["w_total"])
    w_acc.update(batch["w_accuracy"])
    w_stress.update(batch["w_stress"])
    p_acc.update(batch["p_accuracy"])

    tot.add(batch["total"])
    comp.add(batch["completeness"])
    pros.add(batch["prosodic"])
    flu.add(batch["fluency"])
    acc.add(batch["accuracy"])

    return batch


def preprocess_ocean_phones(batch):
    canon_list = []
    real_list = []
    ans_list = []

    for i in range(len(batch["canon"])):
        canon_list.append(re.sub("[0-9*]", "", batch["canon"][i]))
        ans = batch["real"][i]
        ans = re.sub("[0-9*]", "", ans)
        ans = re.sub("<unk>", "ERR", ans)
        ans = re.sub("<DEL>", "", ans)

        real = batch["real"][i]
        real = re.sub("[0-9*]", "", real)
        real = re.sub("<unk>", "ERR", real)

        real_list.append(real)
        ans_list.append(ans)

    batch["canon"] = re.sub(" +", " ", " ".join(canon_list))
    batch["real"] = re.sub(" +", " ", " ".join(real_list))
    batch["ans"] = re.sub(" +", " ", " ".join(ans_list))
    if batch["ans"].strip() == "":
        batch["ans"] = "ERR"

    ocean_phone_set.update(batch["canon"].split())
    ocean_phone_set.update(batch["ans"].split())

    return batch


train_ocean_ds = train_ocean_ds.map(load_ocean)
train_ocean_ds = train_ocean_ds.map(lambda x: {"audio": x["path"]})
train_ocean_ds = train_ocean_ds.cast_column("audio", Audio(sampling_rate=16000))
train_ocean_ds = train_ocean_ds.map(preprocess_ocean_phones)

test_ocean_ds = test_ocean_ds.map(load_ocean)
test_ocean_ds = test_ocean_ds.map(lambda x: {"audio": x["path"]})
test_ocean_ds = test_ocean_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ocean_ds = test_ocean_ds.map(preprocess_ocean_phones)

train_ocean_ds.save_to_disk("/PATH_TO_DATA/speechocean_train_ds")
test_ocean_ds.save_to_disk("/PATH_TO_DATA/speechocean_test_ds")


# make vocabulary set
def extract_all_chars_phone(batch):
    all_text = " ".join(batch["ans"])
    vocab = list(set(all_text.split()))
    vocab.append(" ")
    return {"vocab": [vocab], "all_text": [all_text]}


timit_train = load_from_disk("/PATH_TO_DATA/timit_train/")
timit_test = load_from_disk("/PATH_TO_DATA/timit_test/")
arctic_train = load_from_disk("/PATH_TO_DATA/l2-arctic_train/")
arctic_test = load_from_disk("/PATH_TO_DATA/l2-arctic_test/")

timit_ds = concatenate_datasets([timit_train, timit_test])
arctic_ds = concatenate_datasets([arctic_train, arctic_test])
train_ds = load_from_disk("/PATH_TO_DATA/speechocean_train_ds/")

timit_vocab = timit_ds.map(extract_all_chars_phone, batched=True, batch_size=-1, remove_columns=timit_ds.column_names)
arctic_vocab = arctic_ds.map(extract_all_chars_phone, batched=True, batch_size=-1, remove_columns=arctic_ds.column_names)
train_vocab = train_ds.map(extract_all_chars_phone, batched=True, batch_size=-1, remove_columns=train_ds.column_names)

vocab_list = list(set(timit_vocab["vocab"][0]) | set(arctic_vocab["vocab"][0]) | set(train_vocab["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

with open("../vocab/vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

print("- Finished making datasets and vocabulary")
