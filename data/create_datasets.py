## a code to make TIMIT, L2-arctic, and Speechocean762 utterance list into csv files
import json
import os
import re

from datasets import load_dataset

# make TIMIT utterance list into csv files
PATH = "/PATH_TO_DATA/timit/TIMIT/"
timit_train = open("./timit_train.csv", "w")
timit_test = open("./timit_test.csv", "w")

for r, d, f in os.walk(PATH):
    d.sort()
    f.sort()
    for file in f:
        if ".PHN" in file:
            name = file.split(".PHN")[0]
            audio = name + ".WAV"
            text = name + ".TXT"

            if "TEST" in r:
                print(os.path.join(r, audio), os.path.join(r, file), os.path.join(r, text), sep=",", file=timit_test)
            else:
                print(os.path.join(r, audio), os.path.join(r, file), os.path.join(r, text), sep=",", file=timit_train)

timit_train.close()
timit_test.close()


# make L2-ARCTIC utterance list into csv files
PATH = "/PATH_TO_DATA/l2-arctic/"
testset = ["NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"]
devset = ["MBMPS", "THV", "SVBI", "NCC", "YDCK", "YBAA"]
trainset = ["BWC", "PNV", "EBVS", "HQTV", "ERMS", "HKK", "LXC", "ASI", "SKA", "RRBI", "ABA", "HJK"]
arctic_train = open("./l2arctic_train.txt", "w")
arctic_test = open("./l2arctic_test.txt", "w")
for t in trainset + devset:
    for r, d, f in os.walk(PATH + t):
        f.sort()
        for file in f:
            if "annotation" in r:
                path = r.split("annotation")[0]
                name = file.split(".TextGrid")[0]
                arctic_train.write(
                    path + "wav/" + name + ".wav" + "\t" + path + "annotation/" + file + "\t" + path + "transcript/" + name + ".txt" + "\n"
                )

for t in testset:
    for r, d, f in os.walk(PATH + t):
        f.sort()
        for file in f:
            if "annotation" in r:
                path = r.split("annotation")[0]
                name = file.split(".TextGrid")[0]
                arctic_test.write(
                    path + "wav/" + name + ".wav" + "\t" + path + "annotation/" + file + "\t" + path + "transcript/" + name + ".txt" + "\n"
                )

arctic_train.close()
arctic_test.close()


# make SPEECHOCEAN762 utterance list into csv files
SPEECH_OCEAN_PATH = "/PATH_TO_DATA/INTERSPEECH/speechocean762/"
TRAIN_SCORE_PATH = SPEECH_OCEAN_PATH + "train/all-info.json"
TRAIN_WAV_PATH = SPEECH_OCEAN_PATH + "train/wav.scp"
TEST_SCORE_PATH = SPEECH_OCEAN_PATH + "test/all-info.json"
TEST_WAV_PATH = SPEECH_OCEAN_PATH + "test/wav.scp"
RSC_SCORES_DETAIL = SPEECH_OCEAN_PATH + "resource/scores-detail.json"
RSC_SCORES = SPEECH_OCEAN_PATH + "resource/scores.json"
RSC_TEXT_PHONE = SPEECH_OCEAN_PATH + "resource/text-phone"
with open(TRAIN_SCORE_PATH, "r") as f:
    train_scores = json.load(f)
with open(TEST_SCORE_PATH, "r") as f:
    test_scores = json.load(f)
with open(RSC_SCORES_DETAIL, "r") as f:
    detail_scores = json.load(f)
with open(RSC_SCORES, "r") as f:
    scores = json.load(f)

train_wav_path = load_dataset("csv", data_files=TRAIN_WAV_PATH, delimiter="\t", column_names=["id", "path"], split="train")
test_wav_path = load_dataset("csv", data_files=TEST_WAV_PATH, delimiter="\t", column_names=["id", "path"], split="train")
train_wav_dict = {}
for i in range(len(train_wav_path)):
    idx = str(train_wav_path[i]["id"])
    train_wav_dict[idx] = train_wav_path[i]["path"]
test_wav_dict = {}
for i in range(len(test_wav_path)):
    idx = str(test_wav_path[i]["id"])
    test_wav_dict[idx] = test_wav_path[i]["path"]

ocean_train = open("./speechocean_train.csv", "w")
ocean_test = open("./speechocean_test.csv", "w")

print(
    "ID",
    "accuracy",
    "completeness",
    "fluency",
    "prosodic",
    "total",
    "w_total",
    "w_accuracy",
    "w_stress",
    "p_accuracy",
    "text",
    "phone",
    "canon",
    "real",
    "path",
    "mispronunciations",
    sep="|",
    file=ocean_train,
)
for utt in train_scores:
    phones_list = []
    canon_list = []
    real_list = []
    idx = 0
    w_total_list = []
    w_accuracy_list = []
    w_stress_list = []
    p_accuracy_list = []
    mis_list = []

    for word in train_scores[utt]["words"]:
        w_total_list.append(word["total"])
        w_accuracy_list.append(word["accuracy"])
        w_stress_list.append(word["stress"])

        for phone in word["phones-accuracy"]:
            p_accuracy_list.append(phone)
        for phone in word["phones"]:
            phones_list.append(phone)
            canon_list.append(phone)
            real_list.append(phone)
        for mis in word["mispronunciations"]:
            mis_list.append(mis)
            phone_no_stress = re.sub("[0-9]", "", canon_list[idx + mis["index"]])
            assert phone_no_stress == mis["canonical-phone"]
            real_list[idx + mis["index"]] = mis["pronounced-phone"]
        idx = len(real_list)

    utt_no_zero = re.sub("^0+", "", utt)

    print(
        utt,
        train_scores[utt]["accuracy"],
        round(train_scores[utt]["completeness"]),
        train_scores[utt]["fluency"],
        train_scores[utt]["prosodic"],
        train_scores[utt]["total"],
        w_total_list,
        w_accuracy_list,
        w_stress_list,
        p_accuracy_list,
        train_scores[utt]["text"],
        phones_list,
        canon_list,
        real_list,
        train_wav_dict[utt_no_zero],
        mis_list,
        sep="|",
        file=ocean_train,
    )


print(
    "ID",
    "accuracy",
    "completeness",
    "fluency",
    "prosodic",
    "total",
    "w_total",
    "w_accuracy",
    "w_stress",
    "p_accuracy",
    "text",
    "phone",
    "canon",
    "real",
    "path",
    "mispronunciations",
    sep="|",
    file=ocean_test,
)
for utt in test_scores:
    phones_list = []
    canon_list = []
    real_list = []
    idx = 0
    w_total_list = []
    w_accuracy_list = []
    w_stress_list = []
    p_accuracy_list = []
    mis_list = []

    for word in test_scores[utt]["words"]:
        w_total_list.append(word["total"])
        w_accuracy_list.append(word["accuracy"])
        w_stress_list.append(word["stress"])

        for phone in word["phones-accuracy"]:
            p_accuracy_list.append(phone)
        for phone in word["phones"]:
            phones_list.append(phone)
            canon_list.append(phone)
            real_list.append(phone)
        for mis in word["mispronunciations"]:
            mis_list.append(mis)
            phone_no_stress = re.sub("[0-9]", "", canon_list[idx + mis["index"]])
            assert phone_no_stress == mis["canonical-phone"]
            real_list[idx + mis["index"]] = mis["pronounced-phone"]
        idx = len(real_list)

    utt_no_zero = re.sub("^0+", "", utt)

    print(
        utt,
        test_scores[utt]["accuracy"],
        round(test_scores[utt]["completeness"]),
        test_scores[utt]["fluency"],
        test_scores[utt]["prosodic"],
        test_scores[utt]["total"],
        w_total_list,
        w_accuracy_list,
        w_stress_list,
        p_accuracy_list,
        test_scores[utt]["text"],
        phones_list,
        canon_list,
        real_list,
        test_wav_dict[utt_no_zero],
        mis_list,
        sep="|",
        file=ocean_test,
    )

ocean_train.close()
ocean_test.close()

print("- Finished making datasets for TIMIT, L2-ARCTIC, and SPEECHOCEAN762.")
