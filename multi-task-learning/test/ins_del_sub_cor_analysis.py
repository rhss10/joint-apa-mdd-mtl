#!/usr/bin/env python
# code modified from the following work:
# github code: https://github.com/cageyoko/CTC-Attention-Mispronunciation
# paper: https://arxiv.org/pdf/2104.08428.pdf

# coding: utf-8
# Author: Kaiqi Fu
# usage: require 1. ref_human_detail 2. human_our_detail 3 ref_our_detail
import re
import sys

import pandas as pd
import torch

MODEL = sys.argv[1]

f = open(f"{MODEL}/CANON_ANNOT_align", "r")
dic = {}
insert = 0
delete = 0
sub = 0
cor = 0
count = 0
##  0： ref  1：human 2：ops --- 3: human  4： our  5: ops
for line in f:
    line = line.strip()
    if "ref" in line:
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +", " ", ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]] = []
        dic[ref[0]].append(ref[1])
    elif "hyp" in line:
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +", " ", hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif " op " in line:
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +", " ", op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
        for i in op_seq:
            if i == "I":
                insert += 1
            elif i == "D":
                delete += 1
                count += 1
            elif i == "S":
                sub += 1
                count += 1
            elif i == "C":
                cor += 1
                count += 1
f.close()
## 发音错误统计
# print("insert:" ,insert)
# print("delete:" ,delete)
# print("sub:" ,sub)
# print("cor:" ,cor)
# print("sum", count)

f = open(f"{MODEL}/ANNOT_PREDS_align", "r")
for line in f:
    line = line.strip()
    fn = line.split(" ")[0]
    if fn not in dic:
        continue
    if "ref" in line:
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +", " ", ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]].append(ref[1])
    elif "hyp" in line:
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +", " ", hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif " op " in line:
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +", " ", op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
f.close()


f = open(f"{MODEL}/CANON_PREDS_align", "r")
for line in f:
    line = line.strip()
    fn = line.split(" ")[0]
    if fn not in dic:
        continue
    if "ref" in line:
        ref = line.split("ref")
        ref[0] = ref[0].strip(" ")
        ref[1] = ref[1].strip(" ")
        ref[1] = re.sub(" +", " ", ref[1])
        ref_seq = ref[1].split(" ")
        dic[ref[0]].append(ref[1])
    elif "hyp" in line:
        hyp = line.split("hyp")
        hyp[0] = hyp[0].strip(" ")
        hyp[1] = hyp[1].strip(" ")
        hyp[1] = re.sub(" +", " ", hyp[1])
        hyp_seq = hyp[1].split(" ")
        dic[hyp[0]].append(hyp[1])
    elif " op " in line:
        op = line.split(" op ")
        op[0] = op[0].strip(" ")
        op[1] = op[1].strip(" ")
        op[1] = re.sub(" +", " ", op[1])
        op_seq = op[1].split(" ")
        dic[op[0]].append(op[1])
f.close()


cor_cor = 0
cor_cor1 = 0
cor_nocor = 0

sub_sub = 0
sub_sub1 = 0
sub_nosub = 0

ins_ins = 0
ins_ins1 = 0
ins_noins = 0

del_del = 0
del_del1 = 0
del_nodel = 0


MISPRONUNCIATIONS = []
for i in dic:
    arr = dic[i]
    MISPRONUNCIATION = 0
    # del detection
    ref_seq = arr[0].split(" ")
    ref_seq3 = arr[6].split(" ")
    op = arr[2].split(" ")
    op3 = arr[8].split(" ")
    flag = 0
    for i in range(len(ref_seq)):
        if ref_seq[i] == "***":
            continue
        while ref_seq3[flag] == "***":
            flag += 1
        if ref_seq[i] == ref_seq3[flag] and ref_seq[i] != "***":
            if op[i] == "D" and op3[flag] == "D":
                del_del += 1
                MISPRONUNCIATION += 1
            elif op[i] == "D" and op3[flag] != "D" and op3[flag] != "C":
                del_del1 += 1
                MISPRONUNCIATION += 1
            elif op[i] == "D" and op3[flag] != "D" and op3[flag] == "C":
                del_nodel += 1
            flag += 1

    ## cor ins sub detection
    ref_seq = arr[0].split(" ")
    human_seq = arr[1].split(" ")
    op = arr[2].split(" ")
    human_seq2 = arr[3].split(" ")
    our_seq2 = arr[4].split(" ")
    op2 = arr[5].split(" ")
    flag = 0
    for i in range(len(human_seq)):
        if human_seq[i] == "***":
            continue
        while human_seq2[flag] == "***":
            flag += 1
        if human_seq[i] == human_seq2[flag] and human_seq[i] != "***":
            if op[i] == "C" and op2[flag] == "C":
                cor_cor += 1
            elif op[i] == "C" and op2[flag] != "C":
                cor_nocor += 1
                MISPRONUNCIATION += 1

            if op[i] == "S" and op2[flag] == "C":
                sub_sub += 1
                MISPRONUNCIATION += 1
            elif op[i] == "S" and op2[flag] != "C" and ref_seq[i] != our_seq2[flag]:
                sub_sub1 += 1
                MISPRONUNCIATION += 1
            elif op[i] == "S" and op2[flag] != "C" and ref_seq[i] == our_seq2[flag]:
                sub_nosub += 1

            if op[i] == "I" and op2[flag] == "C":
                ins_ins += 1
                MISPRONUNCIATION += 1
            elif op[i] == "I" and op2[flag] != "C" and op2[flag] != "D":
                ins_ins1 += 1
                MISPRONUNCIATION += 1
            elif op[i] == "I" and op2[flag] != "C" and op2[flag] == "D":
                ins_noins += 1

            flag += 1
    MISPRONUNCIATIONS.append(MISPRONUNCIATION)

sum1 = cor_cor + cor_nocor + sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel
# print("sum:",sum1)
TP = sub_sub + ins_ins + del_del + sub_sub1 + ins_ins1 + del_del1
FP = cor_nocor
FN = sub_nosub + ins_noins + del_nodel

FA = sub_nosub + ins_noins + del_nodel
TR = TP
TA = cor_cor
FR = cor_nocor

err_count = sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel
Correct_Diag = sub_sub + ins_ins + del_del
Error_Diag = sub_sub1 + ins_ins1 + del_del1

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (recall + precision)
cd_precision = TA / (FA + TA)
cd_recall = TA / (TA + FR)
cd_f1 = 2 * cd_precision * cd_recall / (cd_recall + cd_precision)

FAR = 1 - recall
FRR = FR / (TA + FR)
DER = Error_Diag / (Error_Diag + Correct_Diag)

mdd_file = open("./mdd_results.csv", "a")
print(MODEL, cd_precision, cd_recall, cd_f1, precision, recall, f1, file=mdd_file, sep=",")
mdd_file.close()

# print("Precision: %.4f" %(precision))
# print("Recall: %.4f" %(recall))
# print("F1:%.4f" % (f1))

# print("CD Precision: %.4f" %(cd_precision))
# print("CD Recall: %.4f" %(cd_recall))
# print("CD f1:%.4f" % (cd_f1))

# print("TR: %.4f %d" %(TR/(cor_cor+cor_nocor), TR))
# print("TA: %.4f %d" %(TA/(cor_cor+cor_nocor), TA))
# print("FR: %.4f %d" %(FR/(cor_cor+cor_nocor), FR))
# print("FA: %.4f %d" %(FA/err_count, FA))

# print("Correct Diag: %.4f %d" %(Correct_Diag/(Correct_Diag+Error_Diag), Correct_Diag))
# print("Error Diag: %.4f %d" %(Error_Diag/(Correct_Diag+Error_Diag), Error_Diag))

# print("FAR: %.4f" %(FAR))
# print("FRR: %.4f" %(FRR))
# print("DER: %.4f" %(DER))


df = pd.read_csv(f"{MODEL}/prediction.csv")
df["mispronunciations"] = MISPRONUNCIATIONS
df.to_csv(f"{MODEL}/prediction_final.csv", index=False)
