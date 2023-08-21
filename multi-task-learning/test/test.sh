#!/usr/bin/env bash
MODEL=$1

mkdir ${MODEL}
python test.py --model_name_or_path /PATH_TO_THE_MODEL/${MODEL} > ${MODEL}/${MODEL}.log
sh kaldi-align.sh ${MODEL}
python ins_del_sub_cor_analysis.py ${MODEL}
echo "- MDD analysis finished"

echo "- Test finished"