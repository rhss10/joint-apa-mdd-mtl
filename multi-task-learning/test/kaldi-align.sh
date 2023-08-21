#!/usr/bin/env bash
MODEL=$1

echo "- Model: ${MODEL}\n"
# align phones using Kaldi SCTK toolkit
mv ${MODEL}/ANNOT* /home/YOUR_DIRECTORY/kaldi/egs/wsj/s5
mv ${MODEL}/CANON* /home/YOUR_DIRECTORY/kaldi/egs/wsj/s5
mv ${MODEL}/PREDS* /home/YOUR_DIRECTORY/kaldi/egs/wsj/s5
cd /home/YOUR_DIRECTORY/kaldi/egs/wsj/s5
. ./path.sh

align-text --special-symbol='***' ark:CANON_${MODEL} ark:ANNOT_${MODEL} \
ark,t:- | utils/scoring/wer_per_utt_details.pl --special-symbol='***' > CANON_ANNOT_align
echo "- Canon-Annotation Alignment finished\n"

align-text --special-symbol='***' ark:ANNOT_${MODEL} \
ark:PREDS_${MODEL} ark,t:- | utils/scoring/wer_per_utt_details.pl --special-symbol='***' > ANNOT_PREDS_align
echo "- Annotation-Prediction Alignment finished\n"

align-text --special-symbol='***' ark:CANON_${MODEL} \
ark:PREDS_${MODEL} ark,t:- | utils/scoring/wer_per_utt_details.pl --special-symbol='***' > CANON_PREDS_align
echo "- Canon-Prediction Alignment finished\n"

# save it to the original directory
mv ANNOT* /PATH_TO_REPO/multi-task-learning/test/${MODEL}
mv CANON* /PATH_TO_REPO/multi-task-learning/test/${MODEL}
mv PREDS* /PATH_TO_REPO/multi-task-learning/test/${MODEL}
echo "- Alignment finished\n"

