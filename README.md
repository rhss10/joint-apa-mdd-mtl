# A Joint Model for Pronunciation Assessment and Mispronunciation Detection and Diagnosis with Multi-task Learning

## Note

- Code for the paper **'A Joint Model for Pronunciation Assessment and Mispronunciation Detection and Diagnosis with Multi-task Learning'**, presented at Interspeech 2023
- The link to the paper: <https://www.isca-speech.org/archive/interspeech_2023/ryu23_interspeech.html>
- The paper and the code was extended to my Master's thesis (access will be available from October!), where MTL with RMSE+CTC and GoP features brought better results for APA task
- Regarding the license, please refer to the LICENSE.md
- If there are any problems, feel free to email me (<rhss10@snu.ac.kr>) or post an issue.

## Citation

If you find this repository useful, please cite our paper

```
@inproceedings{ryu23_interspeech,
  author={Hyungshin Ryu and Sunhee Kim and Minhwa Chung},
  title={{A Joint Model for Pronunciation Assessment and Mispronunciation Detection and Diagnosis with Multi-task Learning}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={959--963},
  doi={10.21437/Interspeech.2023-337}
}
```

## Code

- python version 3.8 (3.8.0, 3.8.16) was used for training and testing
- For all steps, don't forget to change the path to your own directories!
- For MDD evaluation, Kaldi needs to be installed beforehand to utilize SCTK toolkit. (<https://kaldi-asr.org/doc/install.html>)

1. Prepare the data

```bash
# first, make a csv list of 3 datasets (TIMIT, L2-arctic, Speechocean762)
cd data
python create_datasets.py
# then, make the data into huggingface datasets format, and create vocabulary set
python preprocess_datasets.py
```

2. Do an auxiliary fine-tuning on the SSL model for phone recognition task, to leverage extra knowledge transfer

```bash
cd auxiliary-phone-recognition
# fine-tune the baseline Wav2Vec2-large-robust for phone recognition with TIMIT train split (or other datasets!)
python trainer_train.py --exp_prefix TIM_robust
# or fine-tune HuBERT-large for phone recognition with TIMIT train split...
python trainer_train.py --model_name_or_path facebook/hubert-large-ll60k --exp_prefix TIM_hubert
# test the fine-tuned phone recognition model on TIMIT test split (or other datasets!)
python trainer_test.py TIM_robust_lr0.0001_warm0.1_type-linear >> trainer_test.log
```

2. Train a joint model of APA and MDD with multi-task learning

```bash
cd multi-task-learning
# jointly train APA and MDD, with a model fine-tuned for phone recognition (L1)
python train.py --exp_prefix Joint-CAPT-L1 --model_name_or_path /PATH_TO_YOUR_MODEL/trainer/TIM_robust_lr0.0001_warm0.1_type-linear
# or jointly train APA and MDD, with a raw SSL model (SSL)
python train.py --exp_prefix Joint-CAPT-SSL --model_name_or_path facebook/wav2vec2-large-robust --no_phone_recognition --enable_cls_epochs 50
# or train only on APA, with a raw SSL model (SSL)
python train.py --exp_prefix APA-SSL --model_name_or_path facebook/wav2vec2-large-robust --no_phone_recognition --cls_weight 1.0 --ctc_weight 0.0
```

3. test the model and do correlation analysis

```bash
cd ./test
sh test.sh [YOUR MTL/STL MODEL NAME]
python correlation.py

```
