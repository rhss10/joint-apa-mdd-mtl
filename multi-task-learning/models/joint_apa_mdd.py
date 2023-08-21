from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    AutoModelForCTC,
    HubertForCTC,
    HubertModel,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    WavLMForCTC,
    WavLMModel,
)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x


class JointAPAMDDWavLM(WavLMForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params
        self.embedding_size = 128
        self.wavlm = WavLMModel(config)

        self.dropout = nn.Dropout(config.final_dropout)
        self.projector = nn.LSTM(config.hidden_size, config.classifier_proj_size // 2, batch_first=True, bidirectional=True)

        self.tot_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.pros_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.flu_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.acc_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.enable_cls = True

    def forward(
        self,
        input_values,
        input_lengths=None,
        cls_labels=None,
        ctc_labels=None,
    ):
        outputs = self.wavlm(input_values)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        cls_hidden_states, _ = self.projector(hidden_states)

        # Classification
        tot_logits = self.tot_head(cls_hidden_states)
        pros_logits = self.pros_head(cls_hidden_states)
        flu_logits = self.flu_head(cls_hidden_states)
        acc_logits = self.acc_head(cls_hidden_states)

        cls_logits = torch.stack((tot_logits, pros_logits, flu_logits, acc_logits), dim=-1)
        cls_loss = CrossEntropyLoss()(cls_logits, cls_labels)

        # CTC
        # retrieve loss input_lengths from attention_mask
        ctc_attention_mask = torch.ones_like(input_values, dtype=torch.long)
        ctc_input_lengths = self._get_feat_extract_output_lengths(ctc_attention_mask.sum(-1)).to(torch.long)

        ctc_logits = self.lm_head(hidden_states)
        # assuming that padded tokens are filled with -100 when not being attended to
        ctc_labels_mask = ctc_labels >= 0
        ctc_target_lengths = ctc_labels_mask.sum(-1)
        ctc_flattened_targets = ctc_labels.masked_select(ctc_labels_mask)

        ctc_log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                ctc_log_probs,
                ctc_flattened_targets,
                ctc_input_lengths,
                ctc_target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        # Final loss
        if self.enable_cls:
            loss = self.cfg["cls_weight"] * cls_loss + self.cfg["ctc_weight"] * ctc_loss
        else:
            loss = self.cfg["ctc_weight"] * ctc_loss

        return (
            loss,
            cls_loss,
            ctc_loss,
            hidden_states,
            cls_logits,
            ctc_logits,
        )


class JointAPAMDDHubert(HubertForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params
        self.embedding_size = 128
        self.hubert = HubertModel(config)

        self.dropout = nn.Dropout(config.final_dropout)
        self.projector = nn.LSTM(config.hidden_size, config.classifier_proj_size // 2, batch_first=True, bidirectional=True)

        self.tot_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.pros_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.flu_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.acc_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.enable_cls = True

    def forward(
        self,
        input_values,
        input_lengths=None,
        cls_labels=None,
        ctc_labels=None,
    ):
        # print(input_values)
        outputs = self.hubert(input_values)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        cls_hidden_states, _ = self.projector(hidden_states)

        # Classification
        tot_logits = self.tot_head(cls_hidden_states)
        pros_logits = self.pros_head(cls_hidden_states)
        flu_logits = self.flu_head(cls_hidden_states)
        acc_logits = self.acc_head(cls_hidden_states)

        cls_logits = torch.stack((tot_logits, pros_logits, flu_logits, acc_logits), dim=-1)
        cls_loss = CrossEntropyLoss()(cls_logits, cls_labels)

        # CTC
        # retrieve loss input_lengths from attention_mask
        ctc_attention_mask = torch.ones_like(input_values, dtype=torch.long)
        ctc_input_lengths = self._get_feat_extract_output_lengths(ctc_attention_mask.sum(-1)).to(torch.long)

        ctc_logits = self.lm_head(hidden_states)
        # assuming that padded tokens are filled with -100
        # when not being attended to
        ctc_labels_mask = ctc_labels >= 0
        ctc_target_lengths = ctc_labels_mask.sum(-1)
        ctc_flattened_targets = ctc_labels.masked_select(ctc_labels_mask)

        ctc_log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                ctc_log_probs,
                ctc_flattened_targets,
                ctc_input_lengths,
                ctc_target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        # Final loss
        if self.enable_cls:
            loss = self.cfg["cls_weight"] * cls_loss + self.cfg["ctc_weight"] * ctc_loss
        else:
            loss = self.cfg["ctc_weight"] * ctc_loss

        return (
            loss,
            cls_loss,
            ctc_loss,
            hidden_states,
            cls_logits,
            ctc_logits,
        )


class JointAPAMDD(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params
        self.embedding_size = 128
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.projector = nn.LSTM(config.hidden_size, config.classifier_proj_size // 2, batch_first=True, bidirectional=True)

        self.tot_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.pros_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.flu_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.acc_head = MLP(config.classifier_proj_size, self.cfg["num_classes"])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.enable_cls = True

    def forward(
        self,
        input_values,
        input_lengths,
        cls_labels,
        ctc_labels,
    ):
        # print(input_values)
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        cls_hidden_states, _ = self.projector(hidden_states)

        # Classification
        # cls_pooled_output = cls_hidden_states.mean(dim=1)
        tot_logits = self.tot_head(cls_hidden_states)
        pros_logits = self.pros_head(cls_hidden_states)
        flu_logits = self.flu_head(cls_hidden_states)
        acc_logits = self.acc_head(cls_hidden_states)

        cls_logits = torch.stack((tot_logits, pros_logits, flu_logits, acc_logits), dim=-1)
        cls_loss = CrossEntropyLoss()(cls_logits, cls_labels)

        # CTC
        # retrieve loss input_lengths from attention_mask
        ctc_attention_mask = torch.ones_like(input_values, dtype=torch.long)
        ctc_input_lengths = self._get_feat_extract_output_lengths(ctc_attention_mask.sum(-1)).to(torch.long)

        ctc_logits = self.lm_head(hidden_states)
        # assuming that padded tokens are filled with -100 when not being attended to
        ctc_labels_mask = ctc_labels >= 0
        ctc_target_lengths = ctc_labels_mask.sum(-1)
        ctc_flattened_targets = ctc_labels.masked_select(ctc_labels_mask)

        ctc_log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                ctc_log_probs,
                ctc_flattened_targets,
                ctc_input_lengths,
                ctc_target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        # Final loss
        if self.enable_cls:
            loss = self.cfg["cls_weight"] * cls_loss + self.cfg["ctc_weight"] * ctc_loss
        else:
            loss = self.cfg["ctc_weight"] * ctc_loss

        return (
            loss,
            cls_loss,
            ctc_loss,
            hidden_states,
            cls_logits,
            ctc_logits,
        )
