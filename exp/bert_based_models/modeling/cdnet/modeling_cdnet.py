"""
@Time   :   2021-07-28 17:34:56
@File   :   modeling_cdnet.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""

import os
import operator
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from transformers import BertForMaskedLM
from bbcm.solver.losses import FocalLoss
from torch.optim.lr_scheduler import LambdaLR
from bbcm.engine.csc_trainer import CscTrainingModel
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

import pytorch_lightning as pl


class DetectionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )

        self.context_lstm = nn.GRU(  # or nn.LSTM
            input_size=config.hidden_size,  # 768
            hidden_size=config.hidden_size // 4,  # 192
            num_layers=1,
            bidirectional=True)

        self.eps = 1e-8
        self.scale = 10.  # [4.5398e-05, 9.9995e-01]
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.linear = nn.Linear(self.config.hidden_size, 2)  # [Faulty, Normal]

    def scaling(self, logit_vector):
        value_m = logit_vector.sum(-1).unsqueeze(-1)
        logit_vector = logit_vector / (value_m + self.eps) * self.scale
        return logit_vector
        
    def context_hidden(self, input_ids, token_type_ids=None, attention_mask=None):
        # [batch, sequence_length + 1], in case of right overflow
        am = attention_mask
        inp_ids = self.add_pad(input_ids)
        sequence_length = am.sum(-1)

        # ([batch], [batch])
        sent_indexes = torch.arange(am.shape[0], device=am.device)
        char_indexes = sequence_length.long().to(device=am.device)
        indexes = (sent_indexes, char_indexes)
        input_ids_with_sep = inp_ids.index_put_(
            indexes, torch.tensor(102, device=am.device))

        # [batch, sequence_length w/ [CLS] [SEP], embedding_size]
        embeddings = self.bert.embeddings(
            input_ids=input_ids_with_sep,  # self.add_sep(input_ids),
            token_type_ids=token_type_ids)

        # the LSTM part
        packed = pack(embeddings, sequence_length + 1,  # add [SEP]
                      batch_first=True, enforce_sorted=False)
        token_hidden, (h_n, c_n) = self.context_lstm(packed)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = unpack(token_hidden, batch_first=True)[0]

        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = token_hidden.view(token_hidden.shape[0], token_hidden.shape[1], 2, -1)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size] * 2 directions
        return token_hidden[:, :, 0], token_hidden[:, :, 1]

    def init_embeddings_from_corrector_state_dict(self, state_dict):
        embedding_state_dict = OrderedDict({
            name: value for name, value in state_dict.items()
            if 'embedding' in name})
        self.load_state_dict(embedding_state_dict, strict=False)

    def forward(self, hidden_states):
        out, _ = self.gru(hidden_states)
        prob = self.linear(out)
        prob = self.sigmoid(prob)
        return prob



class BertForCsc(CscTrainingModel):
    # LSTM detection
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        # self.detection = nn.Linear(self.bert.config.hidden_size, 2)  # [Faulty, Normal]
        self.detection = DetectionNetwork(cfg)
        self.eps = 1e-8
        self.scale = 10.  # [4.5398e-05, 9.9995e-01]
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.tokenizer = tokenizer

    def scaling(self, logit_vector):
        value_m = logit_vector.sum(-1).unsqueeze(-1)
        logit_vector = logit_vector / (value_m + self.eps) * self.scale
        return logit_vector

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self.device)
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.device)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])  # <= no higher features
        prob = self.scaling(prob)  # scaling

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob[1], bert_outputs.logits)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.softmax(prob)[0].squeeze(-1),
                       bert_outputs.logits)
        return outputs

