"""
@Time   :   2021-01-22 11:42:52
@File   :   bert4csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import torch
import torch.nn as nn
from transformers import BertForMaskedLM

from abc import ABC
from bbcm.engine.csc_trainer import CscTrainingModel
from bbcm.solver.losses import FocalLoss


class MacBert4Csc(CscTrainingModel, ABC):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels[text_labels == 0] = -100  # -100计算损失时会忽略
            text_labels = text_labels.to(self.device)
        else:
            text_labels = None
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.device)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
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
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits)
        return outputs


class BertForCsc(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.tokenizer = tokenizer

    def generate_det_labels(self, original_token_ids, correct_token_ids):
        return (original_token_ids != correct_token_ids).long()

    def forward_old(self, texts, cor_labels=None, det_labels=None, loss_function='focal'):
        # generate text_labels
        need_det_labels = False
        if cor_labels is not None:
            cor_text = self.tokenizer(cor_labels, padding=True, return_tensors='pt')
            text_labels = cor_text['input_ids']
            text_labels = text_labels.to(self.device)
            text_labels[text_labels == 0] = -100
            need_det_labels = True
        else:
            text_labels = None

        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.device)
        bert_outputs = self.bert(
            **encoded_text, labels=text_labels, 
            return_dict=True, output_hidden_states=True)
        
        if need_det_labels and det_labels is None:
            det_labels = self.generate_det_labels(
                encoded_text['input_ids'], text_labels)

        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            # select loss function
            if loss_function in ['bce']:
                prob = self.sigmoid(prob)
                det_outputs = prob.squeeze(-1)
                det_loss_fct = nn.BCELoss()
            else:  # focal
                det_outputs = self.sigmoid(prob).squeeze(-1)
                det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')

            # pad部分不计算损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())

            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       det_outputs,
                       bert_outputs.logits)
        return outputs

    def forward(self, texts, cor_labels=None, det_labels=None, loss_function='focal'):
        # generate text_labels
        need_det_labels = False
        if cor_labels is not None:
            cor_text = self.tokenizer(cor_labels, padding=True, return_tensors='pt')
            text_labels = cor_text['input_ids']
            text_labels = text_labels.to(self.device)
            text_labels[text_labels == 0] = -100
            need_det_labels = True
        else:
            text_labels = None

        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.device)
        bert_outputs = self.bert(
            **encoded_text, labels=text_labels, 
            return_dict=True, output_hidden_states=True)
        
        if need_det_labels and det_labels is None:
            det_labels = self.generate_det_labels(
                encoded_text['input_ids'], text_labels)

        # 检错概率
        det_logits = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错 prob，纠错 prob
            cor_prob = self.softmax(bert_outputs.logits)
            det_prob = self.sigmoid(det_logits)
            outputs = (det_prob, cor_prob)
        else:
            # select loss function
            if loss_function in ['bce']:
                det_prob = self.sigmoid(det_logits).squeeze(-1)
                det_loss_fct = nn.BCELoss()
            else:  # focal
                det_prob = self.sigmoid(det_logits).squeeze(-1)
                det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')

            # pad部分不计算损失
            active_loss = encoded_text['attention_mask'].view(-1, det_prob.shape[1]) == 1
            active_probs = det_prob.view(-1, det_prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())

            # 检错loss，纠错loss，检错 logits，纠错 logits
            outputs = (det_loss,
                       bert_outputs.loss,
                       det_logits,
                       bert_outputs.logits)
        return outputs