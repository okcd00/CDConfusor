"""
@Time   :   2021-08-25 16:00:59
@File   :   modeling_cdsmb.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""
import operator
import os
from collections import OrderedDict
import transformers as tfs
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead
from transformers.modeling_utils import ModuleUtilsMixin
from bbcm.engine.csc_trainer import CscTrainingModel
from bbcm.solver.losses import FocalLoss
import numpy as np


class DetectionNetwork(nn.Module):
    def __init__(self, config, deeper=0, wider=1):
        super().__init__()
        self.eps = 1e-8  
        self.scale = 10  # [4.5398e-05, 9.9995e-01]
        self.config = config
        self.embeddings = BertEmbeddings(self.config)
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2 * wider,
            num_layers=3 if deeper else 2,  # wider > deeper + wider ≈ deeper
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.linear = nn.Linear(
            self.config.hidden_size * wider, 
            2  # [Normal, Faulty] 
        )

    def get_embeddings(self, encoded_texts):
        return self.embeddings(
            input_ids=encoded_texts['input_ids'],
            token_type_ids=encoded_texts['token_type_ids'])

    def scaling(self, logit_vector):
        # make tensors in the last dimention have the same sum value (scale value).
        value_m = logit_vector.sum(-1).unsqueeze(-1)
        logit_vector = logit_vector / (value_m + self.eps) * self.scale
        return logit_vector

    def forward(self, hidden_states):
        out, _ = self.gru(hidden_states)
        logits = self.linear(out)
        logits = self.scaling(logits)  # scaling = non-scaling: 
        prob = self.softmax(logits)
        return logits[..., 1], prob[..., 1]   # prob = p(x_i is wrong)

    def init_embeddings_from_corrector_state_dict(self, state_dict):
        embedding_state_dict = OrderedDict({
            name: value for name, value in state_dict.items()
            if 'embedding' in name})
        self.load_state_dict(embedding_state_dict, strict=False)


class BertCorrectionModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(self.config)
        self.corrector = BertEncoder(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls = BertOnlyMLMHead(self.config)
        self.kl_loss = nn.functional.kl_div
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # 0 means [PAD]
        self._device = device
        # torch.autograd.set_detect_anomaly(True)

        # for the similarity case
        self.consider_similarity = True
        self.similarity_ignore_target_position = True
        self.loss_case_weight = [.05, 1.]  # [sim_loss, cor_loss]
        self.min_retention_weight = None

    def get_encoded_texts(self, texts):
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        return encoded_texts

    def get_senquence_output(self, encoded_texts, hidden_states, output_attentions=False):
        input_shape = encoded_texts['input_ids'].size()
        device = encoded_texts['input_ids'].device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(encoded_texts['attention_mask'],
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.corrector(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        if output_attentions:
            attention_scores = encoder_outputs[-1]
            return sequence_output, attention_scores
        return sequence_output

    def show_attention_matrix(self, model, tokenizer, sentence_a, layer=None, heads=None):
        inputs = tokenizer.encode_plus(sentence_a, None, return_tensors='pt')
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        # sentence_b_start = token_type_ids[0].tolist().index(1)
        input_id_list = input_ids[0].tolist() # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

        import matplotlib.pyplot as plt
        # mean on layers, see 12 heads
        # scores = (torch.cat(attention, dim=0).mean(dim=0))
        # mean on heads, see 12 layers
        scores = (torch.cat(attention, dim=0).mean(dim=1))
        # single layer 
        # scores = attention[3][0]  
        # print(scores.shape)
        scores = scores.detach().numpy()

        plt.figure(figsize=(20, 12))

        for i in range(12):
            ax = plt.subplot(3,4,i+1)
            plt.sca(ax)
            plt.imshow(scores[i], cmap=plt.cm.hot, vmin=0, vmax=0.2)
            plt.colorbar()
        plt.show()

        """
        from bertviz import head_view, model_view
        inputs = tokenizer.encode_plus(
            sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention = model(input_ids)[-1]
        model_view(attention=attention, tokens=tokens)
        head_view(attention=attention, tokens=tokens)
        """
        return scores

    def calculate_kl_loss(self, pred_dist, truth_dist):
        return self.kl_loss(pred_dist.softmax(dim=-1).log(), 
                            truth_dist.softmax(dim=-1), reduction='sum')

    @staticmethod
    def calculate_similarity(pred_dist, truth_dist, 
                             attention_mask=None, context_mask=None, method='cosine'):
        # dist: [batch_size, sequence_length, hidden_size=768]
        # attn: [batch_size, sequence_length]
        # ctx: [batch_size, sequence_length]
        _p, _t = pred_dist, truth_dist
        if attention_mask is not None:
            _p = _p * attention_mask.unsqueeze(-1)
            _t = _t * attention_mask.unsqueeze(-1)
        if context_mask is not None:
            _p = _p * context_mask.unsqueeze(-1)
            _t = _t * context_mask.unsqueeze(-1)
        
        # the detached `_t` here is a soft-label in distilling.
        return torch.cosine_similarity(_p.view(1, -1), _t.view(1, -1).detach())

    def calculate_sim_loss(self, pred_texts, truth_texts, pred_dist, truth_dist):
        attention_mask = pred_texts['attention_mask']
        # token_type_ids = pred_texts['token_type_ids']

        context_mask = None
        if self.similarity_ignore_target_position:
            context_mask = pred_texts['input_ids'] == truth_texts['input_ids']

        # cosine_sim_loss = 1 - cosine_sim
        # print(pred_dist.shape, attention_mask.shape, context_mask.shape)
        sim_loss = 1. - self.calculate_similarity(  
            pred_dist, truth_dist, 
            attention_mask=attention_mask,
            context_mask=context_mask)
        return sim_loss

    def forward(self, texts, prob=None, embed=None, cor_labels=None, 
                residual_connection=False, consider_similarity=None):
        # original texts
        encoded_texts = self.get_encoded_texts(texts)
        token_type_ids = encoded_texts['token_type_ids']
        if embed is None:
            embed = self.embeddings(input_ids=encoded_texts['input_ids'],
                                    token_type_ids=token_type_ids)
        
        # 此处较原文有一定改动，改动意在完整保留带有 type_ids 及 position_ids 影响的 embedding
        mask_embed = self.embeddings(
            input_ids=torch.ones_like(
                encoded_texts['input_ids']).long() * self.mask_token_id,
            token_type_ids=token_type_ids).detach()
        # 此处的原文实现为：
        # mask_embed = self.embeddings(torch.tensor([[self.mask_token_id]], device=self._device)).detach()

        # [batch_size, sequence_length, hidden_size=768]
        # prob = p(x_i is wrong), so we use `prob*emb([MASK]) + (1-prob)*emb(x_i)`
        if prob is None:
            fusion_hidden_states = embed
        elif self.min_retention_weight is not None:
            scaled_prob = prob * (1. - self.min_retention_weight)
            fusion_hidden_states = scaled_prob * mask_embed + \
                (1. - scaled_prob) * embed
        else:
            fusion_hidden_states = prob * mask_embed + (1. - prob) * embed
        sequence_output = self.get_senquence_output(
            encoded_texts=encoded_texts, 
            hidden_states=fusion_hidden_states)

        # add residual connection
        if residual_connection:
            sequence_output = sequence_output + embed
        prediction_scores = self.cls(sequence_output)

        # Masked language modeling softmax layer
        if cor_labels is not None:

            # correct texts
            encoded_cor_texts = self.get_encoded_texts(cor_labels)
            text_labels = encoded_cor_texts['input_ids'].detach()
            
            # shorten similarity between each sentence in pairs
            sim_loss = 0.
            if consider_similarity is None:
                consider_similarity = self.consider_similarity
            if consider_similarity:
                pred_dist = self.get_senquence_output(
                    encoded_texts=encoded_texts, 
                    hidden_states=embed)
                truth_dist = self.get_senquence_output(
                    encoded_texts=encoded_cor_texts, 
                    hidden_states=self.embeddings(input_ids=text_labels, 
                                                  token_type_ids=token_type_ids))
                sim_loss = self.calculate_sim_loss(
                    pred_texts=encoded_texts,
                    truth_texts=encoded_cor_texts,
                    pred_dist=pred_dist,
                    truth_dist=truth_dist)

            # torch 的 cross entropy loss 计算时会忽略 label 为 -100 -> 0 
            # text_labels_y[text_labels_y == 0] = -100  # -> 0
            text_labels_y = encoded_cor_texts['input_ids'].detach()

            # mlm loss with cross-entropy
            # prediction_scores is the logits for correction
            cor_loss = self.ce_loss(prediction_scores.view(-1, self.config.vocab_size), 
                                    text_labels_y.view(-1))

            # weighted loss
            if self.loss_case_weight is None:
                loss = sim_loss + cor_loss
            else:
                loss = sim_loss * self.loss_case_weight[0] + \
                    cor_loss * self.loss_case_weight[1]
            return (loss, prediction_scores, sequence_output)
        # (cor_loss,) prediction_scores, sequence_output
        return (prediction_scores, sequence_output)

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = tfs.AutoModelForMaskedLM.from_pretrained(gen_fp).state_dict()
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)
        return state_dict


class SoftMaskedBertModel(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.config = tfs.AutoConfig.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.tokenizer = tokenizer
        self._device = cfg.MODEL.DEVICE

        # for the similarity case
        self.consider_similarity_det = True
        self.consider_similarity_cor = True
        self.similarity_ignore_target_position = True
        self.loss_case_weight_det = [.25, 1.]  # [sim_loss, det_loss]
        self.loss_case_weight_cor = [.05, 1.]  # [sim_loss, cor_loss]

        # init corrector from BERT
        self.corrector = BertCorrectionModel(self.config, tokenizer, cfg.MODEL.DEVICE)
        self.corrector.consider_similarity = self.consider_similarity_cor
        self.corrector.loss_case_weight = self.loss_case_weight_cor
        self.corrector.min_retention_weight = 0.1  # None

        # init detector with corrector's embeddings
        self.predict_pinyins = False
        self.detector = DetectionNetwork(self.config)
        self.load_from_transformers_state_dict(gen_fp=self.cfg.MODEL.BERT_CKPT)

        # loss function
        self.det_loss_fn = 'bce'

        # activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        return encoded_texts

    def calculate_detection_loss(self, logits, prob, encoded_texts, det_labels, loss_function='bce'):
        if loss_function in ['bce']:
            det_prob = prob.squeeze(-1)
            det_loss_fct = nn.BCELoss()
            _labels = det_labels.double()
        else:  # focal
            # det_prob = self.sigmoid(logits).squeeze(-1)
            det_prob = logits.squeeze(-1)
            det_loss_fct = FocalLoss(
                num_labels=None, activation_type='sigmoid')
            _labels = det_labels
        
        # pad部分不计算损失
        active_loss = encoded_texts['attention_mask'].view(-1, prob.shape[1]) == 1
        active_probs = prob.view(-1, prob.shape[1])[active_loss]
        active_labels = _labels[active_loss]
        det_loss = det_loss_fct(active_probs, active_labels.float())
        return det_loss, det_prob

    def calculate_sim_loss(self, pred_texts, truth_texts, pred_dist, truth_dist):
        attention_mask = pred_texts['attention_mask']
        context_mask = None
        if self.similarity_ignore_target_position:
            context_mask = pred_texts['input_ids'] == truth_texts['input_ids']

        # cosine_sim_loss = 1 - cosine_sim
        sim_loss = 1. - self.calculate_similarity(  
            pred_dist, truth_dist, 
            attention_mask=attention_mask,
            context_mask=context_mask)
        return sim_loss

    def forward(self, texts, cor_labels=None, det_labels=None):
        encoded_texts = self.get_encoded_texts(texts)

        # detectors' embedding refers from corrector
        # embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],
        #                                   token_type_ids=encoded_texts['token_type_ids'])
        det_embeddings = self.detector.get_embeddings(
            encoded_texts=encoded_texts)
        logits, prob = self.detector(
            hidden_states=det_embeddings)  

        # get outputs from the corrector
        # (cor_loss,) prediction_scores, sequence_output
        cor_out = self.corrector(
            texts, 
            prob=prob.unsqueeze(-1), 
            embed=det_embeddings, 
            cor_labels=cor_labels, 
            residual_connection=True,
            consider_similarity=self.consider_similarity_cor)

        # det_labels are unknown during inference
        if det_labels is not None:  
            det_loss, det_prob = self.calculate_detection_loss(
                logits=logits, prob=prob, 
                encoded_texts=encoded_texts,
                det_labels=det_labels,
                loss_function=self.det_loss_fn)

            if self.consider_similarity_det and cor_labels is not None:
                encoded_truth_texts = self.get_encoded_texts(texts)
                truth_embeddings = self.detector.get_embeddings(
                    encoded_texts=encoded_truth_texts)
                truth_logits, truth_prob = self.detector(
                    hidden_states=truth_embeddings)
                sim_loss_det = self.corrector.calculate_sim_loss(
                    pred_texts=encoded_texts,
                    truth_texts=encoded_truth_texts,
                    pred_dist=logits.unsqueeze(-1), 
                    truth_dist=truth_logits.unsqueeze(-1))
                det_loss = sim_loss_det * self.loss_case_weight_det[0] + det_loss * self.loss_case_weight_det[1]

            # (cor_loss,) prediction_scores, sequence_output
            cor_loss, prediction_scores, sequence_output = cor_out

            # 检错loss，纠错loss，检错 logits 输出，纠错 logits 输出，纠错 hidden
            outputs = (
                det_loss, cor_loss, 
                logits, prediction_scores, sequence_output)
        else:
            prediction_scores, sequence_output = cor_out
            cor_prob = self.softmax(prediction_scores)
            # 检错 prob，纠错 prob
            outputs = (prob.squeeze(-1), cor_prob)

        return outputs

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        state_dict = self.corrector.load_from_transformers_state_dict(gen_fp)
        self.detector.init_embeddings_from_corrector_state_dict(state_dict)
