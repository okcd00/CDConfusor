"""
@Time   :   2022-03-15 11:57:00
@File   :   modeling_cdmac.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""

import os
import sys
sys.path.append("/home/user/bbcm")
sys.path.append("/home/user/BBCM")

import time
import torch
torch.backends.cudnn.benchmark = True

import pickle
import numpy as np
import torch.nn as nn
import transformers as tfs
from transformers import BertForMaskedLM

from bbcm.solver.losses import LOSS_FN_LIB
from bbcm.engine.csc_trainer import CscTrainingModel
from bbcm.data.loaders.collator import DataCollatorForCsc


class CDMacBertForCsc(CscTrainingModel):
    timer = []

    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.tokenizer = tokenizer
        # self._device = cfg.MODEL.DEVICE
        self.cor_weight = cfg.MODEL.HYPER_PARAMS[0]
        self.share_bert = cfg.MODEL.SHARE_BERT  # det/cor share the BERT

        # pre-defined loss functions.
        self.loss_fn = LOSS_FN_LIB

        # generate a mask on prediction as a hard confusion set.
        self.mask_ignore_pos = cfg.MODEL.MASK_IGNORE_POS
        self.mask_confusions = cfg.MODEL.MASK_CONFUSIONS
        self.predict_pinyins = cfg.MODEL.PREDICT_PINYINS
        self.utilize_pinyin_for = {'det': False, 'cor': True}

        self.lazy_load_done = False
        self.mask_value = torch.tensor(-10000.).to(self._device)
        if self.mask_confusions:  # mask generation in GPU
            self.path_to_ps_mapping_case = os.path.join(
                cfg.MODEL.DATA_PATH, 'ps_mapping_case.pkl')
            # for generating pinyin-similar mask
            ps_mapping_case = pickle.load(open(self.path_to_ps_mapping_case, 'rb'))
            # vocab_size, pinyin_vocab_size
            self.char2py_mapping = torch.from_numpy(ps_mapping_case['c2p'])#.bool()
            self.char2py_mapping = self.char2py_mapping.to(self._device)
            # pinyin_vocab_size, pinyin_vocab_size
            self.py2pys_mapping = torch.from_numpy(ps_mapping_case['p2p']).float()
            self.py2pys_mapping = self.py2pys_mapping.to(self._device)
            # pinyin_vocab_size, vocab_size
            self.py2char_mapping = self.char2py_mapping.T.float()
            # self.py2char_mapping = torch.from_numpy(ps_mapping_case['p2c']).to(self._device)

            # print('ps_mapping_case is now loading on', self._device, 
            #     self.char2py_mapping.shape, self.py2pys_mapping.shape)

        need_pinyin_flag = True in (
            list(self.utilize_pinyin_for.values()) + [self.predict_pinyins])
        self.collator = DataCollatorForCsc(
            tokenizer=self.tokenizer,
            augmentation=False,
            need_pinyin=need_pinyin_flag,
            cfg=self.cfg)
        self.n_pinyin = len(self.collator.vocab_pinyin)

        # both correction & detection need BERT as the encoder.
        self.bert_type = cfg.MODEL.ENCODER_TYPE
        if self.bert_type in ['base']:
            self.bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        elif self.bert_type in ['merged', 'pinyin_merged']:
            self.merge_mask_type = cfg.MODEL.MERGE_MASK_TYPE
            # mask version of BERT for pinyin-features' fusion
            from bbcm.modeling.cdnet.modeling_cdpmb import BertCorrectionModel
            self.bert_config = tfs.AutoConfig.from_pretrained(cfg.MODEL.BERT_CKPT)
            self.bert = BertCorrectionModel(
                self.bert_config, self.tokenizer, device=self._device)
            self.bert.load_from_transformers_state_dict(cfg.MODEL.BERT_CKPT)

            # corrector uses another BERT
            if not self.share_bert: 
                self.second_stage_bert = BertCorrectionModel(
                    self.bert_config, self.tokenizer, device=self._device)
                self.second_stage_bert.load_from_transformers_state_dict(
                    cfg.MODEL.BERT_CKPT, bert_name='second_stage_bert')

            if self.bert_type in ['pinyin_merged']:
                # init the pinyin_embeddings layer
                MSK_INDEX = self.collator.vocab_chars.index('[MASK]')  # 103
                MSK_EMB = self.bert.embeddings.word_embeddings.weight[MSK_INDEX].clone()
                MSK_EMB = MSK_EMB.to(self._device)
                self.pinyin_embeddings = nn.Embedding(self.collator.pinyin_vocab_size,
                                                      self.bert.config.hidden_size,
                                                      device=self._device)
                self.pinyin_embeddings.weight[1:-1].data.copy_(MSK_EMB)
                self.pinyin_embeddings.requires_grad_()
                if False:
                    self.conv1d = nn.Conv1d(in_channels=self.bert.config.hidden_size,
                                            out_channels=self.bert.config.hidden_size,
                                            kernel_size=3, padding=1, device=self._device)  
                self.layer_norm = nn.LayerNorm(
                    self.bert.config.hidden_size, 
                    self.cfg.MODEL.EPS,
                    device=self._device)
        else:
            raise ValueError("Invalid bert type:", self.bert_type)

        # predict faulty positions as the detection loss (and outputs)
        self.predict_faulty_positions = cfg.MODEL.PREDICT_FAULTY_POSITIONS
        self.topk_samples = 1  # negative sampling for detection module

        self.input_type_for_detection = 'bert'
        if self.input_type_for_detection.startswith('emb'):
            wider_gru = 3  # how much wider the GRU module needs
            self.faulty_position_gru = nn.GRU(
                self.bert.config.hidden_size,
                self.bert.config.hidden_size // 2 * wider_gru,
                num_layers=2,
                batch_first=True,
                dropout=0.25,
                bidirectional=True)

            self.faulty_position_ffn = self.construct_ffn(
                hidden_size=self.bert.config.hidden_size * wider_gru, 
                intermediate_size=self.bert.config.hidden_size // 2,  # -1
                n_out=1)  # [Normal, Faulty] 

            self.detection = self._detection
        else:  # 'bert
            # Bert-output + single layer LC
            self.detection = nn.Linear(
                self.bert.config.hidden_size, 1)  # [0=Normal, 1=Faulty] 
        self.det_loss_fn = 'bce'
        
        # also predict correct pinyins as an auxiliary loss.
        # nn.Linear(self.bert.config.hidden_size, self.n_pinyin)
        if self.predict_pinyins:
            self.pinyin_predictor_ffn = self.construct_ffn(
                hidden_size=self.bert.config.hidden_size,
                intermediate_size=self.bert.config.hidden_size,
                n_out=self.n_pinyin)
            self.pinyin_predictor = self._pronunciation
            self.pinyin_loss_fn = 'ce'
        
        # for the similarity case
        self.consider_similarity = {'det': False, 'cor': False}
        self.similarity_ignore_target_position = True
        self.loss_case_weight_det = [1., .25, 1.]  # [det_loss, sim_loss, pinyin_loss]
        self.loss_case_weight_cor = [1., .25, 1.]  # [cor_loss, sim_loss, pinyin_loss]
        self.min_retention_weight = None  # 0.1  # None

        # activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
    
    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        if self.debug:
            self.timer.append((information, time.time()))

    def show_timer(self):
        if self.timer:
            phase, start_t = self.timer[0]
            print(phase, time.strftime("%H:%M:%S", time.gmtime(start_t)))
            for phase, t in self.timer[1:]:
                print(phase, '+', t - start_t)

    def construct_ffn(self, hidden_size, intermediate_size=None, n_out=1):
        if intermediate_size is None:
            intermediate_size = hidden_size // 2
        
        return nn.ModuleList([  # for classification
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, n_out)])

    def _detection(self, hidden_states):
        det_logits, _ = self.faulty_position_gru(hidden_states)
        for _layer in self.faulty_position_ffn:
            det_logits = _layer(det_logits)
        # det_prob = self.sigmoid(det_logits)
        # det_pred = det_prob > 0.5

        # (batch, sequence_length, 1)
        return det_logits  #, det_prob, det_pred
    
    def _pronunciation(self, hidden_states):
        pinyin_logits = hidden_states
        for _layer in self.pinyin_predictor_ffn:
            pinyin_logits = _layer(pinyin_logits)
        pinyin_prob = self.softmax(pinyin_logits)
        return pinyin_logits, pinyin_prob

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts = encoded_texts.to(self._device)
        return encoded_texts

    def get_pinyin_embeddings(self, pinyin_inputs):
        # (batch_size, sequence_length, hidden_size)
        embeddings = self.pinyin_embeddings(pinyin_inputs)
        return embeddings

    def get_bert_embeddings(self, encoded):
        embeddings = None
        if self.bert_type in ['base']:
            embeddings = self.bert.bert.embeddings(
                input_ids=encoded['input_ids'],
                token_type_ids=encoded['token_type_ids'])
        elif self.bert_type in ['merged', 'pinyin_merged']:
            embeddings = self.bert.embeddings(
                input_ids=encoded['input_ids'],
                token_type_ids=encoded['token_type_ids'])
        else: 
            raise ValueError("Invalid bert_type for get_bert_embeddings:", 
                             self.bert_type)
        return embeddings

    def get_bert_outputs(
            self, encoded, labels, 
            prob=None, embed=None, py_embed=None,
            residual_connection=False, bert_model=None):
        outputs = None
        _labels = labels
        if labels is not None:
            _labels = _labels.detach() 
        if bert_model is None:
            bert_model = self.bert
        if self.bert_type in ['base']:
            outputs = bert_model(
                **encoded, labels=_labels, 
                return_dict=True, output_hidden_states=True)
            bert_hidden = outputs.hidden_states[-1]
            bert_logits = outputs.logits
            bert_loss = outputs.loss
        elif self.bert_type in ['merged', 'pinyin_merged']:
            if embed is None:
                embed = self.get_bert_embeddings(encoded)
            outputs = bert_model(
                encoded_err=encoded, prob=prob, 
                embed=embed, py_embed=py_embed, 
                text_labels=_labels,
                residual_connection=residual_connection)
            bert_loss, bert_logits, bert_hidden = outputs

        # logits: (batch_size, sequence_length, vocab_size)
        # hidden: (batch_size, sequence_length, hidden_size)
        return bert_loss, bert_logits, bert_hidden

    @staticmethod
    def calculate_similarity(pred_dist, truth_dist, 
                             attention_mask=None, context_mask=None, method='cosine'):
        # dist: [batch_size, sequence_length, hidden_size=768]
        # attn: [batch_size, sequence_length]
        # ctx: [batch_size, sequence_length]
        _p, _t = pred_dist, truth_dist
        # print(_p.shape, _t.shape, attention_mask.unsqueeze(-1).shape)
        if attention_mask is not None:
            _p = _p * attention_mask.unsqueeze(-1)
            _t = _t * attention_mask.unsqueeze(-1)
        if context_mask is not None:
            _p = _p * context_mask.unsqueeze(-1)
            _t = _t * context_mask.unsqueeze(-1)
        
        # the detached `_t` here is a soft-label in distilling.
        sim_loss = torch.cosine_similarity(_p.view(1, -1), _t.view(1, -1).detach())
        # TODO: CKA similarity
        return sim_loss

    def calculate_sim_loss(self, pred_texts, truth_texts, pred_dist, truth_dist):
        attention_mask = pred_texts['attention_mask']
        context_mask = None
        if self.similarity_ignore_target_position:
            # same correct tokens
            context_mask = (pred_texts['input_ids'] == truth_texts['input_ids']).long()

        # cosine_sim_loss = 1 - cosine_sim
        sim_loss = 1. - self.calculate_similarity(  
            pred_dist, truth_dist, 
            attention_mask=attention_mask,
            context_mask=context_mask)
        return sim_loss

    def calculate_detection_loss(
            self, logits, prob, attention_mask, det_labels, 
            loss_function=None, topk_samples=None):
        loss_function = loss_function or self.det_loss_fn
        if loss_function in ['bce']:
            # det_outputs = prob.squeeze(-1)
            det_outputs = logits.squeeze(-1)
        elif loss_function in ['focal']:  # focal
            det_outputs = self.sigmoid(logits).squeeze(-1)
        det_loss_fct = self.loss_fn.get(loss_function)

        # pad部分不计算损失
        active_positions = attention_mask.view(-1, prob.shape[1]) == 1
        active_probs = prob.view(-1, prob.shape[1])[active_positions]
        active_labels = det_labels[active_positions]

        # 取包含正例在内的 k 个位置计算 loss
        topk_samples = topk_samples or self.topk_samples
        if topk_samples > 0:
            error_counts = det_labels.sum().cpu()
            token_counts = attention_mask.sum().cpu()
            select_count = max(
                topk_samples * prob.shape[0],  # 每句采样数量 x 每个batch的句子数
                error_counts * 2)  # 错误总数 x 2
            if select_count > token_counts:  # 防止超过字符总数
                select_count = token_counts
            # (batch_size, topk_samples)
            _, correct_indices = torch.topk(
                active_labels, k=select_count, 
                dim=0, sorted=True)
            _, largest_logit_indices = torch.topk(
                active_probs, k=select_count, 
                dim=0, largest=True, sorted=True)
            
            _probs = torch.cat([
                active_probs[correct_indices],
                active_probs[largest_logit_indices]])
            _labels = torch.cat([
                active_labels[correct_indices],
                active_labels[largest_logit_indices]])

        det_loss = det_loss_fct(_probs, _labels.float())

        # loss, prob
        return det_loss, det_outputs

    def calculate_pinyin_loss(self, logits, pinyin_labels, loss_function=None):
        loss_function = loss_function or self.pinyin_loss_fn

        pinyin_loss_fct = self.loss_fn.get(loss_function)        
        pinyin_loss = pinyin_loss_fct(logits.view(-1, self.n_pinyin), 
                                      pinyin_labels.view(-1))
        return pinyin_loss, logits

    def scaling(self, logit_vector):
        value_m = logit_vector.sum(-1).unsqueeze(-1)
        logit_vector = logit_vector / (value_m + self.eps) * self.scale
        return logit_vector

    def generate_det_labels(self, original_token_ids, correct_token_ids):
        # generate token-level detaction labels. 
        # (note that tokens are from BERT-tokenizer, not actually char-level)
        return (original_token_ids != correct_token_ids).long()

    def generate_dummy_prob_mask(self, det_label):
        # generate dummy `prob` tensor for training the correction module.
        # (batch_size, sequence_length, 1)
        return det_label.unsqueeze(-1).bool().to(self._device).detach()

    def generate_ps_mask(self, token_ids, pinyin_ids=None):
        """
        generate pinyin-similar mask.
        token_ids: (batch_size, sequence_length) \in bert_vocab
        pinyin_ids: (batch_size, sequence_length) \in pinyin_vocab
        """
        # here char_ids has no-grad.
        char_ids = token_ids.clone()

        # mapping token to its all possible pinyins
        # (batch_size, sequence_length) \in bert_vocab
        # => (batch_size, sequence_length, n_pinyin) \in pinyin_vocab
        py_cand_list = self.char2py_mapping[char_ids]
        if pinyin_ids is not None:  
            if True:  # we take the only pinyin for each token
                py_cand_list = torch.scatter(
                    input=torch.zeros_like(py_cand_list), dim=-1, 
                    index=pinyin_ids.unsqueeze(-1), value=1)
            else:  # we take all heteronyms for tokens
                # print(py_cand_list.device, pinyin_ids.device)
                py_cand_list = torch.scatter(
                    input=py_cand_list, dim=-1, 
                    index=pinyin_ids.unsqueeze(-1), value=1)

        # maybe faster?
        mask = py_cand_list.float() @ self.py2pys_mapping @ self.py2char_mapping

        # each token should be mapped to itself
        # (batch_size, sequence_length, vocab_size) \in bert_vocab
        # => (batch_size, sequence_length, vocab_size) \in bert_vocab
        mask = torch.scatter(
            input=(mask > 0), dim=-1, 
            index=token_ids.unsqueeze(-1), value=1)
        return mask

    def generate_ps_mask_in_steps(self, token_ids, pinyin_ids=None):
        """
        generate pinyin-similar mask.
        token_ids: (batch_size, sequence_length) \in bert_vocab
        pinyin_ids: (batch_size, sequence_length) \in pinyin_vocab
        """
        # here char_ids has no-grad.
        char_ids = token_ids.clone()

        # mapping token to its all possible pinyins
        # (batch_size, sequence_length) \in bert_vocab
        # => (batch_size, sequence_length, n_pinyin) \in pinyin_vocab
        py_cand_list = self.char2py_mapping[char_ids]
        if pinyin_ids is not None:
            # print(py_cand_list.device, pinyin_ids.device)
            py_cand_list = torch.scatter(
                input=py_cand_list, dim=-1, 
                index=pinyin_ids.unsqueeze(-1), value=1)

        # all possible candidate pinyins for the target token
        # (batch_size, sequence_length, n_pinyin) \in pinyin_vocab
        # @ (n_pinyin, n_pinyin)
        # => (batch_size, sequence_length, n_pinyin) \in pinyin_vocab
        similar_pinyin_ids = torch.matmul(
            py_cand_list.float(), self.py2pys_mapping) > 0

        # all possible candidate tokens for the target token
        # (batch_size, sequence_length, n_pinyin) \in pinyin_vocab
        # @ (n_pinyin, vocab_size)
        # => (batch_size, sequence_length, vocab_size) \in bert_vocab
        # mask = self.py2char_mapping[similar_pinyin_ids].sum(-2) > 0
        mask = torch.matmul(
            similar_pinyin_ids.float(), self.py2char_mapping) > 0

        # each token should be mapped to itself
        # (batch_size, sequence_length, vocab_size) \in bert_vocab
        # => (batch_size, sequence_length, vocab_size) \in bert_vocab
        mask = torch.scatter(
            input=mask, dim=-1, 
            index=token_ids.unsqueeze(-1), value=1)

        return mask

    def re_generate_the_inputs(self, det_labels, pinyin_labels, 
                               enc_err, enc_cor, correct_texts=None):
        # we prepare labels for correction
        text_labels = enc_cor['input_ids']
        
        # we generate 0/1 labels for detection
        if self.predict_faulty_positions:
            if not torch.is_tensor(det_labels):
                # here we can also re-generate the detection labels, 
                # in case of faulty annotations.
                det_labels = self.collator.generate_det_labels(
                    enc_err['input_ids'], enc_cor['input_ids'])
            det_labels = det_labels.to(self._device).detach()
        
        # we generate pinyin labels for both module    
        if self.predict_pinyins:
            # generate pinyin_labels
            if pinyin_labels is None:
                pinyin_lists = self.collator.generate_pinyin_labels(
                    enc_cor['input_ids'], texts=correct_texts,
                    similar_pinyins=False, in_batch=True)
                pinyin_labels = torch.from_numpy(
                    np.stack(pinyin_lists)).detach().squeeze(-1)
                # pinyin_labels = pinyin_labels.squeeze(-1).long().to(self._device).detach()
            pinyin_labels = pinyin_labels.long().to(self._device).detach()

        # change text labels `-100`` for calculating BERT loss
        bert_text_labels = text_labels.clone()
        bert_text_labels[bert_text_labels == 0] = -100 
        return det_labels, pinyin_labels, bert_text_labels

    def tensors_lazy_load(self, device):
        if self.lazy_load_done:
            return
        if self.char2py_mapping.device != device:
            # tensors
            self.mask_value = self.mask_value.to(device)
            self.char2py_mapping = self.char2py_mapping.to(device)
            self.py2pys_mapping = self.py2pys_mapping.to(device)
            self.py2char_mapping = self.py2char_mapping.to(device)
            # print("Loaded char2py, need grad:", self.char2py_mapping.data.requires_grad)
            self.lazy_load_done = True

    def forward(self, texts, cor_labels=None, det_labels=None, 
                pinyin_inputs=None, confusion_mask=None, pinyin_labels=None, det_mask=None):
        """
        det_mask: tell the model about escape positions with NER or something (batch_size, sequence_length)
        confusion_mask: 
        """
        # record all outputs
        self.tensors_lazy_load(self._device)
        self.record_time("forward starts")
        # model_outputs = {}
        det_loss_case = []
        cor_loss_case = []

        # we generate <labels> for all losses
        encoded_err = self.get_encoded_texts(texts)
        bert_text_labels = None
        if cor_labels is not None:
            encoded_cor = self.get_encoded_texts(cor_labels)
            _labels = self.re_generate_the_inputs(
                det_labels, pinyin_labels, 
                encoded_err, encoded_cor, 
                correct_texts=cor_labels)
            det_labels, pinyin_labels, bert_text_labels = _labels
        self.record_time("labels generated")

        # we treat BERT as the token encoder here
        # logits: (batch_size, sequence_length, vocab_size)
        # hidden: (batch_size, sequence_length, hidden_size)
        bert_embeddings = self.get_bert_embeddings(encoded_err)
        bert_loss, bert_logits, bert_hidden = self.get_bert_outputs(
            encoded=encoded_err, labels=bert_text_labels)
        self.record_time("bert embeddings generated")

        # the detection module for faulty positions
        # (batch_size, sequence_length, 1)
        det_prob, det_logits = None, None
        if self.predict_faulty_positions:
            if self.input_type_for_detection.startswith('emb'):
                # take embeddings as the inputs for detection
                inputs_for_detection = bert_embeddings
            else:  # 'bert' / 'bert_logits'
                # take bert-outputs as the inputs for detection
                inputs_for_detection = bert_hidden
            # (batch_size, sequence_length, 1)
            det_logits = self.detection(inputs_for_detection)
            if self.mask_ignore_pos and det_mask is not None:  # we ignore the tokens on masked positions
                det_mask = det_mask.unsqueeze(-1).to(self._device)
                # det_logits -= 100. * det_mask.unsqueeze(-1)
                det_logits = torch.where(
                    det_mask==1, self.mask_value, det_logits)
            det_prob = self.sigmoid(det_logits)
        self.record_time("faulty positions generated")
        
        if self.bert_type in ['pinyin_merged']:
            if pinyin_inputs is None:
                pinyin_lists = self.collator.generate_pinyin_labels(
                    encoded_err['input_ids'], texts=texts,
                    similar_pinyins=False, in_batch=True)
                pinyin_inputs = torch.from_numpy(
                    np.stack(pinyin_lists)).detach().squeeze(-1)
            pinyin_inputs = pinyin_inputs.long().to(self._device).detach()            
            # print(pinyin_inputs.shape)  # given pinyin inputs are too huge
            # print(pinyin_inputs)
            py_embeddings = self.get_pinyin_embeddings(pinyin_inputs)
            self.record_time("pinyin embeddings generated")
            
            if self.merge_mask_type.lower() == 'hard':
                prob_mask = det_prob > 0.5
            elif self.merge_mask_type.lower() == 'gold' and det_labels is not None:
                prob_mask = self.generate_dummy_prob_mask(det_labels)
            else:  # 'soft'
                prob_mask = det_prob  # from detector's prediction
            second_stage_outputs = self.get_bert_outputs(
                encoded=encoded_err, labels=bert_text_labels, prob=prob_mask,
                embed=bert_embeddings, py_embed=py_embeddings,
                residual_connection=False, 
                bert_model=self.bert if self.share_bert else self.second_stage_bert)
            # override the bert_loss if we take this bert_type
            bert_loss, bert2_logits, bert2_hidden = second_stage_outputs
            self.record_time("corrector loss and logits generated")

        # the correction module for correcting errors in text
        # (batch_size, sequence_length, vocab_size)
        cor_prob = None
        cor_logits = None
        # in case when we want to train the detector only (then remove "True or")
        if True or self.cor_weight > 0:  
            # we take the second-stage logits with char&pinyin fusion input
            if self.bert_type in ['pinyin_merged']:
                cor_logits = bert2_logits
            else:  # or we take the logits with char input only
                cor_logits = bert_logits

            # mask for pinyin-similar candidates: 
            if self.mask_confusions:
                if confusion_mask is None:
                    confusion_mask = self.generate_ps_mask(
                        token_ids=encoded_err['input_ids'],
                        pinyin_ids=pinyin_inputs)  # .bool()
                    # confusion_mask = self.collator.generate_ps_mask(
                    #     input_ids=encoded_err['input_ids'], texts=texts)
                # (batch_size, sequence_length, vocab_size)
                confusion_mask = confusion_mask.to(self._device)
                # cor_logits = cor_logits - (1 - confusion_mask) * 100.
                cor_logits = torch.where(
                    confusion_mask, cor_logits, self.mask_value)
            
            if False:  # testing ps masking
                for ids, pyc, cfm in zip(encoded_err['input_ids'], pinyin_inputs, confusion_mask):
                    print(ids.shape, pyc.shape, cfm.shape)
                    for token_idx, py_idx, conf in zip(ids, pyc, cfm):
                        print(self.collator.vocab_chars[token_idx], self.collator.vocab_pinyin[py_idx])
                        for i, sign in enumerate(conf):
                            if sign:
                                print(self.collator.vocab_chars[i], end=', ')
                        print("")
            
            # (batch_size, sequence_length, vocab_size)
            cor_prob = self.softmax(cor_logits)
            self.record_time("corrector results predicted")

        # the pinyin module for predicting correct pinyins
        # (batch_size, sequence_length, pinyin_vocab_size)
        pinyin_prob = None
        if self.predict_pinyins:
            # the inputs for pinyin module come from 
            pinyin_logits, pinyin_prob = self.pinyin_predictor(bert_hidden)
            self.record_time("detector logits and results predicted")

        # output results if we don't have annotations/labels
        if cor_labels is None:
            # detection probability: (batch, sequence_length)
            if torch.is_tensor(det_prob):
                det_prob = det_prob.squeeze(-1)
            # correction probability: (batch, sequence_length, vocab_size)
            # pinyin probability: (batch, sequence_length, pinyin_vocab_size)
            outputs = (0., 0., det_prob, cor_prob, pinyin_prob)  # dummy losses for placeholders
            return outputs

        # ---------------------------------------------- #
        # calculate losses if we have annotations/labels #
        # ---------------------------------------------- #

        # loss for the correction module (BERT here)
        if self.cor_weight > 0:  # we always need the correction loss
            cor_loss = bert_loss
            cor_loss_case.append(
                cor_loss * self.loss_case_weight_cor[0])

            # add for similarity loss
            if self.consider_similarity.get('cor'):
                # the logits and hidden vectors from truth inputs
                _, tbert_logits, tbert_hidden = self.get_bert_outputs(
                    encoded=encoded_cor, labels=bert_text_labels,
                    bert_model=self.bert if self.share_bert else self.second_stage_bert)

                truth_cor_logits = tbert_logits

                sim_loss_cor = self.calculate_sim_loss(
                    pred_texts=encoded_err,
                    truth_texts=encoded_cor,
                    pred_dist=cor_logits, 
                    truth_dist=truth_cor_logits)
                cor_loss_case.append(
                    sim_loss_cor * self.loss_case_weight_cor[1])
                self.record_time("corrector similarity loss calculated")

        # loss for the faulty position predictor
        if self.predict_faulty_positions:
            det_loss, _ = self.calculate_detection_loss(
                logits=det_logits, prob=det_prob, 
                attention_mask=encoded_err['attention_mask'],
                det_labels=det_labels,
                loss_function=self.det_loss_fn)
            det_loss_case.append(
                det_loss * self.loss_case_weight_det[0])
            self.record_time("detector loss calculated")

            # add similarity loss
            if self.consider_similarity.get('det'):
                # the logits and hidden vectors from truth inputs
                _, tbert_logits, tbert_hidden = self.get_bert_outputs(
                    encoded=encoded_cor, labels=bert_text_labels)
                    
                truth_det_logits = self.detection(tbert_hidden)
                sim_loss_det = self.calculate_sim_loss(
                    pred_texts=encoded_err,
                    truth_texts=encoded_cor,
                    pred_dist=det_logits, 
                    truth_dist=truth_det_logits)
                det_loss_case.append(
                    sim_loss_det * self.loss_case_weight_det[1])
                self.record_time("detector similarity loss calculated")

            # add pinyin loss
            if self.predict_pinyins:
                # append loss for pinyin predictor for training
                pinyin_loss_det, _ = self.calculate_pinyin_loss(
                    logits=pinyin_logits,
                    pinyin_labels=pinyin_labels)
                det_loss_case.append(
                    pinyin_loss_det * self.loss_case_weight_det[2])
                self.record_time("detector pinyin loss calculated")
            
        # detection_loss, correction_loss: scalar(float)
        # detection_logits, correction_logits: 
        # (batch_size, sequence_length, num_class=1 or 21128)
        final_det_loss = sum(det_loss_case) if det_loss_case else 0.
        final_cor_loss = sum(cor_loss_case) if cor_loss_case else 0.
        outputs = (final_det_loss, final_cor_loss, det_logits, cor_logits)
        
        del encoded_err, encoded_cor
        return outputs


def model_test(text, method='eval'):
    dir_path = '/data/chendian/bbcm_checkpoints/cdmac_autodoc_cor_3rd_220922/'
    ckpt_file = dir_path + '/epoch=00_train_loss_epoch=0.1338_train_det_f1_epoch=0.9004_train_cor_f1_epoch=0.1794.ckpt'
    config_file = dir_path + '/config.yml'

    from tools.inference import load_model_directly
    model = load_model_directly(
        ckpt_file=ckpt_file, 
        config_file=config_file)

    if method in ['eval']:
        model.predict([text])
        model.show_timer()

    if method in ['train']:
        from tqdm import tqdm
        from bbcm.config import cfg
        from bbcm.utils import get_abs_path
        config_file='csc/train_cdmac_aug.yml'
        cfg.merge_from_file(get_abs_path('configs', config_file))

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

        from bbcm.data.loaders.collator import DataCollatorForCsc
        ddc = DataCollatorForCsc(tokenizer=tokenizer)

        data = ([(text, text, [9]), (text, text, [9]), (text, text, [9])])
        for _ in tqdm(range(10)):
            batch = ddc(data)
            model.train()
            model.forward(*batch)
            model.show_timer()
            

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_test(text='杨博说要去乡下探亲访友', method='eval')
