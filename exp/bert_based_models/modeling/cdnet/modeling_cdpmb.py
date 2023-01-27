"""
@Time   :   2022-03-15 11:57:00
@File   :   modeling_cdmac.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""

from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import torch.nn as nn
import transformers as tfs
from collections import OrderedDict
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import (BertEmbeddings,
                                                    BertEncoder,
                                                    BertOnlyMLMHead)

from bbcm.solver.losses import LOSS_FN_LIB
from bbcm.engine.csc_trainer import CscTrainingModel
from bbcm.data.loaders.collator import DataCollatorForCsc


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
            1  # 0=Normal, 1=Faulty
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
        prob = self.sigmoid(logits)
        # prob = self.softmax(logits)
        # (batch_size, sequence_length, 1)
        return logits, prob   # prob = p(x_i is wrong)

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
        self.cls = BertOnlyMLMHead(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self._device = device

    def forward(self, encoded_err, prob=None, embed=None, py_embed=None, 
                text_labels=None, residual_connection=False):
        # remember to set this:
        # torch 的 cross entropy loss 会忽略值为 -100 的 label
        # text_labels[text_labels == 0] = -100
        
        if embed is None:
            embed = self.embeddings(input_ids=encoded_err['input_ids'],
                                    token_type_ids=encoded_err['token_type_ids'])
        
        if prob is None:
            cor_embed = embed
        elif prob.dtype == torch.bool:
            cor_embed = prob * py_embed + ~prob * embed
        else:
            # emb_{cor} = p(is_faulty) * emb_{pinyin} + (1-p(is_faulty)) * emb_{token}
            # prob: (batch_size, sequence_length, 1)
            # py_embed: (batch_size, sequence_length, hidden_size)
            # embed: (batch_size, sequence_length, hidden_size)
            cor_embed = prob * py_embed + (1.- prob) * embed
            # print(prob.shape, py_embed.shape, embed.shape)

        input_shape = encoded_err['input_ids'].size()
        device = encoded_err['input_ids'].device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(encoded_err['attention_mask'],
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.corrector(
            cor_embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]

        # (batch_size, sequence_length, hidden_size)
        sequence_output = sequence_output + embed \
            if residual_connection else sequence_output
        # (batch_size, sequence_length, vocab_size)
        prediction_scores = self.cls(sequence_output)

        # Masked language modeling softmax layer
        cor_loss = 0.
        if text_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            cor_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), 
                text_labels.view(-1))
        return cor_loss, prediction_scores, sequence_output

    def load_from_transformers_state_dict(self, gen_fp, bert_name='bert'):
        state_dict = OrderedDict()
        gen_state_dict = tfs.AutoModelForMaskedLM.from_pretrained(gen_fp).state_dict()
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
                if name.startswith('bert'):
                    name = bert_name + name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)
        del gen_state_dict
        return state_dict
        

class CDPinyinMaskedBert(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self._device = cfg.MODEL.DEVICE

        # pre-defined loss functions.
        self.loss_fn = LOSS_FN_LIB

        # switches
        self.mask_confusions = True
        self.predict_pinyins = True  # True
        self.two_step_inference = False
        self.utilize_pinyin_features = True
        self.predict_faulty_positions = True

        # generate a mask on prediction as a hard confusion set.
        self.need_pinyin = self.mask_confusions or self.predict_pinyins
        self.collator = DataCollatorForCsc(
            tokenizer=self.tokenizer,
            augmentation=False,
            need_pinyin=self.need_pinyin,
            cfg=self.cfg)

        # basic correction & detection network.
        # self.bert = tfs.BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.bert_config = tfs.AutoConfig.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.bert = BertCorrectionModel(self.bert_config, self.tokenizer, self._device)
        
        # auxiliary pinyin features.
        if self.utilize_pinyin_features:
            MSK_INDEX = self.collator.vocab_chars.index('[MASK]')  # 103
            MSK_EMB = self.bert.embeddings.word_embeddings.weight[MSK_INDEX].clone()
            self.pinyin_embeddings = nn.Embedding(self.collator.pinyin_vocab_size,
                                                  self.bert.config.hidden_size)
            self.pinyin_embeddings.weight[1:-1].data.copy_(MSK_EMB)
            self.pinyin_embeddings.requires_grad_()
            self.conv1d = nn.Conv1d(in_channels=self.bert.config.hidden_size,
                                    out_channels=self.bert.config.hidden_size,
                                    kernel_size=3, padding=1)  
            self.layer_norm = nn.LayerNorm(
                self.bert.config.hidden_size, self.cfg.MODEL.EPS)

        # also predict faulty positions as the detection loss (and outputs)
        self.detector = DetectionNetwork(self.bert_config)
        self.load_from_transformers_state_dict(gen_fp=self.cfg.MODEL.BERT_CKPT)

        """
        wider = 1  # how much wider the GRU module needs
        self.faulty_position_gru = nn.GRU(
            self.bert.config.hidden_size,
            self.bert.config.hidden_size // 2 * wider,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )
        self.faulty_position_ffn = self.construct_ffn(
            hidden_size=self.bert.config.hidden_size * wider, 
            intermediate_size=self.bert.config.hidden_size // 2,  # -1
            n_out=2)  # [Normal, Faulty] 
        """
        self.det_loss_fn = 'bce'

        # also predict correct pinyins as an auxiliary loss.
        # nn.Linear(self.bert.config.hidden_size, self.collator.pinyin_vocab_size)
        if self.predict_pinyins:
            self.pinyin_ffn = self.construct_ffn(
                hidden_size=self.bert.config.hidden_size,
                intermediate_size=self.bert.config.hidden_size,
                n_out=self.collator.pinyin_vocab_size)
            self.pinyin_loss_fn = 'ce'
        
        # remembering n-grams (as DCN)
        self.predict_transfer_matrix = False
        # if self.predict_transfer_matrix:
        #     do_something()  

        # for the similarity case
        self.consider_similarity = {'det': True, 'cor': True}
        self.similarity_ignore_target_position = True
        self.loss_case_weight_det = [1., 1., .1]  # [det_loss, sim_loss, pinyin_loss]
        self.loss_case_weight_cor = [1., 1., .1]  # [cor_loss, sim_loss, pinyin_loss]
        self.min_retention_weight = None  # 0.1  # None

        # activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
    
    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        state_dict = self.bert.load_from_transformers_state_dict(gen_fp)
        print(f"Loaded state dict from {gen_fp}")
        self.detector.init_embeddings_from_corrector_state_dict(state_dict)

    def construct_ffn(self, hidden_size, intermediate_size=None, n_out=1):
        # half hidden_size as default
        if intermediate_size is None:
            intermediate_size = hidden_size // 2

        # one-layer full-connect
        if intermediate_size == -1:
            return nn.ModuleList([nn.Linear(hidden_size, n_out)])

        # return a module list, remember to call it iterately
        return nn.ModuleList([  # for classification
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, n_out)])
    
    def detection(self, embeddings, method='detector'):
        if method == 'detector':
            logits, prob = self.detector(
                hidden_states=embeddings)  
            det_pred = prob > 0.5
            # (batch, sequence_length, 1)
            return logits, prob, det_pred
        """
        det_logits, _ = self.faulty_position_gru(embeddings)
        for _layer in self.faulty_position_ffn:
            det_logits = _layer(det_logits)
        det_prob = self.softmax(det_logits)
        det_pred = torch.argmax(det_prob, -1)
        return det_logits[..., 1:], det_prob[..., 1:], det_pred.unsqueeze(-1)
        """

    def pinyin_predictor(self, hidden_vector):
        pinyin_logits = hidden_vector
        for _layer in self.pinyin_ffn:
            pinyin_logits = _layer(pinyin_logits)
        pinyin_prob = self.softmax(pinyin_logits)
        return pinyin_logits, pinyin_prob

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        return encoded_texts

    def merge_pinyin_features(self, hidden_vector, pinyin_inputs, conv=True):
        # add pinyin features into the original representation
        input_pinyin_embeddings = self.pinyin_embeddings(pinyin_inputs)
        ret_vector = input_pinyin_embeddings
        if conv:
            ret_vector = self.conv1d(ret_vector.permute(0, 2, 1))
            ret_vector = ret_vector.permute(0, 2, 1)
        if hidden_vector is not None:
            hidden_vector = self.layer_norm(hidden_vector + ret_vector)
            # ret_vector = hidden_vector + ret_vector
        return ret_vector

    def calculate_detection_loss(self, logits, prob, 
            attention_mask, det_labels, loss_function=None):

        # pad部分不计算损失
        active_loss = attention_mask.view(-1, prob.shape[1]) == 1

        # logits / prob (batch_size, sequence_length, 1)
        loss_function = loss_function or self.det_loss_fn
        if loss_function in ['focal']:
            _logits = logits.squeeze(-1)
            _labels = det_labels
        elif loss_function in ['bce']:
            _logits = self.sigmoid(logits).double()
            _labels = det_labels.double()
        det_loss_fct = self.loss_fn.get(loss_function)

        active_logits = _logits.view(-1, prob.shape[1])[active_loss]
        active_labels = _labels.view(-1, prob.shape[1])[active_loss]
        det_loss = det_loss_fct(active_logits, active_labels)

        return det_loss
        
    def calculate_pinyin_loss(self, logits, attention_mask, pinyin_labels, loss_function=None):
        # logits (batch_size, sequence_length, vocab_pinyin)
        # pinyin_labels (batch_size, sequence_length)
        loss_function = loss_function or self.pinyin_loss_fn
        pinyin_loss_fct = self.loss_fn.get(loss_function)        

        # pad 部分不计算损失
        sequence_length = logits.shape[1]
        active_loss = attention_mask.view(-1, sequence_length) == 1
        active_logits = logits.view(-1, sequence_length, self.collator.pinyin_vocab_size)[active_loss]
        active_labels = pinyin_labels.view(-1, sequence_length)[active_loss]

        pinyin_loss = pinyin_loss_fct(active_logits, active_labels)
        return pinyin_loss

    @staticmethod
    def calculate_similarity(pred_dist, truth_dist, 
                             attention_mask=None, context_mask=None, method='cosine'):
        # dist: [batch_size, sequence_length, hidden_size=768 or 1]
        # attn: [batch_size, sequence_length]
        # ctx: [batch_size, sequence_length]
        _p, _t = pred_dist, truth_dist
        # print(_p)
        # print(_t)
        # print(_p.shape, _t.shape, attention_mask.unsqueeze(-1).shape)
        if attention_mask is not None:
            _p = _p * attention_mask.unsqueeze(-1)
            _t = _t * attention_mask.unsqueeze(-1)
        if context_mask is not None:
            _p = _p * context_mask.unsqueeze(-1)
            _t = _t * context_mask.unsqueeze(-1)
        
        # the detached `_t` here is a soft-label in distilling.
        sim_loss = torch.cosine_similarity(_p.view(1, -1), _t.view(1, -1).detach())
        # TODO: try CKA similarity
        return sim_loss[0]

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

    def auxiliary_similarity_loss(
            self, encoded_err, encoded_cor, 
            det_logits, truth_det_logits,
            cor_logits, truth_cor_logits):
        # fusion for detection similarity loss
        det_sim_loss = 0.
        if self.consider_similarity.get('det'):
            det_sim_loss = self.calculate_sim_loss(
                pred_texts=encoded_err,
                truth_texts=encoded_cor,
                pred_dist=det_logits, 
                truth_dist=truth_det_logits)

        # fusion for correction similarity loss
        cor_sim_loss = 0.
        if self.consider_similarity.get('cor'):
            cor_sim_loss = self.calculate_sim_loss(
                pred_texts=encoded_err,
                truth_texts=encoded_cor,
                pred_dist=cor_logits, 
                truth_dist=truth_cor_logits)
        return det_sim_loss, cor_sim_loss

    def re_generate_the_inputs(self, det_labels, pinyin_labels, enc_err, enc_cor, correct_texts=None):
        # we prepare labels for correction
        text_labels = enc_cor['input_ids']
        
        # we generate 0/1 labels for detection
        if self.predict_faulty_positions:
            if not torch.is_tensor(det_labels):
                # here we can also re-generate the detection labels, 
                # in case of faulty annotations.
                det_labels = self.collator.generate_det_labels(
                    enc_err['input_ids'], enc_cor['input_ids'])
            det_labels = det_labels.to(self._device)
        
        # we generate pinyin labels for both module    
        if self.predict_pinyins:
            # generate pinyin_labels
            if pinyin_labels is None:
                pinyin_lists = self.collator.generate_pinyin_labels(
                    enc_cor['input_ids'], texts=correct_texts,
                    similar_pinyins=False, in_batch=True)
                pinyin_labels = torch.from_numpy(np.stack(pinyin_lists)).detach()
                # pinyin_labels = pinyin_labels.squeeze(-1).long().to(self._device).detach()
            pinyin_labels = pinyin_labels.squeeze(-1).long().to(self._device)

        # change text labels `-100`` for calculating BERT loss
        bert_text_labels = text_labels.clone()
        bert_text_labels[bert_text_labels == 0] = -100 
        return det_labels, pinyin_labels, bert_text_labels

    def forward(self, texts, correct_texts=None, 
                det_labels=None, pinyin_inputs=None, 
                confusion_mask=None, pinyin_labels=None):
        # record all outputs
        # model_outputs = {}
        det_loss_case = []
        cor_loss_case = []
        
        # from text to tensors
        encoded_err = self.get_encoded_texts(texts)
        
        # we generate <labels> for all losses
        if correct_texts is not None:  
            # from text to tensors
            encoded_cor = self.get_encoded_texts(correct_texts)
            det_labels, pinyin_labels, bert_text_labels = self.re_generate_the_inputs(
                det_labels, pinyin_labels, encoded_err, encoded_cor, 
                correct_texts=correct_texts)

        # we generate pinyin <inputs>
        if self.utilize_pinyin_features:
            if pinyin_inputs is None:
                pinyin_lists = self.collator.generate_pinyin_labels(
                    encoded_err['input_ids'], texts=texts,
                    similar_pinyins=False, in_batch=True)
                pinyin_inputs = torch.from_numpy(np.stack(pinyin_lists))
            pinyin_inputs = pinyin_inputs.squeeze(-1).long().to(self._device)
            # print(pinyin_inputs, pinyin_labels)

        # take the detector's embedding
        token_embeddings = self.detector.get_embeddings(
            encoded_texts=encoded_err)

        # take the corrector's embedding
        """
        token_embeddings = self.bert.embeddings(
            input_ids=encoded_err['input_ids'],
            token_type_ids=encoded_err['token_type_ids'])
        """

        # we treat bert as the token encoder here
        cor_loss, cor_logits, token_hidden = self.bert(
            encoded_err=encoded_err, prob=None, 
            embed=token_embeddings, py_embed=None, 
            text_labels=None if correct_texts is None else bert_text_labels,
            residual_connection=True)
        # TODO: need or not?
        # cor_loss_case.append(cor_loss * self.loss_case_weight_cor[0])

        # 检错 module 使用何种输入
        # det_hidden = token_hidden
        # det_hidden = token_embeddings

        """
        if self.utilize_pinyin_features:
            # 使用输入字符串的拼音作为输入
            det_hidden = self.merge_pinyin_features(
                hidden_vector=det_hidden, 
                pinyin_inputs=pinyin_inputs, conv=True)
        """

        # faulty detection module
        det_prob = None
        if self.predict_faulty_positions:     
            # 检错输出 logits / 检错概率 prob (batch_size, sequence_length, 1)
            # 检错结果 (batch_size, sequence_lenth)  
            # 1 代表当前位置有错, 多出的一维用于作为 MASK 时和 768 维做 broadcast
            det_logits, det_prob, det_pred = self.detection(token_embeddings)

        # pinyin encoder and predictor
        # 字 or 字+拼音混合输入，拼音输出
        if self.predict_pinyins:
            # (batch_size, sequence_length, pinyin_vocab_size)
            det_pinyin_logits, det_pinyin_prob = self.pinyin_predictor(token_hidden)
            # (batch_size, sequence_length)
            det_pinyin_pred = torch.argmax(det_pinyin_prob, -1)

        # we treat the same bert as the correction module here
        # (batch_size, sequence_length, hidden_size)
        if self.two_step_inference:
            if False:  # 使用预测得出的拼音作为输入
                pinyin_embeddings = self.merge_pinyin_features(
                    hidden_vector=None, pinyin_inputs=det_pinyin_pred, conv=False)
            else:  # 使用原句的拼音作为输入
                pinyin_embeddings = self.merge_pinyin_features(
                    hidden_vector=None, pinyin_inputs=pinyin_inputs, conv=False)
            # 纠错 module 使用字+读音作为输入 (hard mask)
            # prob (batch_size, sequence_length, 1) masking
            cor_loss, cor_logits, cor_token_hidden = self.bert(
                encoded_err=encoded_err, prob=det_pred, 
                embed=token_embeddings, py_embed=pinyin_embeddings, 
                text_labels=None if correct_texts is None else bert_text_labels,
                residual_connection=False)
            cor_loss_case.append(cor_loss * self.loss_case_weight_cor[0])
            # also predict pinyins on the 2nd step
            if self.predict_pinyins:
                cor_pinyin_logits, cor_pinyin_prob = self.pinyin_predictor(cor_token_hidden)
            # cor_pinyin_pred = torch.argmax(cor_pinyin_prob, -1)

        # 纠错输出 logits (batch_size, sequence_length, vocab_size)
        if self.mask_confusions:
            # drop the impossible candidates using the confusion mask
            if confusion_mask is None:
                # (batch_size, sequence_length, vocab_size)
                confusion_mask = self.collator.generate_ps_mask(
                    input_ids=encoded_err['input_ids'])
            confusion_mask = confusion_mask.to(self._device).detach()
            # (batch_size, sequence_length, vocab_size)
            cor_logits = cor_logits - (1 - confusion_mask) * 100.

        # we calculate losses here
        if correct_texts is not None:
            # faulty position loss (detection loss)
            det_loss = 0.
            if self.predict_faulty_positions:
                det_loss = self.calculate_detection_loss(
                    logits=det_logits, prob=det_prob, 
                    attention_mask=encoded_err['attention_mask'],
                    det_labels=det_labels,
                    loss_function=self.det_loss_fn)
            det_loss_case.append(det_loss * self.loss_case_weight_det[0])

            # similarity losses (detection and correction)
            if True in self.consider_similarity.values():
                # prepare truth_bert_outputs for calculating similarity loss
                _, truth_logits, _ = self.bert(
                    encoded_err=encoded_cor, prob=None, 
                    text_labels=None, residual_connection=True)
                """
                if self.utilize_pinyin_features:
                    # 使用真实字符串的用字和拼音
                    truth_token_hidden = self.merge_pinyin_features(
                        hidden_vector=truth_token_hidden, 
                        pinyin_inputs=pinyin_labels, conv=True)
                """
                if self.mask_confusions:
                    # drop the impossible candidates using the (input) confusion mask
                    # (batch_size, sequence_length, vocab_size)
                    truth_logits = truth_logits - (1 - confusion_mask) * 100.
                # add similarity losses into detection and correction modules
                
                # take detector's embedding
                truth_token_embeddings = self.detector.get_embeddings(encoded_cor)

                # take corrector's embedding
                """
                truth_token_embeddings = self.bert.embeddings(
                    input_ids=encoded_cor['input_ids'],
                    token_type_ids=encoded_cor['token_type_ids'])
                """
                # (batch_size, sequence_length, 1)
                truth_det_logits, _, _ = self.detection(truth_token_embeddings)
                det_sim_loss, cor_sim_loss = self.auxiliary_similarity_loss(
                    encoded_err, encoded_cor, 
                    det_logits=det_logits, 
                    truth_det_logits=truth_det_logits,
                    cor_logits=cor_logits,
                    truth_cor_logits=truth_logits)
                det_loss_case.append(det_sim_loss * self.loss_case_weight_cor[1])
                cor_loss_case.append(cor_sim_loss * self.loss_case_weight_cor[1])
            
            # calculate pinyin prediction losses
            if self.predict_pinyins:
                if self.predict_faulty_positions:
                    det_pinyin_loss = self.calculate_pinyin_loss(
                        logits=det_pinyin_logits,
                        attention_mask=encoded_err['attention_mask'],
                        pinyin_labels=pinyin_labels)
                    det_loss_case.append(det_pinyin_loss * self.loss_case_weight_det[2])
                if self.two_step_inference:
                    cor_pinyin_loss = self.calculate_pinyin_loss(
                        logits=cor_pinyin_logits,
                        attention_mask=encoded_cor['attention_mask'],
                        pinyin_labels=pinyin_labels)
                    cor_loss_case.append(cor_pinyin_loss * self.loss_case_weight_cor[2])

            # 检错部分loss，纠错部分loss，检错 logits 输出，纠错 logits 输出
            # print(det_logits.shape, cor_logits.shape)  # [8,30,1], [8,30,21128]
            # print(det_loss_case)
            # print(cor_loss_case)
            outputs = (sum(det_loss_case), sum(cor_loss_case), det_logits, cor_logits)
        else:
            cor_prob = torch.softmax(cor_logits, -1)
            # 检错概率 (batch, sequence_length)
            # 纠错概率 (batch, sequence_length, vocab_size)
            # 拼音概率 (batch, sequence_length, pinyin_vocab_size)
            if self.two_step_inference and self.predict_pinyins:
                outputs = (det_prob.squeeze(-1), cor_prob, det_pinyin_prob, cor_pinyin_prob)
            else:
                outputs = (det_prob.squeeze(-1), cor_prob, det_pinyin_prob)
        return outputs
