"""
@Time   :   2021-01-21 10:57:33
@File   :   csc_trainer.py
@Author :   okcd00, Abtion
@Email  :   okcd00@qq.com
"""

import os
import gc 
import time
import psutil
import numpy as np
from copy import deepcopy
from pprint import pprint
from mem_top import mem_top
from collections import defaultdict

import torch
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from urllib.parse import quote
from bbcm.utils import flatten
from bbcm.utils.text_utils import (split_2_short_text, convert_to_unicode, is_chinese_char)
from bbcm.data.loaders.collator import DataCollatorForCsc
from bbcm.utils.evaluations import (compute_detector_prf, 
                                    compute_corrector_prf,
                                    compute_sentence_level_prf, 
                                    report_prf)
from bbcm.engine.bases import BaseTrainingEngine


def show_memory_allocates():
    results = mem_top(
        limit=15, width=128, sep='\n',
        refs_format='{num}\t{type} {obj}', 
        bytes_format='{num}\t {obj}', 
        types_format='{num}\t {obj}',
        verbose_types=None, 
        verbose_file_name='/tmp/mem_top')
    print(results)


class CscTrainingModel(BaseTrainingEngine):
    """
    用于CSC的BaseModel, 定义了训练及预测步骤
    """
    stanford_model_path = r'/home/chendian/download/stanford-corenlp-4.2.2/'

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg = cfg
        self.timer = []
        self.debug = False
        self.memory_leak_test = True

        # model
        self.w = float(cfg.MODEL.HYPER_PARAMS[0])  # loss weight for cor & det
        self.output_type = 'logits'
        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
        # self.tokenizer._add_tokens(['“', '”'])

        # for inference
        self.entity_recognizer = None
        self.vocab_pinyin = None
        self.collator = None

        # threshold for prediction judgment
        self.judge_line = 0.5
        self.has_explict_detector = True
        self.show_result_steps = int(1e4)  # 10k steps per hour

        # record lists
        self.train_loss_epoch = []
        self.train_loss_window = []
        
        # (mean, count) case
        self.train_det_f1_case = []
        self.train_cor_f1_case = [] 

        # record matrix
        self.count_matrix_epoch = None
        self.count_matrix = None
        self.reset_matrix_epoch()
        self.reset_matrix()        

        # logging
        self.board_writer = None
        self.recorder = defaultdict(int)
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR + '/tensorboard_logs/')

        # process info
        self.process = psutil.Process(os.getpid())

    @staticmethod
    def pt2list(_tensor):
        ret = _tensor.cpu().numpy().tolist()
        return ret

    def init_collator(self): 
        return DataCollatorForCsc(
            cfg=self.cfg, tokenizer=self.tokenizer, need_pinyin=True)

    def load_vocab_pinyin(self):
        return [line.strip() for line in open(
            os.path.join(self.cfg.MODEL.DATA_PATH, 'vocab_pinyin.txt'))]

    def update_mean_value(self, case, val, cnt=1):
        if len(case) != 2:
            val_mean, val_count = 0., 0.
        else:    
            val_mean, val_count = case[0], case[1]
        val_mean = (val_mean * val_count + val * cnt) / (val_count + cnt)
        val_count += cnt
        return [val_mean, val_count]

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        self.timer.append((information, time.time()))

    def show_timer_trainer(self, to_print=True):
        if not self.timer:
            return []
        phase, start_t = self.timer[0]
        logs = []
        time_str = time.strftime("%H:%M:%S", time.gmtime(start_t))
        logs.append(f"{phase}\t{time_str}")
        for phase, t in self.timer[1:]:
            logs.append(f"{phase}\t{t - start_t}")
        if to_print:
            for line in logs:
                self.print(line)
        return logs       

    def init_entity_recognizer(self):
        from stanfordcorenlp import StanfordCoreNLP
        stanford_model_path = self.stanford_model_path

        self.print(f"Now loading NER model from {stanford_model_path}")
        stanford_model = StanfordCoreNLP(stanford_model_path, lang='zh', quiet=True)
        self.entity_recognizer = stanford_model
        # usage: word_tags = stanford_model.ner(quote(str))

    def logging_all(self, item, name, record_tb=False):
        if self.trainer.is_global_zero:
            self._logger.info(f'{name}: {item}')
        self.log(f'{name}', item)
        if record_tb:
            self.tb_writer.add_scalar(f'{name}', item, self.recorder[f'{name}'])
            # auto indexing
            self.recorder[f'{name}'] += 1

    def update_matrix(self, details):
        count_matrix = {}
        for key in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
            if key + '_counts' in details:
                tp, fp, fn = deepcopy(self.count_matrix[f"{key}_counts"])
                _tp, _fp, _fn = deepcopy(details[f"{key}_counts"])
                count_matrix[f"{key}_counts"] = [tp+_tp, fp+_fp, fn+_fn]
        self.count_matrix = count_matrix

    def reset_matrix(self):
        count_matrix = {}
        flag = False
        if self.count_matrix is None:
            flag = True
        for key in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
            if not flag:
                tp, fp, fn = deepcopy(self.count_matrix_epoch[f"{key}_counts"])
                _tp, _fp, _fn = deepcopy(self.count_matrix[f"{key}_counts"])
                self.count_matrix_epoch[f"{key}_counts"] = [tp+_tp, fp+_fp, fn+_fn]
            count_matrix[f"{key}_counts"] = [0, 0, 0]  # [TP, FP, FN]
        self.count_matrix = deepcopy(count_matrix)
        del count_matrix

    def reset_matrix_epoch(self):
        count_matrix_epoch = {}
        for key in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
            count_matrix_epoch[f"{key}_counts"] = [0, 0, 0]  # [TP, FP, FN]
        self.count_matrix_epoch = count_matrix_epoch

    def print_prf_for_fusion_matrix(self, n_step, output_key=None, count_matrix=None):
        # output_key is a list or a string
        results = {} 
        if output_key is None:
            output_key = ['cor_sent']
        if count_matrix is None:
            count_matrix = self.count_matrix
        for key in count_matrix:
            if self.w == 0. and 'cor_' in key:
                continue
            phase = f"{key.replace('_counts', '')} at {n_step}-th steps"
            if n_step == 0:
                phase = f"{key.replace('_counts', '')} at epoch end."
            tp, fp, fn = deepcopy(count_matrix[key])
            _logger = self._logger if self.trainer.is_global_zero else None
            precision, recall, f1_score = report_prf(
                tp, fp, fn, logger=_logger, phase=phase)
            if key.replace('_counts', '') in output_key:
                results[key.replace('_counts', '')] = (precision, recall, f1_score)
        return results

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts_cuda = encoded_texts.to(self._device)
        del encoded_texts  # try to explictly delete, instead of overwrite on same var_name.
        return encoded_texts_cuda 

    def get_results_from_outputs(self, outputs, batch, output_type=None):
        if output_type is None:
            output_type = self.output_type 
        # batch: 原句，纠正句，错误位置
        ori_text, cor_text, det_y = batch[:3]
        # outputs: 检错loss，纠错loss，检错输出(logits)，纠错输出(logits)
        # outputs: 检错输出(prob)，纠错输出(prob)，拼音输出(prob)
        det_output, cor_output = outputs[2], outputs[3]

        if det_output is None:
            det_prob = None
        elif output_type.startswith('logit'):
            det_prob = torch.sigmoid(det_output).squeeze(-1)
        elif output_type.startswith('prob'):  # 'prob':
            det_prob = det_output.squeeze(-1)
        else:
            raise ValueError("Invalid output type:", str(output_type))
        
        # (batch_size, sequence_length)
        # the judge_line is allowed not to be 0.5.
        if det_output is not None:
            det_pred = det_prob > self.judge_line
            det_y_hat = det_pred.to(self._device).long()

        # (batch_size, sequence_length)
        if cor_output is not None:
            encoded_y = self.get_encoded_texts(cor_text)
            cor_y_hat = torch.argmax(cor_output, dim=-1)
            cor_y_hat *= encoded_y['attention_mask']
            cor_y = encoded_y['input_ids']

        results = []
        for _idx, src in enumerate(ori_text):
            # [CLS] [1: len(text)+1] [SEP] [PAD]s
            _src = list(self.tokenizer(src, add_special_tokens=False)['input_ids'])
            if self.memory_leak_test:
                __src = deepcopy(_src)
                del _src
                _src = __src

            # whether a token from correction is the same as the truth token.
            if cor_output is None:
                c_tgt = c_predict = [0] * len(_src)
            else:
                # tgt, predict = cor_y[_idx], cor_y_hat[_idx]
                c_tgt = self.pt2list(cor_y[_idx][1: len(_src) + 1])
                c_predict = self.pt2list(cor_y_hat[_idx][1: len(_src) + 1])

            # whether a token from detection is the same as the truth token.
            if det_output is None:
                d_tgt = d_predict = [0] * len(_src)
            else:
                # det_label, det_predict = det_y[_idx], det_y_hat[_idx]
                d_tgt = self.pt2list(det_y[_idx][1: len(_src) + 1])
                d_predict = self.pt2list(det_y_hat[_idx][1: len(_src) + 1])

            # a tuple of lists, all token_ids
            results.append((_src, c_tgt, c_predict, d_tgt, d_predict))

        # results for calculating PRF
        return results

    def show_train_windows(self, n_step=0, is_epoch_end=False):
        results = self.print_prf_for_fusion_matrix(
            n_step=n_step, 
            output_key=['det_sent', 'cor_sent'],
        )  # det_sent/cor_sent
        _, _, _det_f = results.get('det_sent', (0., 0., 0.))
        _, _, _cor_f = results.get('cor_sent', (0., 0., 0.))
        avg_trn_loss = self.train_loss_window[0] if self.train_loss_window else 0.
        
        # logger info
        self.logging_all(avg_trn_loss, name='average_train_loss')
        self.logging_all(_det_f, name='average_train_det_f1')
        self.logging_all(_cor_f, name='average_train_cor_f1')
        self.train_loss_window = []
        self.reset_matrix()  # reset for next round

        if is_epoch_end:
            print("[{}] The validation on {}th epoch's starts.".format(
                self._device, self.current_epoch))
            # micro
            _det_f1 = self.train_det_f1_case[0] if self.train_det_f1_case else 0.
            _cor_f1 = self.train_cor_f1_case[0] if self.train_cor_f1_case else 0.
            self.logging_all(_det_f1, name='train_det_f1_epoch', record_tb=True)
            self.logging_all(_cor_f1, name='train_cor_f1_epoch', record_tb=True)
            self.train_det_f1_case = []
            self.train_cor_f1_case = []
            # macro
            results = self.print_prf_for_fusion_matrix(
                n_step=n_step, 
                output_key=['det_sent', 'cor_sent'],
                count_matrix=self.count_matrix_epoch,
            )  # det_sent/cor_sent
            _, _, _det_f1 = results.get('det_sent', (0., 0., 0.))
            _, _, _cor_f1 = results.get('cor_sent', (0., 0., 0.))
            self.logging_all(_det_f1, name='train_det_macro_f1_epoch', record_tb=True)
            self.logging_all(_cor_f1, name='train_cor_macro_f1_epoch', record_tb=True)
            self.reset_matrix_epoch()
        else:  # reduce space complexity
            self.train_det_f1_case = self.update_mean_value(
                case=self.train_det_f1_case, val=_det_f)
            self.train_cor_f1_case = self.update_mean_value(
                case=self.train_cor_f1_case, val=_cor_f)
        del results
        
        if self.trainer.is_global_zero:
            show_memory_allocates()  # covers gc.collect()
        else:
            gc.collect()
        self.print(f"Step {n_step}: memory {self.process.memory_info().rss // 1000000} MB")

    # @profile
    def training_step(self, batch, batch_idx):
        # ori_text, cor_text, det_labels = batch
        # outputs: 检错loss，纠错loss，检错logits，纠错logits
        if not batch:
            return 0.
        if self.debug:
            self.record_time(f'batch {batch_idx}')
        
        outputs = self.forward(*batch)
        det_loss, cor_loss = outputs[0], outputs[1]
        d_loss = det_loss.item() if torch.is_tensor(det_loss) else det_loss
        c_loss = cor_loss.item() if torch.is_tensor(cor_loss) else cor_loss
        if False and d_loss < c_loss and self.w < 0.5:
            # 先学好 det_loss，再注重学 cor_loss，所以 w 一般设置 < 0.5
            loss = (1 - self.w) * cor_loss + self.w * det_loss    
        else:  # learn more for detection as default
            loss = self.w * cor_loss + (1. - self.w) * det_loss
        
        if self.show_result_steps > 0:
            # record results
            results = self.get_results_from_outputs(
                outputs, batch, output_type='logits')
            details = compute_corrector_prf(results, logger=None) 
            if self.has_explict_detector:
                details_det = compute_detector_prf(results, logger=None)
                for _key, _value in details_det.items():
                    details[_key] = _value
            self.update_matrix(details)
            del details_det, details
            
            _loss = loss.item()
            self.train_loss_window = self.update_mean_value(
                case=self.train_loss_window, val=_loss)
            # curr_epoch = self.current_epoch
            # curr_step = self.global_step
            self.tb_writer.add_scalar('train_loss', _loss, self.global_step)
            if self.w < 1. and False:  # reduce writer CPU memory capacity
                self.tb_writer.add_scalar('train_det_loss', d_loss, self.global_step)
            if self.w > 0. and False:  # reduce writer CPU memory capacity
                self.tb_writer.add_scalar('train_cor_loss', c_loss, self.global_step)

            # show results for every k steps
            if batch_idx > 0 and self.show_result_steps != -1:
                # self.log('global_step', self.global_step)
                if batch_idx % self.show_result_steps == 0:
                    _val, _cnt = self.train_loss_window
                    self.train_loss_epoch = self.update_mean_value(
                        case=self.train_loss_epoch,
                        val=_val, cnt=_cnt)
                    # the train_loss_window will be cleaned after show func
                    self.show_train_windows(n_step=batch_idx)
                    if self.memory_leak_test:
                        gc.collect()
        return loss

    def validation_step(self, batch, batch_idx):
        # ori_text, cor_text, det_labels = batch[:3]
        # outputs: 检错loss，纠错loss，检错输出，纠错输出
        with torch.no_grad():
            outputs = self.forward(*batch)
            det_loss, cor_loss = outputs[:2]
            d_loss = det_loss.item() if torch.is_tensor(det_loss) else det_loss
            c_loss = cor_loss.item() if torch.is_tensor(cor_loss) else cor_loss
        if False and d_loss < c_loss and self.w < 0.5:
            loss = (1 - self.w) * c_loss + self.w * d_loss    
        else:  # learn more for detection as default
            loss = self.w * c_loss + (1 - self.w) * d_loss

        results = self.get_results_from_outputs(
            outputs, batch, output_type='logits')
        return loss, d_loss, c_loss, results

    def validation_epoch_end(self, outputs, log_detail_dict=False, is_testing=False):
        results = []

        logs = self.show_timer_trainer(to_print=False)
        if logs:
            with open("./logs_220811.txt", 'a') as f:
                for line in logs:
                    f.write(f"{line}\n")
                logs = []

        # loss, d_loss, c_loss, results
        for out in outputs:
            # a list of (_src, c_tgt, c_predict, d_tgt, d_predict)
            results += out[-1]
        loss = np.mean([out[0] for out in outputs])
        det_loss = np.mean([out[1] for out in outputs])
        cor_loss = np.mean([out[2] for out in outputs])
        if self.memory_leak_test:
            del outputs
        
        # record and refresh train loss case (the last-k-steps)
        if not is_testing:
            # testing phase has no logs from training
            if len(self.train_loss_window) > 0:
                _val, _cnt = self.train_loss_window
                self.train_loss_epoch = self.update_mean_value(
                    case=self.train_loss_epoch, 
                    val=_val, cnt=_cnt)
                self.train_loss_window = []
            self.show_train_windows(  # epoch ends
                n_step=0, is_epoch_end=True)
            if len(self.train_loss_epoch) > 0:
                train_loss_epoch = self.train_loss_epoch[0]
                self.logging_all(train_loss_epoch, name='train_loss_epoch')

        # measure functions
        details = compute_corrector_prf(results, logger=self._logger)
        if self.has_explict_detector:
            details_det = compute_detector_prf(results, logger=None)
            for _key, _value in details_det.items():
                details[_key] = _value
            if self.memory_leak_test:
                del details_det
            # pprint({k: v for k, v in details_det.items() if 'det' in k})
        if self.memory_leak_test:
            del results
        pprint({k: v for k, v in details.items()})
        # take sent-level as the targeted measure.
        det_f1, cor_f1 = details['det_sent_f1'], details['cor_sent_f1']
        det_acc, cor_acc = details['det_sent_acc'], details['cor_sent_acc']

        # logger info
        if not is_testing:
            # testing phase has no rest logs from training
            self.logging_all(self.current_epoch, 'epoch')
            self.logging_all(loss, 'valid_loss')
            if self.w < 1.:
                self.logging_all(det_loss, 'valid_det_loss')
            if self.w > 0.:
                self.logging_all(cor_loss, 'valid_cor_loss')

        # detailed log_dict of performance
        if log_detail_dict:
            log_dict = {k: v for k, v in details.items() if not k.endswith('_counts')}
            counts_offset = ['TP', 'FP', 'FN']
            for key in details:
                if key.endswith('_counts'):
                    for i, num in enumerate(details[key]):
                        log_dict[f"{key}_{counts_offset[i]}"] = num
                    # details[key] = np.array(details[key], dtype=int)
            self.log_dict(deepcopy(log_dict), logger=True)
            if self.memory_leak_test:
                del log_dict
        if self.memory_leak_test:
            del details  # values still remains

        print("[{}] The validation on {}th epoch's ends.".format(
            self._device, self.current_epoch))
        print(f"Detection Acc: {det_acc}, Detection F1: {det_f1}")
        print(f"Correction Acc: {cor_acc}, Correction F1: {cor_f1}")
        return det_acc, cor_acc, det_f1, cor_f1

    def test_step(self, batch, batch_idx):
        # ori_text, cor_text, det_labels = batch[:3]
        # outputs: 检错loss，纠错loss，检错输出，纠错输出
        with torch.no_grad():
            outputs = self.forward(*batch)
            
        results = self.get_results_from_outputs(
            outputs, batch, output_type='prob')
        return 0., 0., 0., results

    def test_epoch_end(self, outputs):
        det_acc, cor_acc, det_f1, cor_f1 = self.validation_epoch_end(
            outputs, log_detail_dict=True, is_testing=True)
        print("[{}] The Test on {}th epoch's ends.".format(
            self._device, self.current_epoch))
        print(f"Detection Acc: {det_acc}, Detection F1: {det_f1}")
        print(f"Correction Acc: {cor_acc}, Correction F1: {cor_f1}")
        self.tb_writer.close()
        return det_acc, cor_acc, det_f1, cor_f1
    
    def generate_pinyin_inputs_for_predict(self, texts):
        encoded_err = self.get_encoded_texts(texts)
        if self.collator is None:
            self.collator = self.init_collator()
        pinyin_lists = self.collator.generate_pinyin_labels(
            token_ids=encoded_err['input_ids'], texts=texts,
            similar_pinyins=False, in_batch=True)
        pinyin_inputs = torch.from_numpy(
            np.stack(pinyin_lists)).squeeze(-1)
        return pinyin_inputs

    def generate_ignore_mask_for_predict(self, texts):
        # the mask consists of 0(normal) and 1(ignore)
        if isinstance(texts, list):
            mask_case = [self.generate_ignore_mask_for_predict(text) for text in texts]
            max_len = max(map(len, mask_case))
            mask_case = [1 - np.pad(msk, (0, (max_len-len(msk)))) for msk in mask_case]
            return torch.from_numpy(np.stack(mask_case))

        def judge_valid_token_for_csc(token):
            if token.startswith('##'):
                token = token[2:]
            if len(token) > 1:
                return 0  # invalid
            if is_chinese_char(ord(token)):
                return 1  # valid
            return 0  # invalid

        def mark_entities(ignore_case, tok_case, _text):
            pivot = 1
            char_to_word_idx = []
            for tok_idx, tok in enumerate(tok_case):
                if tok.startswith('##'):
                    tok = tok[2:]
                for c in tok:
                    char_to_word_idx.append(tok_idx)
            char_to_word_idx.append(len(tok_case))  # end
            rest_str = f"{_text}"
            if self.entity_recognizer is None:
                self.init_entity_recognizer()
            _entities = self.entity_recognizer.ner(quote(texts))
            # print(char_to_word_idx)
            for word, tag in _entities:
                if tag in ['O', 'TITLE']:
                    continue
                # print(word, tag)
                offset = rest_str.index(word) + len(word)   
                # print(offset)
                pivot += offset
                # print(pivot, pivot-len(word), pivot)
                ignore_case[char_to_word_idx[pivot-len(word)]: char_to_word_idx[pivot-1]+1] = 0
                # print(pivot+1, pivot+1+len(word), ignore_case)
                rest_str = rest_str[offset:]
                # print(rest_str)
            return ignore_case            

        tok_case = self.tokenizer.tokenize(convert_to_unicode(texts))
        # [CLS] ... [SEP]
        ignore_case = np.array([0] + [judge_valid_token_for_csc(tok) for tok in tok_case] + [0]) 
        ignore_case = mark_entities(ignore_case, tok_case, texts)
        return ignore_case

    def convert_ids_to_pinyins(self, ids):
        if self.vocab_pinyin is None:
            self.vocab_pinyin = self.load_vocab_pinyin()
        return [self.vocab_pinyin[_id] for _id in ids]

    def predict(self, texts, detail=False, 
                predict_shorter=False, unk_sign='֍', det_labels=None, det_mask=None):
        from bbcm.utils.text_utils import clean_text
        if not isinstance(texts, list):  
            # single string without a list container
            texts = [texts]
        texts = [clean_text(text).lower() for text in texts]
        
        if predict_shorter:  # split long strings into substrings
            texts = [split_2_short_text(text) for text in texts]
            texts = [_sentence for _sentence, _idx in texts]
            parts_rec = [len(text_case) for text_case in texts]
            texts = flatten(texts)
        
        inputs = self.tokenizer(
            texts, padding=True, return_tensors='pt')
        inputs.to(self.cfg.MODEL.DEVICE)
        
        with torch.no_grad():
            pinyin_inputs = None
            if 'pinyin' in self.cfg.MODEL.ENCODER_TYPE:
                pinyin_inputs = self.generate_pinyin_inputs_for_predict(texts)
            if 'gold' not in self.cfg.MODEL.MERGE_MASK_TYPE:
                # during training, det_mask is given for gold mask
                # during inference, det_mask is None for soft mask
                det_labels = None
            
            # if det_mask is None:
            #     det_mask = self.generate_ignore_mask_for_predict(texts)

            # outputs: 检错loss，纠错loss，检错logits，纠错logits
            outputs = self.forward(
                texts=texts, 
                cor_labels=None,  # testset has no cor_labels
                det_labels=det_labels,
                pinyin_inputs=pinyin_inputs, 
                confusion_mask=None,  # generated by the model
                det_mask=det_mask,  # generate by the trainer
            )
            
            # 检测 prob，纠错 prob [, pinyin output]
            # (batch_size, sequence_length)
            # (batch_size, sequence_length, vocab_size)
            # 检错loss，纠错loss，检错输出，纠错输出
            det_prob, cor_prob, pinyin_prob = outputs[2:5]
            if self.cfg.MODEL.PREDICT_PINYINS:
                # (batch_size, sequence_length, pinyin_vocab_size)
                y_pinyins = torch.argmax(pinyin_prob, dim=-1).cpu().numpy()
            expand_text_lens = torch.sum(
                inputs['attention_mask'], dim=-1) - 1
        
        prob_detection = []
        prob_correction = []
        results_detection = []
        results_text, results_pinyin = [], []
        results_tokens = []

        if cor_prob is not None:
            y_tokens = torch.argmax(cor_prob, dim=-1).cpu().numpy()
        if det_prob is not None:
            det_prob = det_prob.cpu().numpy()
            y_detection = (det_prob > self.judge_line)

        for sent_idx, t_len in enumerate(expand_text_lens):
            original_text = texts[sent_idx]
            original_tokens = self.tokenizer.tokenize(original_text)

            # correction
            _y_hat = y_tokens[sent_idx]
            predict_tokens = self.tokenizer.convert_ids_to_tokens(_y_hat[1: t_len])
            predict_tokens = [_tok if _tok != '[UNK]' else unk_sign for _tok in predict_tokens]
            # predict_text = self.tokenizer.convert_tokens_to_string(predict_tokens).replace(' ', '')
            
            for tok_idx, (original_tok, pred_tok) in enumerate(zip(original_tokens, predict_tokens)):
                if pred_tok == '[UNK]': 
                    predict_tokens[tok_idx] = original_tok  # re-use the original text
                if pred_tok.startswith('##'):
                    predict_tokens[tok_idx] = predict_tokens[tok_idx][2:]

            predict_text = "".join(predict_tokens)
            results_text.append(predict_text)
            results_tokens.append(predict_tokens)

            # also predict detection
            if det_prob is not None:
                prob_detection.append(det_prob[sent_idx][1: t_len])
                results_detection.append(y_detection[sent_idx][1: t_len])

            # also predict pinyin
            if self.cfg.MODEL.PREDICT_PINYINS:
                _y_pinyin = y_pinyins[sent_idx]
                predict_pinyins = self.convert_ids_to_pinyins(_y_pinyin[1: t_len])
                predict_pinyins = [_tok if _tok != '[UNK]' else unk_sign 
                                   for _tok in predict_pinyins]
                results_pinyin.append(predict_pinyins)

        def join_shorter_results(_results):
            rst_temp = []
            pivot = 0
            for case_len in parts_rec:
                rst_temp.append(flatten(_results[pivot: pivot+case_len]))
            _results = rst_temp
            return _results

        # concat substrings
        if predict_shorter:  
            prob_detection = join_shorter_results(prob_detection)
            results_detection = join_shorter_results(results_detection)
            results_text = join_shorter_results(results_text)
            results_pinyin = join_shorter_results(results_pinyin)

        if detail:
            max_length = max(map(len, prob_detection))
            prob_detection = np.stack(
                [np.concatenate([p, np.zeros(max_length-len(p))]) 
                 for p in prob_detection])
            prob_detection = np.around(prob_detection, decimals=5)
            results_detection = np.concatenate(results_detection)
            detail_info = {
                'detection_prob': prob_detection,
                'detection': results_detection,
                'text': results_tokens,  # 'correction'
                'pinyin': results_pinyin,
            }
            return results_text, detail_info
        return results_text

