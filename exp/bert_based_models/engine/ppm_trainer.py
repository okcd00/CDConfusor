"""
@Time   :   2021-01-07 18:23:33
@File   :   ppm_trainer.py
@Author :   okcd00
@Email  :   okcd00@qq.com
"""
import operator

import numpy as np
from pypinyin.core import pinyin
import torch
import tensorboard
from transformers.models import bert
from bbcm.data.loaders.pinyin_collator import PinyinDataCollator
from bbcm.utils import flatten
from bbcm.utils.evaluations import (compute_corrector_prf,
                                    compute_sentence_level_prf, report_prf)
from transformers import BertTokenizer
from bbcm.engine.bases import BaseTrainingEngine


class CscTrainingModel(BaseTrainingEngine):
    """
        用于CSC的BaseModel, 定义了训练及预测步骤
        """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.collator = PinyinDataCollator(tokenizer=self.tokenizer)
        # self.tokenizer._add_tokens(['“', '”'])

        # threshold for prediction judgment
        self.judge_line = 0.5
        self.show_result_steps = int(2e4)  # 20k steps per hour

        # record lists
        self.train_loss_window = []
        self.board_writer = None

    @staticmethod
    def pt2list(_tensor):
        return _tensor.cpu().numpy().tolist()

    def training_step(self, batch, batch_idx):
        # encoded_pinyin_inputs, encoded_cor_labels = batch
        outputs = self.forward(*batch)
        cor_loss, prediction_scores, sequence_output = outputs
        return cor_loss

    def validation_step(self, batch, batch_idx):
        encoded_pinyin_inputs, encoded_cor_labels = batch
        outputs = self.forward(*batch)
        cor_loss, prediction_scores, sequence_output = outputs

        hits = (sequence_output == encoded_cor_labels['input_ids']) * encoded_cor_labels['attention_mask']
        total = encoded_cor_labels['attention_mask'].sum()
        results = {
            'hits': hits.sum().cpu().item(),
            'total': total.cpu().item()}

        return cor_loss.cpu().item(), results

    def validation_epoch_end(self, outputs):
        results = []
        
        loss_case = []
        hits, total = 0, 0
        for loss, results in outputs:
            # det_acc_labels += flatten(out[1])
            # cor_acc_labels += flatten(out[2])
            loss_case.append(loss)
            hits += results['hits']
            total += results['total']
        loss = np.mean(loss_case)
        print(f"[{type(loss)} {loss.shape}] loss:", loss)
        print(f"[Validation] Acc: {hits}/{total} = {hits/total}")

        # logger info
        self._logger.info(f'loss: {loss}')
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def generate_pinyin_inputs_from_texts(self, texts):
        from bbcm.utils.text_utils import clean_text
        cor_texts = [clean_text(t) for t in texts]
        
        # clean_text() also transform the charator into lower()
        encoded_cor_texts = [self.tokenizer.tokenize(t) for t in cor_texts]
        max_len = max([len(t) for t in encoded_cor_texts]) + 2  # cls & sep 
        encoded_pinyin_inputs = [self.collator.tokens_to_ids(tokens, padding_to=max_len)
                                 for tokens in encoded_cor_texts]
        pinyin_inputs = self.stack_for_encoded_inputs(encoded_pinyin_inputs)
        # text_outputs = self.tokenizer(cor_texts, padding=True, return_tensors='pt')
        return pinyin_inputs

    def generate_pinyin_inputs_from_pinyin_lists(self, pinyin_lists):
        # a list of pinyin_lists
        max_len = max([len(t) for t in pinyin_lists]) + 2  # cls & sep 
        tensor_list = [self.collator.pinyin_list_to_ids(pinyin_list, add_sep_cls=True, padding_to=max_len)
                       for pinyin_list in pinyin_lists]
        print(tensor_list)
        return self.collator.stack_for_encoded_inputs(tensor_list)
        
    def predict(self, inputs, input_type='pinyin', detail=False):
        if input_type in ['pinyin']:
            inputs = self.generate_pinyin_inputs_from_pinyin_lists(inputs)
        elif input_type in ['text', 'texts']:
            inputs = self.generate_pinyin_inputs_from_texts(inputs)        

        with torch.no_grad():
            # 检测输出，纠错输出
            outputs = self.forward(*inputs)
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            tok = self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', '')
            rst.append(tok)
        if detail:
            return rst, outputs
        return rst


if __name__ == "__main__":
    from tools.bases import args_parse
    cfg = args_parse("ppm/train_ppm.yml")

    ctm = CscTrainingModel(cfg)
    print(ctm.predict([['ni', 'hao'], ['wo', 'hao', '！']]))
