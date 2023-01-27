# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: build with pycorrector for our models' inference
@modification: okcd00(okcd00@qq.com)
"""

import os
import sys
import torch
import operator
from glob import glob
from transformers import BertTokenizer

sys.path.append('.')
sys.path.append('..')

from bbcm.modeling.csc import BertForCsc, SoftMaskedBertModel
from bbcm.modeling.cdnet import CSC_MODEL_CLASSES
from bbcm.utils import get_abs_path, get_file_modified_time
from bbcm.config import cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSC_UNK_SIGN = '֍'
UNK_TOKENS = ['“', '”', '‘', '’', '\t', '\n', '…', '—', '', CSC_UNK_SIGN]


def load_model_directly(ckpt_file, config_file=None):
    # Example:
    # ckpt_fn = 'SoftMaskedBert/epoch=02-val_loss=0.02904.ckpt' (find in checkpoints)
    # config_file = 'csc/train_SoftMaskedBert.yml' (find in configs)
    
    # load model_dir
    if os.path.exists(ckpt_file):
        if os.path.isdir(ckpt_file):
            if os.path.exists(f'{ckpt_file}/config.yml'):
                config_file = f'{ckpt_file}/config.yml'
            ckpt_file = sorted(glob(f"{ckpt_file}/*.ckpt"), 
                               key=lambda x: get_file_modified_time(x))[-1]
    else:
        ckpt_file = get_abs_path('checkpoints', ckpt_file)
        if os.path.isdir(ckpt_file):
            if os.path.exists(f'{ckpt_file}/config.yml'):
                config_file = f'{ckpt_file}/config.yml'
            ckpt_file = sorted(glob(f"{ckpt_file}/*.ckpt"), 
                               key=lambda x: get_file_modified_time(x))[-1]
    cp = ckpt_file

    # load cfg
    if os.path.exists(config_file):
        if os.path.isdir(config_file):
            config_file = f"{config_file}/config.yml"
    else:
        config_file = get_abs_path('checkpoints', ckpt_file)
        if os.path.isdir(config_file):
            config_file = f"{config_file}/config.yml"
    cfg.merge_from_file(config_file)
   
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    # load model from checkpoint        
    if cfg.MODEL.NAME in ["bert4csc", "macbert4csc"]:
        model = BertForCsc
    else:  # SoftMaskedBert
        model = CSC_MODEL_CLASSES.get(
            cfg.MODEL.NAME.lower(), SoftMaskedBertModel)
    
    model = model.load_from_checkpoint(cp,
                                       cfg=cfg,
                                       map_location=device,
                                       tokenizer=tokenizer)
    print("Loaded model from", cp)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    return model
  

class Inference:
    def __init__(self, 
                 ckpt_path='cdsmb_bbcm_220304_bothsim/',
                 cfg_path=None,
                 vocab_path='bbcm/data/vocab.txt'):
        
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.model = load_model_directly(
            ckpt_file=ckpt_path, 
            config_file=cfg_path)
        self.cfg = self.model.cfg
        print("init model instance with device:", device)

    def get_errors(self, corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                continue
            if ori_char in [' ']:
                # deal with blank word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if corrected_text[i] == CSC_UNK_SIGN:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i+1:]
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def get_errors_old(self, _corrected_text, _origin_text, blanks_cleaned=True, unk_sign='֍'):
        sub_details = []
        for i, ori_char in enumerate(_origin_text):
            if i >= len(_corrected_text):
                continue
            if ori_char == " ":
                # add blank word
                _corrected_text = _corrected_text[:i] + ori_char + _corrected_text[i if blanks_cleaned else i + 1:]
                continue
            if ori_char in UNK_TOKENS:
                # add unk word
                _corrected_text = _corrected_text[:i] + ori_char + _corrected_text[i + 1:]
                continue
            if ori_char != _corrected_text[i]:
                if (ori_char.lower() == _corrected_text[i]) or _corrected_text[i] == unk_sign:
                    # pass english upper char
                    _corrected_text = _corrected_text[:i] + ori_char + _corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, _corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return _corrected_text, sub_details

    def predict(self, sentence_list):
        """
        文本纠错模型预测
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list)
        """
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(sentence_list)
        if is_str:
            return corrected_texts[0]
        return corrected_texts

    def predict_with_all_info(self, sentence_list, unk_sign=CSC_UNK_SIGN, det_mask=None):
        """
        文本纠错模型预测，结果带完整预测信息
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list), details(list)
        """
        details = []
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        # outputs is a dict 
        corrected_texts, outputs = self.model.predict(
            sentence_list, unk_sign=unk_sign, 
            detail=True, det_mask=det_mask)

        modified_texts = []
        for corrected_text, text in zip(corrected_texts, sentence_list):
            corrected_text, sub_details = self.get_errors(corrected_text, text)
            modified_texts.append(corrected_text)
            details.append(sub_details)

        if is_str:
            return modified_texts[0], details[0], outputs
        return modified_texts, details, outputs

    def predict_with_error_detail(self, sentence_list, unk_sign=CSC_UNK_SIGN, det_mask=None):
        """
        文本纠错模型预测，结果带错误位置信息
        Args:
            sentence_list: 输入文本列表
            unk_sign: model tells an unknown token
            det_mask: the detection work is already done
        Returns: tuple
            corrected_texts(list), details(list)
        """
        details = []
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(
            sentence_list, unk_sign=unk_sign, det_mask=det_mask)

        modified_texts = []
        for corrected_text, text in zip(corrected_texts, sentence_list):
            corrected_text, sub_details = self.get_errors(corrected_text, text)
            modified_texts.append(corrected_text)
            details.append(sub_details)

        if is_str:
            return modified_texts[0], details[0]
        return modified_texts, details


if __name__ == "__main__":
    # model_path = 'cdmac_bbcm_overfit_train'
    # model_path = 'cdmac_bbcm_with_py_220317'
    # model_path = 'cdmac_sighan_zs4_psmask2'
    model_path = 'cdmac_sighan_pymerged_goldmask_220330'
    if len(sys.argv) > 1:
        model_path = str(sys.argv[1])

    # m = Inference(ckpt_path=model_path, cfg_path=None)
    m = Inference(ckpt_path=model_path)

    inputs = [
        '它的本领是呼风唤雨，因此能灭火防灾。狎鱼后面是獬豸。獬豸通常头上长着独角，有时又被称为独角羊。它很聪彗，而且明辨是非，象征着大公无私，又能镇压斜恶。',
        '老是较书。',
        '少先队 员因该 为老人让 坐',
        '感谢等五分以后，碰到一位很棒的奴生跟我可聊。',
        '遇到一位很棒的奴生跟我聊天。',
        '遇到一位很美的女生跟我疗天。',
        '他们只能有两个选择：接受降新或自动离职。',
        '王天华开心得一直说话。',
        '你说：“怎么办？”我怎么知道？',
    ]

    # inputs = ['少先队 员因该 为老人让 坐', '你说：“怎么办？”我怎么知道？']
    import numpy as np
    import pypinyin
    # print(pypinyin.lazy_pinyin('獬豸'))
    outputs, details, info = m.predict_with_all_info(inputs)
    vp = m.model.collator.vocab_pinyin
    for key, value in info.items():
        print(key)
        for sent_idx, sent in enumerate(value):
            print(inputs[sent_idx])
            print(sent)
    
    outputs, details = m.predict_with_error_detail(inputs)
    # print(details)
    if True:
        for a, b in zip(inputs, outputs):
            print('input  :', a)
            print('predict:', b)
            print()
    # 在sighan2015 test数据集评估模型
    # macbert4csc Sentence Level: acc:0.7845, precision:0.8174, recall:0.7256, f1:0.7688, cost time:10.79 s
    # softmaskedbert4csc Sentence Level: acc:0.6964, precision:0.8065, recall:0.5064, f1:0.6222, cost time:16.20 s

    from bbcm.utils.evaluations import eval_sighan2015_by_model
    print("Now testing on SIGHAN15")

    # acc, precision, recall, f1
    results = eval_sighan2015_by_model(
        correct_fn=m.predict_with_error_detail, 
        # sighan_path="/data/chendian/bbcm_datasets/csc_aug/sighan15_train_zs4.220308.json",
        sighan_path="/home/chendian/BBCM/datasets/csc_aug/sighan15_test.ct_covered.w.json",
        dump_path="./checkpoints/cdmac_sighan_pymerged_goldmask_220330/prediction.test.txt",
        verbose=True)
