# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2022 All rights reserved.
#
#   filename : inference.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-03-14
#   desc     : APIs for inference call with DCN.
# ==========================================================================
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append('./')

from predict_DCN import result_predict
from evaluate import highlight_positions, compute_corrector_prf_faspell
from transformers import (
    AutoConfig,
    DCNForMaskedLM,
    DcnTokenizer)


def load_model(model_path):
    # Load pretrained model and tokenizer
    # model_path: dcn_models/findoc_finetuned_230313/

    # init model
    config = AutoConfig.from_pretrained(model_path)
    model = DCNForMaskedLM.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,)

    # init tokenizer
    inference_script_path = os.path.dirname(os.path.abspath(__file__))
    tokenizer = DcnTokenizer(
        os.path.join(inference_script_path, 'vocab', 'vocab.txt'), 
        os.path.join(inference_script_path, 'vocab', 'pinyin_vocab.txt'))
    
    # load model to GPU
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    return model, tokenizer


def B2Q(uchar):
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e: # 不是半角字符就返回原来的字符
        return uchar 
    if inside_code == 0x0020: # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code).upper()


def clean_texts(input_lines):
    # B2Q + upper
    return [''.join([B2Q(c) for c in t]) for t in input_lines]


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


def is_pure_chinese_phrase(phrase_str):
    # all tokens are Chinese
    return False not in list(map(is_chinese_char, map(ord, phrase_str)))


def predict_on_texts(input_lines, model, tokenizer, 
                     batch_size=4, max_len=180, return_fo=False):
    # pre-process texts:
    dcn_lines = clean_texts(input_lines)
    dcn_items = [[w[2:] if w.startswith('##') else w
                  for w in tokenizer.tokenize(text)] 
                 for text in dcn_lines]
    
    # record missing blanks
    blank_indexes = [
        [i for i, c in enumerate(line) if c == B2Q(' ')]
        for line in enumerate(dcn_lines)]
    
    # predict:
    result_items = result_predict(
        sentence_list=dcn_lines, 
        tokenizer=tokenizer, model=model, device='cuda', 
        batch_size=batch_size, max_seq_length=max_len)
    result_items = [[w[2:] if w.startswith('##') else w
                     for w in res] 
                    for res in result_items]
    
    # compare
    outputs = []
    for idx, (inp, out) in enumerate(zip(dcn_items, result_items)):
        offset = 0
        blanks = blank_indexes[idx]
        corrected_line = f"{input_lines[idx]}"
        if inp != out:    
            for i, (c1, c2) in enumerate(zip(inp, out)):
                if offset in blanks:
                    offset += 1
                elif c1.startswith('[') and c1.endswith(']'):
                    # [UNK] and [PAD]
                    offset += 1
                else:
                    offset += len(c1)
                if c1 == c2:
                    continue
                if is_pure_chinese_phrase(c1) and is_pure_chinese_phrase(c2):
                    # print(i, (c1, c2))
                    corrected_line = f"{corrected_line[:offset-len(c1)]}{c2}{corrected_line[offset:]}"
        outputs.append(corrected_line)
    if return_fo: # return full outputs
        return outputs, dcn_items, result_items
    return outputs


def show_diff(err, cor, pred=None):
    assert len(err) == len(cor)
    faulty_indexes = [i for i in range(len(err)) if err[i] != cor[i]]
    highlight_positions(err, faulty_indexes, 'red', default_color='white')
    highlight_positions(cor, faulty_indexes, 'green', default_color='white')
    if pred:
        highlight_positions(cor, faulty_indexes, 'orange', default_color='white')


class Inference(object):
    def __init__(self, model_path, max_len=192, detailed_evaluate=False):
        self.model, self.tokenizer = load_model(model_path)
        self.detailed_evaluate = detailed_evaluate
        self.max_len = max_len

    def evaluate(self, input_lines, truth_lines, batch_size=4):
        outputs = predict_on_texts(
            input_lines=input_lines, 
            model=self.model, tokenizer=self.tokenizer,
            batch_size=batch_size, max_len=self.max_len)
        
        results = list(zip(input_lines, truth_lines, outputs))
        metrics = compute_corrector_prf_faspell(results, strict=True)
        if self.detailed_evaluate:
            return metrics
        p, r, f, acc = metrics['cor_sent_p'], metrics['cor_sent_r'], \
            metrics['cor_sent_f1'], metrics['cor_sent_acc']
        p, r, f, acc = list(map(lambda x: round(x*100, 2), [p, r, f, acc]))
        print(f"{p}/{r}/{f} | {acc}")
        return p, r, f, acc

    def evaluate_on_tsv(self, tsv_path, batch_size=4):
        line_items = [line.strip().split('\t') 
                      for line in open(tsv_path, 'r') if line.strip()]
        input_lines = [items[0] for items in line_items]
        truth_lines = [items[1] for items in line_items]
        return self.evaluate(input_lines, truth_lines, batch_size=batch_size)

    def evaluate_on_dcn_file(self, dcn_path, batch_size=4):
        line_items = [line.strip().split('\t') 
                      for line in open(dcn_path, 'r') if line.strip()]
        def unk_to_1c(_token):
            if _token == '[UNK]':
                return '馊'  # not in vocab
            if _token.startswith('##'):
                return _token[2:]
            return _token
        input_lines = [''.join([unk_to_1c(c) for c in items[0].split(' ')]) 
                       for items in line_items]
        truth_lines = [''.join([unk_to_1c(c) for c in items[1].split(' ')]) 
                       for items in line_items]
        return self.evaluate(input_lines, truth_lines, batch_size=batch_size)

    def evaluate_bad_cases(self, input_lines, truth_lines, show_num=10):
        results, input_items, result_items = predict_on_texts(
            input_lines=input_lines, 
            model=self.model, tokenizer=self.tokenizer,
            max_len=min(192, self.tokenizer.max_len),
            return_fo=True)
        err, hit, total = 0, 0, 0
        for idx, (o, p, d, r, t) in enumerate(
            zip(input_lines, results, 
                input_items, result_items, truth_lines)):
            if o != t:
                total += 1
                if p != t:  # has error but not found
                    d = ''.join(d)
                    r = ''.join(r)
                    if r.endswith('[PAD]'):  # too-long-sentences
                        continue
                    err += 1
                    if err < show_num:
                        print(idx)
                        print("O:", o)
                        print("I:", d)
                        print("R:", r)
                        print("P:", p)
                        print("T:", t)
                        print("---------")
                        # show_diff(p, t)
                else:
                    hit += 1
        print(err, err/total, hit, hit/total, total)

    def predict(self, input_lines, batch_size=4, return_fo=False):
        # return_fo: return full outputs
        # False: return outputs (a list of strings)
        # True: return outputs, input_token_items, result_token_items
        return predict_on_texts(
            input_lines=input_lines, 
            model=self.model, tokenizer=self.tokenizer,
            batch_size=batch_size, max_len=self.max_len,
            return_fo=return_fo)

    def __call__(self, texts):
        return self.predict(input_lines=texts, batch_size=4)


def main():
    instance = Inference(
        model_path='dcn_models/findoc_finetuned_230316/checkpoint-52761/')
    # input_lines, truth_lines = 
    # instance.evaluate_bad_cases(input_lines, truth_lines)
    p, r, f, acc = instance.evaluate_on_tsv(
        '../data/cn/Wang271k/dcn_train.tsv')
        # '../data/cn/rw/rw_test.tsv')
        # '../data/cn/sighan15/sighan15_test.tsv')


if __name__ == "__main__":
    main()
