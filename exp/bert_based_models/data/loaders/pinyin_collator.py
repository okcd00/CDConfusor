"""
@Time   :   2022-01-07 11:25:50
@File   :   ppm_collator.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""
from pypinyin.core import pinyin
from .collator import *
from bbcm.utils.pinyin_utils import PinyinUtils


class PinyinDataCollator:
    def __init__(self, tokenizer, augmentation=False):
        self.tokenizer = tokenizer
        self.pinyin_vocab = [line.strip() for line in open('/home/chendian/bbcm/datasets/full_input_list.txt', 'r')]
        self.augmentation = augmentation
        self.pinyin_utils = PinyinUtils()

    def tokens_to_pinyin_list(self, tokens):
        pinyin_list = self.pinyin_utils.to_pinyin(tokens)
        pinyin_list = self.tokenizer.convert_tokens_to_ids(pinyin_list)
        # list: vocab_ids
        return pinyin_list

    def pinyin_list_to_ids(self, pinyin_list, add_sep_cls=True, add_brackets=True, padding_to=0):
        if add_brackets:
            pinyin_list = [f'^{_py}^' if _py in self.pinyin_vocab else _py for _py in pinyin_list]
        pinyin_ids = self.tokenizer.convert_tokens_to_ids(pinyin_list)
        if add_sep_cls:
            pinyin_ids.insert(0, 101)  # [CLS]
            pinyin_ids.append(102)  # [SEP]
        if padding_to and padding_to > 0:
            pad_size = padding_to - len(pinyin_ids)
            pinyin_ids.extend([0] * pad_size)
        return pinyin_ids

    def tokens_to_ids(self, tokens, return_tensors='pt', padding_to=0):
        # input tokens from one single sentence
        pinyin_list = self.tokens_to_pinyin_list(tokens)
        if return_tensors in ['id']:
            # list: vocab_ids
            return pinyin_list
        pinyin_ids = self.pinyin_list_to_ids(
            pinyin_list, add_sep_cls=True, padding_to=padding_to)
        if return_tensors in ['pt']:
            torch.tensor(pinyin_ids)  # .long()
        return pinyin_ids

    def stack_for_encoded_inputs(self, tensor_list):
        # the input `tensor_list` should be padded.
        if isinstance(tensor_list[0], list):   
            tensor_list = [torch.tensor(t) for t in tensor_list]
        ret = {
            'input_ids': torch.stack(tensor_list, dim=0),
            'attention_mask': torch.stack([(t!=0).long() for t in tensor_list], dim=0),
            'token_type_ids': torch.stack([torch.zeros_like(t).long() for t in tensor_list], dim=0)
        }
        return ret

    def __call__(self, data):
        ori_texts, cor_texts, _ = zip(*data)
        cor_texts = [clean_text(t) for t in cor_texts]
        
        # clean_text() also transform the charator into lower()
        encoded_cor_texts = [self.tokenizer.tokenize(t) for t in cor_texts]
        max_len = max([len(t) for t in encoded_cor_texts]) + 2  # cls & sep 
        encoded_pinyin_inputs = [self.tokens_to_ids(tokens, padding_to=max_len)
                                 for tokens in encoded_cor_texts]
        ret = (
            self.stack_for_encoded_inputs(encoded_pinyin_inputs),
            self.tokenizer(cor_texts, padding=True, return_tensors='pt'))
        return ret


if __name__ == "__main__":
    pass