import random
from copy import deepcopy

import torch
import pickle
import pypinyin
import numpy as np
from collections import defaultdict
from bbcm.data.datasets.csc import *
from bbcm.utils.basic_utils import flatten
from bbcm.utils.text_utils import clean_text
from bbcm.data.loaders.confusor import default_confusor


class DataCollatorForCsc:
    def __init__(self, tokenizer, augmentation=False, need_pinyin=False, cfg=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.need_pinyin = need_pinyin
        
        self.debug = False
        self.skip_clean = True
        self.timer = []

        self.augmentation = augmentation
        self.confusor = default_confusor() if augmentation else None

        if self.need_pinyin:
            self.vocab_chars = [line.strip() for line in open(
                os.path.join(cfg.MODEL.DATA_PATH, 'vocab.txt'))]
            self.vocab_pinyin = [line.strip() for line in open(
                os.path.join(cfg.MODEL.DATA_PATH, 'vocab_pinyin.txt'))]
            self.vocab_size = len(self.vocab_chars)
            self.pinyin_vocab_size = len(self.vocab_pinyin)

            """
            # version 1: char2char
            phonetically_similar_mask = np.load(cfg.MODEL.DATA_PATH + 'ps_mask.npz')['matrix'].astype(np.int8)
            pre_defined_mat = torch.from_numpy(phonetically_similar_mask).detach()
            pre_defined_mat = nn.Parameter(pre_defined_mat, requires_grad=False)
            self.psw_mask = nn.Embedding(
                *phonetically_similar_mask.shape, padding_idx=0)
            self.psw_mask.weight = pre_defined_mat
            """

            # version 2: py2char
            self.need_confusion_mask = False
            self.vocab_mapping = pickle.load(open(
                os.path.join(cfg.MODEL.DATA_PATH, 'vocab_mapping.pkl'), 'rb'))
            self.pinyin_mapping = self.vocab_mapping['p2p']
            self.pinyin_same1 = defaultdict(list)
            [self.pinyin_same1[_py[0]].append(_py) for _py in self.vocab_pinyin if _py[0] != '[']
            # self.pinyin_mapping = load_json(
            #     cfg.MODEL.DATA_PATH + 'pinyin_mapping.json')

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        if self.debug:
            self.timer.append((information, time.time()))

    def show_timer(self):
        phase, start_t = self.timer[0]
        print(phase, time.strftime("%H:%M:%S", time.gmtime(start_t)))
        for phase, t in self.timer[1:]:
            print(phase, '+', t - start_t)

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(
            texts, padding=True, return_tensors='pt')
        # encoded_texts.to(self._device)
        return encoded_texts

    def get_clean_tokens_from_text(self, tokens, text=None, debug=False, return_none=False):
        clean_tokens = [_tok[2:] if _tok.startswith('##') else _tok 
                        for _wi, _tok in enumerate(tokens)]
        # if '[UNK]' not in tokens:
        #     return clean_tokens
        if text is None:
            return clean_tokens
        
        rest_str = f"[CLS]{text}[SEP]"
        original_tokens = []
        not_solved_flag = False
        for ti, tok in enumerate(clean_tokens):
            if tok in ['[CLS]', '[SEP]', '[PAD]']:
                original_tokens.append(tok)
                rest_str = rest_str[5:]
                continue
            if tok in ['[UNK]']:
                if ti + 1 < len(clean_tokens) and \
                        clean_tokens[ti+1] not in ['[UNK]']:
                    length_of_this_unk = rest_str.index(f'{clean_tokens[ti+1]}')
                    original_tokens.append(rest_str[:length_of_this_unk])
                    rest_str = rest_str[length_of_this_unk:]
                else:
                    unk_streak = 1
                    while ti+unk_streak < len(clean_tokens) \
                            and clean_tokens[ti+unk_streak] in ['[UNK]']:
                        unk_streak += 1
                    
                    if f'{clean_tokens[ti+unk_streak]}' in ['[SEP]']:
                        length_of_this_unk = len(rest_str)
                    else:
                        length_of_this_unk = rest_str.index(f'{clean_tokens[ti+unk_streak]}')
                        
                    if length_of_this_unk == unk_streak:
                        original_tokens.append(rest_str[:1])
                        rest_str = rest_str[1:]
                    else:
                        not_solved_flag = True
                        break
                continue

            if rest_str.startswith(tok):
                original_tokens.append(tok)
                rest_str = rest_str[len(tok):]
            else:
                print("broken streak:", tok)
                not_solved_flag = True
                break
                # original_tokens.append(tok)
                # rest_str = rest_str[len(tok):]
        
        if not_solved_flag:
            print("Not solved sample:", len(tokens), len(text), '\n', text)
            if return_none:
                return None
            else:
                return clean_tokens
            
        try:
            assert len(original_tokens) == len(tokens)
        except Exception as e:
            print(e)
            print(text)
            print(len(original_tokens), len(tokens))
            if debug:
                for _i in range(max(len(original_tokens), len(tokens))):
                    if _i < len(original_tokens):
                        print(original_tokens[_i], end=',')
                    if _i < len(clean_tokens):
                        print(clean_tokens[_i], end=',')
                    print("")
            print(original_tokens)
            print(tokens)
            raise ValueError()
        
        def _clean(token, word_index):
            _tok = token
            if _tok.startswith('##'):
                return _tok[2:]
            elif _tok in ['[UNK]']:
                return original_tokens[word_index]
            return _tok

        clean_tokens = [_clean(_tok, _wi) for _wi, _tok in enumerate(tokens)]
        return clean_tokens

    def generate_pinyin_labels(self, token_ids, texts=None, similar_pinyins=True, in_batch=False, debug=False):
        # pinyin to pinyin (strings)
        mat_p2p = self.pinyin_mapping  
        PAD_INDEX = self.vocab_pinyin.index('[PAD]')
        UNK_INDEX = self.vocab_pinyin.index('[UNK]')

        def get_similar_pinyins(pinyin_list):
            try:
                sp = set(flatten(
                    [mat_p2p[_p] + self.pinyin_same1[_p] 
                     for _p in pinyin_list]))
            except Exception as e:
                sp = set(flatten(
                    [mat_p2p.get(_p, []) + self.pinyin_same1.get(_p, []) 
                     for _p in pinyin_list]))
            return sp

        if in_batch:  # the token_ids and texts are lists of samples
            return [
                self.generate_pinyin_labels(
                    token_ids=_tok_ids, 
                    texts=texts if texts is None else texts[sent_idx], 
                    similar_pinyins=similar_pinyins, in_batch=False) 
                for sent_idx, _tok_ids in enumerate(token_ids)]

        # here tokens has [CLS] and [SEP] in it.
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        try:
            clean_tokens = self.get_clean_tokens_from_text(
                tokens=tokens, text=texts)
            pinyin_candidates = pypinyin.pinyin(
                clean_tokens, 
                heteronym=similar_pinyins, style=pypinyin.NORMAL, 
                errors=lambda _x: '[PAD]' if _x in ['[PAD]', '[CLS]', '[SEP]'] else '[UNK]')    
            if token_ids.shape[0] != len(pinyin_candidates):
                print("Not solved the pinyin list:", texts)
                print(token_ids)
                return None  # not solved
        except Exception as e:
            print(texts)
            print(token_ids)
            print(token_ids.shape)
            # print(len(clean_tokens))
            # print(len(pinyin_candidates))
            # print(pinyin_candidates)
        pinyin_candidates_indexes = [[(self.vocab_pinyin.index(_py) if _py in self.vocab_pinyin else UNK_INDEX)
                                      for _py in (get_similar_pinyins(_tok_py) if similar_pinyins else _tok_py)] 
                                     for _tok_py in pinyin_candidates]
                                     
        # a list of pinyin lists with same/different(similar=True) lengths
        if debug:
            print(pinyin_candidates)
            if similar_pinyins:
                for sent in pinyin_candidates_indexes:
                    print(self.convert_ids_to_pinyins(flatten(sent)))
            else:
                for sent in pinyin_candidates_indexes:
                    print(self.convert_ids_to_pinyins(sent))
        return pinyin_candidates_indexes

    def generate_ps_mask(self, input_ids, texts=None, method='py2char'):
        # generate pinyin-similar mask as a hard-confusion set.
        """
        if method in ['char2char']:
            # version 1: char2chars lookup
            confusion_mask = self.psw_mask(input_ids).detach()
            return confusion_mask
        """
        mat_p2c = self.vocab_mapping['p2c']  # pinyin to characters (indexes)

        def py_indexes_to_ps_mask(py_indexes, tok_index=None):
            ts = mat_p2c[py_indexes, :].sum(-2)
            if tok_index is not None:
                ts[tok_index] = 1
            ts[ts > 1] = 1
            return ts

        if method in ['py2char']:
            # version 2: char2pinyins -> pinyin2chars -> combine
            ps_mask_case = []
            for sent_idx, token_ids in enumerate(input_ids):
                # (sequence_length, ?)
                pinyin_candidates_indexes = self.generate_pinyin_labels(
                    token_ids=token_ids, 
                    texts=texts if texts is None else texts[sent_idx],
                    similar_pinyins=True, in_batch=False)
                if pinyin_candidates_indexes is None:
                    continue
                # print(pinyin_candidates_indexes)
                ps_mask = np.stack([py_indexes_to_ps_mask(_tok, tok_index=token_ids[_i]) 
                                    for _i, _tok in enumerate(pinyin_candidates_indexes)])
                ps_mask_case.append(ps_mask)
            
            # (batch_size, sequence_length, vocab_size)
            confusion_mask = torch.from_numpy(np.stack(ps_mask_case))
            return confusion_mask

        raise ValueError("Invalid ps mask method: {}".format(method))

    def generate_det_labels(self, original_token_ids, correct_token_ids):
        det_labels = original_token_ids != correct_token_ids
        return det_labels.long()  # .squeeze(-1)

    def generate_det_labels_old(self, ori_texts, cor_texts):
        # clean_text() also transform the charator into lower()
        encoded_ori_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        encoded_cor_texts = [self.tokenizer.tokenize(t) for t in cor_texts]

        max_len = max([len(t) for t in encoded_ori_texts]) + 2  # cls & sep 
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_ori_text, encoded_cor_text) in enumerate(zip(encoded_ori_texts, encoded_cor_texts)):
            wrong_ids = [_i for _i, (_ot, _ct) in enumerate(zip(encoded_ori_text, encoded_cor_text)) 
                         if _ot != _ct]
            for idx in wrong_ids:
                margins = []
                for word in encoded_ori_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_ori_text)) \
                        or encoded_ori_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1

        return det_labels

    def convert_ids_to_pinyins(self, ids):
        return [self.vocab_pinyin[id] for id in ids]

    def generate_model_inputs(self, ori_texts, cor_texts):
        try:
            encoded_err = self.get_encoded_texts(ori_texts)
            encoded_cor = self.get_encoded_texts(cor_texts)
            self.record_time("get Bert-encoded texts")
        except Exception as e:
            print(len(ori_texts), len(cor_texts))
            print(ori_texts, cor_texts)
            for ot in ori_texts:
                print(type(ot), len(ot), ot)
            for ot in cor_texts:
                print(type(ot), len(ot), ot)
            raise ValueError(str(e))

        # generate a tensor-form detection labels.
        # det_labels = self.generate_det_labels_old(
        #     ori_texts=ori_texts, cor_texts=cor_texts)
        try:
            det_labels = self.generate_det_labels(
                encoded_err['input_ids'], 
                encoded_cor['input_ids'])
            self.record_time("generate detection labels")
        except Exception as e:
            print(e)
            return None
        if det_labels is not None:
            det_labels = det_labels.detach()
        
        if self.need_pinyin:
            # (batch_size, sequence_length, vocab_size)
            confusion_mask = None
            if self.need_confusion_mask:
                confusion_mask = self.generate_ps_mask(  # <= slow
                    input_ids=encoded_err['input_ids'], texts=ori_texts)        
            self.record_time("generate pinyin-similar masks")
            # list for pinyin_inputs    
            pinyin_lists = self.generate_pinyin_labels(  # <= pre-generate labels
                encoded_err['input_ids'], 
                # sometimes, inputs need similar pinyins
                similar_pinyins=False, in_batch=True)  
            self.record_time("generate pinyin inputs")
            # (batch_size, sequence_length)
            pinyin_inputs = torch.from_numpy(
                np.stack(pinyin_lists)).squeeze(-1)
            self.record_time("generate pinyin tensor inputs")
            # list for pinyin_labels    
            pinyin_lists = self.generate_pinyin_labels(
                encoded_cor['input_ids'], texts=cor_texts,
                similar_pinyins=False, in_batch=True)
            self.record_time("generate pinyin labels")
            try:
                # (batch_size, sequence_length)
                if None in pinyin_lists:
                    print("pinyin_lists contains None:")
                    print(cor_texts[pinyin_lists.index(None)])
                    remain_indexes = [i for i, _py in enumerate(pinyin_lists) if _py is not None]
                    pinyin_lists = [_py for i, _py in enumerate(pinyin_lists) if i in remain_indexes]
                    ori_texts, cor_texts, det_labels, pinyin_inputs, confusion_mask = \
                        ori_texts[remain_indexes], cor_texts[remain_indexes], \
                        det_labels[remain_indexes], pinyin_inputs[remain_indexes], confusion_mask[remain_indexes]
                    cor_pinyin = torch.from_numpy(np.stack(pinyin_lists)).squeeze(-1)
                else:
                    cor_pinyin = torch.from_numpy(np.stack(pinyin_lists)).squeeze(-1)
                    self.record_time("generate pinyin tensor labels")
            except Exception as e:
                print(str(e))
                print(pinyin_lists)
                cor_pinyin = None
            if det_labels is not None: det_labels = det_labels.detach()
            if pinyin_inputs is not None: pinyin_inputs = pinyin_inputs.detach()
            if cor_pinyin is not None: cor_pinyin = cor_pinyin.detach()
            return ori_texts, cor_texts, det_labels, pinyin_inputs, confusion_mask, cor_pinyin
        return ori_texts, cor_texts, det_labels

    def __call__(self, data, debug=False):
        self.debug = debug
        self.record_time("generate starts")

        # ignore original det_labels, and then re-generate one with any tokenizer.
        ori_texts, cor_texts, _ = zip(*data)
        if self.skip_clean:
            ori_texts = [t.strip()[:510] for t in ori_texts if t.strip()]
            cor_texts = [t.strip()[:510] for t in cor_texts if t.strip()]
        else:
            ori_texts = [clean_text(t.strip())[:510] for t in ori_texts if t.strip()]
            cor_texts = [clean_text(t.strip())[:510] for t in cor_texts if t.strip()]
        self.record_time("clean texts")

        drop_no_aligned_pairs = []
        for idx in range(len(ori_texts)):
            if len(ori_texts[idx]) != len(cor_texts[idx]):
                drop_no_aligned_pairs.append(idx)
            elif ori_texts[idx] == "" or cor_texts[idx] == "":
                drop_no_aligned_pairs.append(idx)
        if len(drop_no_aligned_pairs) > 0:
            ori_texts = [t for _i, t in enumerate(ori_texts) if _i not in drop_no_aligned_pairs]
            cor_texts = [t for _i, t in enumerate(cor_texts) if _i not in drop_no_aligned_pairs]        
        self.record_time("drop no-aligned pairs")

        model_inputs = self.generate_model_inputs(ori_texts, cor_texts)
        self.record_time("generate model inputs")
        return model_inputs


def test_sample_augment(ddc):
    print(ddc.change_words('路遥知马力'))
    data = ([('Overall路遥知马力，日久见人心012345！', 'Overall路遥知马力，日久现人心012345！', [9])])
    item = ddc.sample_augment_single(data)
    print(item)
    for each in ddc(data):
        print(each)


def test_clean_tokens_from_text(ddc):
    # test_clean_tokens_from_text(ddc)
    text = '碪碪她疲惫不碪的halloweenbadapple'
    text = "首先德国laurèl破产原因是其资本结极和制度设计出现问题，幵非品牌自身缺陷；其次国内市场独立二德国市场，公司有独立的设计和销售团队运作该品牌，未来前景依然看好。"
    tokens = ['[CLS]'] + ddc.tokenizer.tokenize(text) + ['[SEP]', '[PAD]']
    print(tokens)
    results = ddc.get_clean_tokens_from_text(tokens=tokens, text=text)
    print(results)
    print(len(tokens), len(results))


if __name__ == "__main__":
    from tqdm import tqdm
    from bbcm.config import cfg
    from bbcm.utils import get_abs_path
    config_file='csc/train_cdmac_aug.yml'
    cfg.merge_from_file(get_abs_path('configs', config_file))

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    from bbcm.data.loaders.collator import *
    ddc = DataCollatorForCsc(
        tokenizer=tokenizer, need_pinyin=True, cfg=cfg)
    ddc.debug = True

    data = ([('购买日之前持有的股权投资，采用金融工具确认和计量准则进行会计处理的，将该股权投资的公允价值加上新增投资成本之和，作为改按成本法核算的初始投资成本，原持有股权的公允价值与账面价值的差额与原计入其他综合收益的累计公允价值变动全部铸入改按成本法核算的当期投资损益。Overall路遥知马力，日久涧人心01234≥5！', 
              '购买日之前持有的股权投资，采用金融工具确认和计量准则进行会计处理的，将该股权投资的公允价值加上新增投资成本之和，作为改按成本法核算的初始投资成本，原持有股权的公允价值与账面价值的差额与原计入其他综合收益的累计公允价值变动全部转入改按成本法核算的当期投资损益。Overall路遥知马力，日久现人心01234≥5！', 
              [138])])
    for _ in tqdm(range(10)):
        ddc(data, debug=True)
    ddc.show_timer()
    
