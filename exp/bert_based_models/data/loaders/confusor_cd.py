"""
@Time   :   2022-04-12 10:06:06
@File   :   confusor.py
@Author :   okcd00, pangchaoxu
"""
import os
import sys
sys.path.append("../../../")

import re
import time
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from itertools import product
from collections import defaultdict
from bbcm.utils.pinyin_utils import PinyinUtils
from bbcm.utils.confuser_utils import uniform_sample, cosine_similarity


CONFUSOR_DATA_DIR = '/data/chendian/'

# embeddings and vocab in tx_embedding
SCORE_MAT_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/score_data/'
EMBEDDING_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/sound_tokens/'
PINYIN2TOKEN_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/pinyin2token_new.pkl'

# char/word frequency
ZI_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/csc_findoc_corpus/char_count.220409.json'
WORD_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/csc_findoc_corpus/word_count.220409.json'

# pre-defined knowledge
SIGHAN_CFS_PATH = f'{CONFUSOR_DATA_DIR}/sighan_confusion.txt'
PINYIN_MAPPING_PATH = f'{CONFUSOR_DATA_DIR}/pinyin_mapping.json'
CP_MAPPING_PATH = f'{CONFUSOR_DATA_DIR}/vocab_mapping.pkl'  # char2py, py2char


def default_confusor():
    return Confusor(
        cos_threshold=(0.2, 0.8), 
        method='beam',  
        token_sample_mode='sort',  # [sort|random]
        pinyin_sample_mode='mapping',  # 
        weight=[1, 0.33, 1],  
        conf_size=20, 
        debug=False)


CONFUSOR_AMB_DATA = {
    'del_mat': {
        'h': ['z', 'c', 's'], 
        'g': ['n']},
    'rep_mat': {
        'l': ['r', 'n'], 
        'n': ['l'], 
        'r': ['l'], 
        'f': ['h'], 
        'h': ['f']}
}

CONFUSOR_KEYBOARD_DATA = [
    '            ',
    ' qwertyuiop ',
    ' asdfghjkl  ',
    '  zxcvbnm   ',
    '            ',
]


class Confusor(object):
    PUNC_LIST = "，；。？！…"

    def __init__(self, 
                 cos_threshold=(0.2, 0.8), 
                 method='beam',  
                 weight=(1, 0.5, 1),  # weights for feature fusion
                 keep_num=500,  # keep how many words (most frequency) to calculate similarity
                 conf_size=10,  # output a list as the confusion set
                 pinyin_sample_mode='mapping', 
                 token_sample_mode='freq', 
                 debug=False):
        self.debug = debug
        self.cos_threshold = cos_threshold
        self.method = method
        self.weight = weight
        self.keep_num = keep_num
        self.conf_size = conf_size
        
        self.pu = PinyinUtils()
        self.pinyin_sample_mode = pinyin_sample_mode
        self.token_sample_mode = token_sample_mode

        if debug:
            self.timer = []
            print("Use {} method.".format(method))
            print("Pinyin sampling mode: {}.".format(pinyin_sample_mode))
            print("Token sampling mode: {}.".format(token_sample_mode))

        # pre-defined custom confusion set
        self.custom_char_confusion_set = {}
        self.custom_word_confusion_set = {}
        self.load_char_confusion_set()
        self.load_word_confusion_set()

        # load the word frequency data and hanzi frequency data.
        if debug:
            print("Now Loading char/word freuency data.")
        self.word_freq = json.load(open(WORD_FREQ_PATH, 'r'))
        self.zi_freq = json.load(open(ZI_FREQ_PATH, 'r'))
        # self.word_vocab: self.word_freq.get(word, 0)
        # self.zi_vocab: self.word_freq.get(char, 0)

        # pinyin2token corpus
        if debug:
            print("Now loading vocabs.")
        self.py2tokens = pickle.load(open(PINYIN2TOKEN_PATH, 'rb'))
        cp_mapping = pickle.load(open(CP_MAPPING_PATH, 'rb'))
        self.py2char = cp_mapping['p2c']
        self.char2py = cp_mapping['c2p']
        self.pinyin_mapping = json.load(open(PINYIN_MAPPING_PATH, 'r'))
        self.py_vocab = [k.strip() for k, v in self.pinyin_mapping.items()]
        self.fzimu2pinyin = defaultdict(list)
        [self.fzimu2pinyin[_py[0]].append(_py) for _py in self.py_vocab if _py[0] != '[']
        self.load_score_matrix()

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        self.timer.append((information, time.time()))

    def load_char_confusion_set(self):
        # SIGHAN
        if os.path.exists(SIGHAN_CFS_PATH):
            for line in open(SIGHAN_CFS_PATH, 'r'):
                key, val = line.strip().split(':')
                self.custom_char_confusion_set.setdefault(key, [])
                self.custom_char_confusion_set[key].extend([c for c in val])

    def load_word_confusion_set(self):
        # self.custom_word_confusion_set

        # Done: pre-processing words with tx embeddings
        # tx_corpus = '/home/chendian/BBCM/datasets/'
        # https://ai.tencent.com/ailab/nlp/zh/embedding.html

        # TODO: pre-processing words with
        # https://github.com/fighting41love/funNLP/tree/master/data
        pass

    def load_score_matrix(self):
        """
        Load and generate the RED score matrix.
        """
        keyboard_distance = {}
        offset = [
            (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, 0), (1, 1)
        ]
        for r, line in enumerate(CONFUSOR_KEYBOARD_DATA):
            for c, key in enumerate(line):
                if key == ' ':
                    continue
                keyboard_distance[(key, key)] = 0
                for r_off, c_off in offset:
                    other_key = CONFUSOR_KEYBOARD_DATA[r+r_off][c+c_off]
                    if other_key == ' ':
                        continue
                    keyboard_distance[(key, other_key)] = 1
                
        return self.score_matrix

    def load_embeddings(self, tokens):
        """
        Given a list of tokens, return the dict {token: embedding}.
        """
        def load_related_emb(tokens):
            spaths = []
            length = str(len(tokens[0]))
            for tok in tokens:
                pinyins = self.pu.to_pinyin(tok[:2])
                spath = '/'.join(pinyins)
                if spath not in spaths and os.path.exists(
                    EMBEDDING_PATH + length + '/' + spath + '.pkl'):
                    spaths.append(spath)
            tok2emb = {}
            for spath in spaths:
                emb_dict = pickle.load(open(
                    EMBEDDING_PATH + length + '/' + spath + '.pkl', 'rb'))
                tok2emb.update(emb_dict)
            if length == '1':
                emb_dict = pickle.load(open(
                    EMBEDDING_PATH + length + '/unknown.pkl', 'rb'))
                tok2emb.update(emb_dict)
            return tok2emb

        # print("Load word embeddings.")
        tok2emb = load_related_emb(tokens)
        tok_embeddings = {}
        for tok in tokens:
            emb = tok2emb.get(tok, None)
            if emb is not None:
                tok_embeddings[tok] = emb
            else:
                hanzis = list(tok)
                zi2emb = load_related_emb(hanzis)
                zi_emblist = []
                for hz in hanzis:
                    hz_emb = zi2emb.get(hz, None)
                    if hz_emb is not None:
                        zi_emblist.append(hz_emb)
                    else:
                        if self.debug:
                            print("Warning: {}, can't find the embedding of {}.".format(tok, hz))
                        zi_emblist.append(zi2emb['unk'])
                tok_embeddings[tok] = np.stack(zi_emblist).mean(axis=0)
        return tok_embeddings

    def word_frequency_score(self, word, mode='zero'):
        """
        Compute the frequency score of a given word.
        @param mode: options: {'min', 'avg', 'zero'}.
        """
        freq = self.word_freq.get(word, None)
        if not freq:
            # print("warning: Word not exsits.")
            if mode == 'min':
                freq = min([self.zi_freq.get(hanzi, 0) for hanzi in word])
            elif mode == 'avg':
                freq = sum([self.zi_freq.get(hanzi, 0) for hanzi in word]) / len(word)
            elif mode == 'zero':
                freq = 0
            else:
                raise ValueError("invalid mode: {}.".format(mode))
        return freq

    def get_pinyin_list(self, token):
        pinyins = self.pu.to_pinyin(token)
        if re.match('^[a-z]+$', ''.join(pinyins)):
            return pinyins
        else:
            return None

    def get_pinyin_sequence(self, token, cand_pinyin_num=None, method=None, pinyin_sample_mode=None, spec_func=None, **args):
        """
        Given a token, returns the candidate pinyin sequences.
        :param method: options: {'baseline', 'two-stage', 'dcc', 'beam'}
        :return: a list of candidate pinyin sequences.
        """
        method = method or self.method
        pinyin_sample_mode = pinyin_sample_mode or self.pinyin_sample_mode
        spec_func = spec_func or self.spec_func
        cand_pinyin_num = cand_pinyin_num or self.cand_pinyin_num
        first_cand_pinyin_num = 500  # the number of cand pinyins before final sampling.
        if method == 'baseline':
            cand_pinyins = self.get_pinyin_sequence_baseline(token,  first_cand_pinyin_num)
        elif method == 'two-stage':
            cand_fzimu_num = args.get('cand_fzimu_num', None) or self.cand_fzimu_num
            top_cand = self.first_stage_filtering(token, cand_fzimu_num)
            if self.debug:
                self.record_time('first stage filtering')
            cand_pinyins = self.get_pinyin_sequence_two_stage(token, first_cand_pinyin_num, top_cand)
        elif method == 'dcc':
            cand_dcc_num = args.get('cand_dcc_num', None) or self.cand_dcc_num
            top_cand = self.dcc_filtering(token, cand_dcc_num)
            if self.debug:
                self.record_time('dcc filtering')
            cand_pinyins = self.get_pinyin_sequence_two_stage(token, first_cand_pinyin_num, top_cand)
        elif method == 'beam':
            cand_zi_num = args.get('cand_zi_num', None) or self.cand_zi_num
            keep_num = args.get('keep_num', None) or self.keep_num
            cand_pinyins = self.beam_search_retrieval(token, cand_zi_num, keep_num, first_cand_pinyin_num)
        elif method == 'mapping':
            cand_pinyins = self.pinyin_mapping(token)
        else:
            raise ValueError("invalid method: {}".format(method))
        if cand_pinyins is None:
            return None
        cand_pinyins = self.sample_cand_pyseq(
            token, cand_pinyins, cand_pinyin_num, pinyin_sample_mode, 
            spec_func=spec_func)
        return cand_pinyins

    def pinyin_retrieval_recall_evaluator(self, token, evaluate_num, **args):
        self.timer = []
        self.record_time('start')
        cand_pinyins = self.get_pinyin_sequence(token, cand_pinyin_num=evaluate_num, **args)
        self.record_time('pinyin retrieval')
        return self.timer, cand_pinyins

    def get_pinyin_sequence_baseline(self, token, cand_pinyin_num):
        """
        Baseline version. Use new RED score computing strategy.
        @param corpus: a dict {token_len: {pinyin: {'tokens': [tokens], 'pylist':pinyin list}}
        @return: a list of tuples (pinyin, score) sorted by scores.
        """
        pinyin = self.get_pinyin_list(token)
        if pinyin is None:
            return None
        candpylist = {py: toks['pylist'] for py, toks in self.corpus[len(token)].items()}
        if self.debug:
            self.record_time('get pinyin sequence initialization.')
        cand_py = {py: self.compute_RED(pylist, pinyin) for py, pylist in candpylist.items()}
        top_cand = sorted(cand_py.items(), key=lambda x: x[1])
        return top_cand[:cand_pinyin_num]

    def first_stage_filtering(self, token, cand_fzimu_num):
        """
        First stage: shengmu sequences retrieval.
        @return: a list of candidate pinyin sequences.
        """
        pinyin = self.get_pinyin_list(token)
        if pinyin is None:
            return None
        fzimu = [p[0] for p in pinyin]
        candfscore = {}
        for candf in self.fzimu2pinyin[len(token)].keys():
            candfs = candf.split('|')
            assert len(candfs) == len(fzimu)
            scores = [self.fzimuREDscore[candfs[i]][fzimu[i]] for i in range(len(fzimu))]
            candfscore[candf] = sum(scores)
        top_cand = sorted(candfscore.items(), key=lambda x: x[1])[:cand_fzimu_num]
        cand_pinyins = []
        for fzimu, _ in top_cand:
            cand_pinyins.extend(self.fzimu2pinyin[len(token)][fzimu])
        return cand_pinyins

    def dcc_filtering(self, token, cand_dcc_num):
        """
        Divide and conquer filtering.
        @param cand_dcc_num: decide how many hanzi pinyins to retain for each hanzi.
        @return: a list of candidate pinyin sequences.
        """
        pinyin = self.get_pinyin_list(token)
        if pinyin is None:
            return None
        candpys = []
        for py in pinyin:
            candpy = sorted(self.ziREDscore[py].items(), key=lambda x: x[1])[:cand_dcc_num]
            candpys.append([p[0] for p in candpy])
        exsit_pys = []
        if self.debug:
            self.record_time('dcc cut')
        for gen_py in product(*candpys):
            gen_pyseq = ''.join(gen_py)
            if gen_pyseq in self.corpus[len(token)]:
                exsit_pys.append(gen_pyseq)
        return exsit_pys

    def get_pinyin_sequence_two_stage(self, token, cand_pinyin_num, cand_pinyins):
        """
        Given a list of candidate pinyin sequences, return the top candidates.
        @param cand_pinyins: a list of candidate pinyin sequences.
        @return: a list of tuples (pinyin, score) sorted by scores.
        """
        pinyin = self.get_pinyin_list(token)
        if pinyin is None:
            return None
        candpylist = {py: self.corpus[len(token)][py]['pylist'] for py in cand_pinyins}
        cand_py = {py: self.compute_RED(pylist, pinyin) for py, pylist in candpylist.items()}
        top_cand = sorted(cand_py.items(), key=lambda x: x[1])
        return top_cand[:cand_pinyin_num]

    def beam_search_retrieval(self, token, cand_zi_num, keep_num, cand_pinyin_num):
        """
        Use the beam search method to filter pinyin sequences.
        @param cand_zi_num: the number of candidate zi-pinyins in each iteration.
        @param keep_num: the max number of remained pinyin sequence after each iteration.
        @return: a list of tuples (pinyin, score) sorted by scores.
        """
        tok_len = len(token)
        token_pinyins = self.get_pinyin_list(token)
        if token_pinyins is None:
            return None
        candpys = [self.get_zi_pinyin_scores_for_beam(py)[:cand_zi_num] for py in token_pinyins]
        keep = []  # list of pinyin list
        top_res = []
        if tok_len == 1:
            top_res = [[[pylist], score / max([len(pylist), len(''.join(token_pinyins))])] \
                       for pylist, score in candpys[0][:keep_num]]  # must normalize the score for tokens of length 1 !!
        else:
            for i in range(tok_len):
                if keep == []:
                    keep = [[p[0]] for p in candpys[0][:keep_num]]
                    continue
                cand_new_pylist = []
                for cand_pylist in keep:
                    for zipy, _ in candpys[i]:
                        new_pyseq = [*cand_pylist, zipy]
                        score = self.compute_RED(new_pyseq, token_pinyins[:i + 1])
                        cand_new_pylist.append((new_pyseq, score))
                if i == tok_len - 1:
                    top_res = sorted(cand_new_pylist, key=lambda x: x[1])
                    break
                else:
                    top_res = sorted(cand_new_pylist, key=lambda x: x[1])[:keep_num]
                keep = [res[0] for res in top_res]
        # cand_pyseq = [(''.join(pylist), score) for pylist,score in top_res if ''.join(pylist) in self.corpus[tok_len]]
        cand_pyseq = []
        for pylist, score in top_res:
            pyseq = ''.join(pylist)
            if pyseq in self.corpus[tok_len]:
                if pylist == self.corpus[tok_len][pyseq]['pylist']:
                    cand_pyseq.append((pyseq, score))
        return cand_pyseq[:cand_pinyin_num]

    def sample_cand_pyseq(self, token, pinyin_scores, size, pinyin_sample_mode, spec_func=None, sort_ratio=0.7, random_state=10, debug=False):
        """
        Sample candidate (pinyin, score) pairs.
        @param token: target word.
        @param pinyin_scores:  list of (pinyin, score) pairs to sample.
        @param size: the number of output pairs.
        @param pinyin_sample_mode: options: {'sort', 'random', 'special'}.
        @param spec_func: determine whether the pinyin sequence fits a pre-defined pattern.
        @param sort_ratio: sort_ratio*size of the sampling results will be generated through the 'sort' mode.
        @return: sampling results.
        """
        sort_num = int(sort_ratio * size)
        random.seed(random_state)
        if len(pinyin_scores) < size:
            return pinyin_scores
        if pinyin_sample_mode == 'special':
            pylist = self.pu.to_pinyin(token)
            spec_res = [pair for pair in pinyin_scores if spec_func(pylist, pair[0])]
            random.shuffle(spec_res)
            cand_pyseq = spec_res[:size - sort_num]
            if debug:
                print(cand_pyseq)
            for pair in pinyin_scores:
                if len(cand_pyseq) >= size:
                    break
                if pair not in cand_pyseq:
                    cand_pyseq.append(pair)
        elif pinyin_sample_mode == 'random':
            cand_pyseq = pinyin_scores[:sort_num]
            top_others = uniform_sample(pinyin_scores[sort_num:], size - sort_num)
            cand_pyseq.extend(top_others)
        elif pinyin_sample_mode == 'sort':
            cand_pyseq = pinyin_scores[:size]
        else:
            raise ValueError("invalid pinyin sampling mode: {}".format(pinyin_sample_mode))
        if debug:
            print(cand_pyseq)
        return cand_pyseq

    def get_confuse_tokens(self, token, pinyin_scores, cos_threshold, token_sample_mode, weight, size, sort_ratio=0.7, debug=False):
        """
        @param pinyin_scores: a list of tuples (pinyin, score) sorted by scores.
        @param token_sample_mode: options: {'sort', 'random'}
        @param weight: final_score = -weight[0] * pinyin_score + weight[1] * cosine_similarity + weight[2] * frequency_score.
        @size: the number of output tokens.
        """
        if pinyin_scores is None:
            return [token]
        weight_py, weight_cos, weight_freq = weight
        candpy2score = {p[0]: p[1] for p in pinyin_scores}
        cand_tokens = [token]
        for pin in candpy2score:
            cand_tokens.extend(self.corpus[len(token)][pin]['tokens'])
        tok2emb = self.load_embeddings(cand_tokens)
        if self.debug:
            self.record_time('load embeddings')
        filtered_cand_toks = []
        for tok in cand_tokens:
            if tok == token:
                continue
            cos_sim = cosine_similarity(tok2emb[token], tok2emb[tok])
            if cos_threshold[0] <= cos_sim <= cos_threshold[1]:
                filtered_cand_toks.append(tok)
        # print("{} candidate tokens in total.".format(len(filtered_cand_toks)))
        cand2score = {}
        tok_scores = {}
        for tok in filtered_cand_toks:
            cosine_sim = cosine_similarity(tok2emb[token], tok2emb[tok])
            pinyin_score = candpy2score[''.join(self.pu.to_pinyin(tok))]
            freq_score = self.word_frequency_score(tok)
            final_score = -weight_py * pinyin_score + weight_cos * cosine_sim + weight_freq * freq_score
            # new here: candidates prefer using common characters.
            # final_score += self.common
            cand2score[tok] = final_score
            if debug:
                tok_scores.setdefault(tok, [])
                tok_scores[tok].append(final_score)
                tok_scores[tok].append(-weight_py * pinyin_score)
                tok_scores[tok].append(weight_cos * cosine_sim)
                tok_scores[tok].append(weight_freq * freq_score)
        sort_cand = sorted(cand2score.items(), key=lambda x: x[1], reverse=True)
        if debug:
            for cand, _ in sort_cand[:20]:
                print(
                    f'token: {cand}  |final_score:{round(tok_scores[cand][0], 2)} | pinyin_score: {round(tok_scores[cand][1], 2)} |' + \
                    f' cos_sim: {round(tok_scores[cand][2], 2)} | freq_score: {round(tok_scores[cand][3], 2)}')
        if len(sort_cand) <= size:
            return [p[0] for p in sort_cand]
        if token_sample_mode == 'random':
            sort_num = int(size * sort_ratio)
            cand_pairs = sort_cand[:sort_num]
            cand_pairs.extend(uniform_sample(sort_cand[sort_num:], size - sort_num))
            token_res = [p[0] for p in cand_pairs]
        elif token_sample_mode == 'sort':
            token_res = [p[0] for p in sort_cand[:size]]
        else:
            raise ValueError("invalid mode: {}".format(token_sample_mode))
        return token_res


    def __call__(self, word, context=None, word_position=None):
        if self.debug:
            self.timer = []
            self.record_time('start')
        cand_pinyins = self.get_pinyin_sequence(
            token=word, 
            method=self.method, 
            pinyin_sample_mode=self.pinyin_sample_mode)
        if self.debug:
            self.record_time('get pinyin sequence')
        confusion_set = self.get_confuse_tokens(
            token=word, 
            pinyin_scores=cand_pinyins, 
            cos_threshold=self.cos_threshold, 
            token_sample_mode=self.token_sample_mode, 
            weight=self.weight, 
            size=self.conf_size)
        if self.debug:
            self.record_time('get confusion set')
            print(confusion_set)
            return self.timer
        else:
            return confusion_set


if __name__ == "__main__":
    conf = Confusor(cos_threshold=(0.1, 0.5), method='beam', token_sample_mode='sort', 
                    pinyin_sample_mode='special', weight=[1, 0.5, 1], debug=False)
    print(conf('久'))
    print(conf('这边'))
    print(conf('好么'))
    print(conf('工具人'))
    print(conf('一鼓作气'))
    print(conf('!@#$'))   # test 
