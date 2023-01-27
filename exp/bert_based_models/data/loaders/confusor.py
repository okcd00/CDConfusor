"""
@Time   :   2021-08-03 17:38:56
@File   :   confusor.py
@Author :   pangchaoxu, okcd00
"""
import os
import sys
sys.path.append("../../../")

from tqdm import tqdm

import re
import time
import json
import random
import pickle
import numpy as np
from pprint import pprint
from itertools import product
from bbcm.utils import edit_distance
from bbcm.utils.file_io import get_filesize
from bbcm.utils.pinyin_utils import PinyinUtils
from bbcm.utils.confuser_utils import (
    generate_score_matrix,
    cosine_similarity,
    complete_ziRED,
    complete_zi_sim_matrix,
    uniform_sample
)
from bbcm.utils.spec_func import same_last_hanzi_first_letter


CONFUSOR_DATA_DIR = '/data/chendian/'
# CONFUSOR_DATA_DIR = '/data/pangchaoxu'

SCORE_MAT_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/score_data/'
EMBEDDING_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/sound_tokens/'

CORPUS_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/pinyin2token_noname.pkl'
SIMILAR_PINYIN_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/similar_pinyins.pkl'
FZIMU2PINYIN_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/fzimu2pinyin.pkl'
REDSCORE_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/ziREDscore.pkl'
FZIMURED_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/fzimuREDscore.pkl'
ZISIMMATRIX = f'{CONFUSOR_DATA_DIR}/tencent_embedding/zi_sim_matrix.pkl'

USE_WC_WORD_FREQ = True
WC_WORD_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/wc_data_records/wc_word_frequency_score_01.pkl'
WC_2GRAM_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/wc_data_records/wc_word2_frequency_score.pkl'
# Findoc Word Freq
WORD_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/wc_data_records/findoc_word_frequency_score_01.pkl'
WORD_2GRAM_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/wc_data_records/findoc_word2_frequency_score.pkl'
# ZI_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/wc_data_records/findoc_char_frequency.pkl'

# [deprecated]
# WORD_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/word_freq.pkl'
# ZI_FREQ_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/zi_freq.pkl'

BERT_VOCAB_PATH = f'{CONFUSOR_DATA_DIR}/pretrained_bert_models/chinese-macbert-base/vocab.txt'
SIGHAN_CFS_PATH = f'{CONFUSOR_DATA_DIR}/sighan_confusion.txt'
PINYIN_MAPPING_PATH = f'{CONFUSOR_DATA_DIR}/pinyin_mapping.json'


class Confusor(object):
    PUNC_LIST = "，；。？！…"
    FIRST_PINYIN_PENALTY = 0.05

    def __init__(self, amb_score=0.5, inp_score=0.25, cos_threshold=(0.1, 0.5), method='beam', cand_pinyin_num=10,
                 cand_fzimu_num=100, cand_dcc_num=50, cand_zi_num=50, keep_num=500, weight=(1, 0.5, 1), conf_size=10,
                 pinyin_sample_mode='sort', token_sample_mode='special', spec_func=same_last_hanzi_first_letter, debug=False):
        """
        @param amb_score: [0, 1) score of the ambiguous sounds.
        @param inp_score: [0, 1) score of the input errors.
        @param cos_threshold: the threshold of the cosine similarity filtering.
        @param method: pinyin sequence retrieve method, options: {'baseline', 'two-stage', 'dcc', 'beam'}.
        @param cand_pinyin_num: the number of candidate pinyin sequences.
        @param cand_dcc_num: the number of candidate zi-pinyins for each hanzi. (only for 'dcc' method)
        @param cand_fzimu_num: the number of candidate first stage pinyin sequences. (only for 'two-stage' method)
        @param cand_zi_num: the number of candidate zi-pinyins in each iteration. (only for 'beam' method)
        @param keep_num: the max number of remained pinyin sequence after each iteration. (only for 'beam' method)
        @param pinyin_sample_mode: options: {'sort', 'random', 'special', 'mapping'} the 'sort' mode sorts candidates by weighted scores.
        @param token_sample_mode: options: {'sort', 'random'} the 'sort' mode sorts candidates by weighted scores.
        @param weight: final_score = -weight[0] * pinyin_score + weight[1] * cosine_similarity + weight[2] * frequency_score.
        @param conf_size: the size of confusion set.
        """
        self.debug = debug
        self.timer = []
        self.amb_score = amb_score
        self.inp_score = inp_score
        self.cos_threshold = cos_threshold
        self.cand_dcc_num = cand_dcc_num
        self.method = method
        self.cand_pinyin_num = cand_pinyin_num
        self.cand_fzimu_num = cand_fzimu_num
        self.cand_zi_num = cand_zi_num
        self.keep_num = keep_num
        self.weight = weight
        self.conf_size = conf_size
        self.pu = PinyinUtils()
        self.pinyin_sample_mode = pinyin_sample_mode
        self.token_sample_mode = token_sample_mode
        self.spec_func = spec_func
        print("Use {} method.".format(method))
        print("Pinyin sampling mode: {}.".format(pinyin_sample_mode))
        print("Token sampling mode: {}.".format(token_sample_mode))

        self.bert_vocab = [line.strip() for line in open(BERT_VOCAB_PATH, 'r')]
        self.char_confusion_set = {}
        self.word_confusion_set = {}  # a function is better than a dict
        self.external_word_set = {}

        # self.load_sighan_confusion_set()
        self.load_word_confusion_set()

        # pinyin2token corpus
        print("Now loading pinyin2token corpus.")
        self.pinyin2token_corpus = self.load_pickle(CORPUS_PATH)
        self.similar_pinyins = self.load_pickle(SIMILAR_PINYIN_PATH)
        if 'two-stage' in self.method:
            print("Now loading fzimu2pinyin corpus.")
            self.fzimu2pinyin = self.load_pickle(FZIMU2PINYIN_PATH)
        if 'beam' in self.method:
            print("Now loading zi_sim_matrix.")
            self.zi_sim_matrix = self.load_pickle(ZISIMMATRIX)

        print("Now loading REDscore:")
        self.ziREDscore = self.load_pickle(REDSCORE_PATH)
        if 'two-stage' in self.method:
            print("Now loading fzimuREDscore:")
            self.fzimuREDscore = self.load_pickle(FZIMURED_PATH)

        # load and generate the score matrix
        print("Now generating score matrix.")
        self.score_matrix = self.load_score_matrix()

        # load the word frequency data and hanzi frequency data.
        print("Now Loading word freuency data:")
        self.pinyin_mapping = json.load(open(PINYIN_MAPPING_PATH, 'r'))
        self.vocab_pinyin = sorted(self.pinyin_mapping.keys())
        self.fzimu_bucket = {_fzimu: sorted([_p for _p in self.vocab_pinyin if _p.startswith(_fzimu)]) 
                             for _fzimu in 'abcdefghijklmnopqrstuvwxyz'}
        self.word_freq = self.load_pickle(WORD_FREQ_PATH)
        self.word_2gram_freq = self.load_pickle(WORD_2GRAM_FREQ_PATH)
        self.wc_word_freq = {}
        self.wc_2gram_freq = {}
        if USE_WC_WORD_FREQ:
            self.wc_word_freq = self.load_pickle(WC_WORD_FREQ_PATH)
            self.wc_2gram_freq = self.load_pickle(WC_2GRAM_FREQ_PATH)

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        self.timer.append((information, time.time()))

    def is_pinyin_similar_char(self, py1, py2):
        ed = edit_distance(py1, py2, rate=False)
        if ed > 2 or ed / max(len(py1), len(py2)) > 0.5:
            return False
        return True

    def is_pinyin_similar(self, py_case1, py_case2):
        if len(py_case1) != len(py_case2):
            print("Pinyin sequence length is not equal: {} vs {}".format(py_case1, py_case2))
            return False
        token_length = len(py_case1)
        same_case = [py_case1[_i] == py_case2[_i] for _i in range(token_length)]
        simliar_case = [self.is_pinyin_similar_char(py_case1[_i], py_case2[_i]) 
                        for _i in range(token_length)]
        if token_length == 1: 
            return simliar_case[0]
        elif same_case.count(False) / token_length > 0.5: 
            # if more than half of the pinyin sequence is different, it is not similar.
            return False
        if False in simliar_case:
            # special-case: ABCD -> AB'CD / ABCD -> ABCD' / ABC -> ABC'
            # will be added in 'first-freedom' method
            return False
        return True

    def load_sighan_confusion_set(self):
        for line in open(SIGHAN_CFS_PATH, 'r'):
            key, val = line.strip().split(':')
            self.char_confusion_set.setdefault(key, [])
            self.char_confusion_set[key].extend([c for c in val])

    def load_word_confusion_set(self):
        # tx_corpus = '/home/chendian/BBCM/datasets/'

        # Done: pre-processing words with tx embeddings
        # https://ai.tencent.com/ailab/nlp/zh/embedding.html

        # TODO: pre-processing words with
        # https://github.com/fighting41love/funNLP/tree/master/data
        pass

    def load_score_matrix(self):
        """
        Load and generate the RED score matrix.
        """
        amb_data = pickle.load(open(SCORE_MAT_PATH + 'amb_data.pkl', 'rb'))
        inp_data = pickle.load(open(SCORE_MAT_PATH + 'inp_data.pkl', 'rb'))
        self.score_matrix = generate_score_matrix(
            amb_data, self.amb_score, inp_data, self.inp_score)
        return self.score_matrix

    def load_pickle(self, fp):
        print(f"Loading {fp.split('/')[-1]} ({get_filesize(fp)}MB)")
        if not os.path.exists(fp):
            print("Failed")
            return None
        return pickle.load(open(fp, 'rb'))

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

    def compute_RED(self, pylist1, pylist2):
        """
        Compute the RED score of two pinyin lists.
        """
        assert len(pylist1) == len(pylist2)
        scores = []
        updated = False
        for i in range(len(pylist2)):
            fpy = pylist1[i]
            spy = pylist2[i]
            if fpy not in self.ziREDscore:
                self.ziREDscore = complete_ziRED(fpy, self.ziREDscore, self.score_matrix)
                updated = True
            if spy not in self.ziREDscore:
                self.ziREDscore = complete_ziRED(spy, self.ziREDscore, self.score_matrix)
                updated = True
            scores.append(self.ziREDscore[fpy][spy])
        if updated:
            print("save new ziREDscore to path: {}".format(REDSCORE_PATH))
            pickle.dump(self.ziREDscore, open(REDSCORE_PATH, 'wb'))
        return sum(scores) / max([len(''.join(pylist1)), len(''.join(pylist2))])

    def word_frequency_score(self, word, mode='prod'):
        """
        Compute the frequency score of a given word.
        @param mode: options: {'min', 'avg', 'zero'}.
        """
        
        wc_freq = self.wc_word_freq.get(word, 0.)
        wc_2gram_freq = self.wc_2gram_freq.get(word, 0.)
        findoc_freq = self.word_freq.get(word, 0.)
        findoc_2gram_freq = self.word_2gram_freq.get(word, 0.)

        def char_freq_to_word_freq(mode):
            # print("warning: Word not exsits.")
            if mode == 'min':
                freq = min([self.zi_freq.get(hanzi, 0) for hanzi in word])
            elif mode == 'avg':
                freq = sum([self.zi_freq.get(hanzi, 0) for hanzi in word]) / len(word)
            elif mode == 'prod':
                freq = np.prod([self.zi_freq.get(hanzi, 0) for hanzi in word])
            elif mode == 'zero':
                freq = 0
            else:
                raise ValueError("invalid mode: {}.".format(mode))
            return freq
        
        if len(word) == 1 and findoc_freq == 0.:
            findoc_freq = 2.  # OOV char in FinDoc
        # if freq > 0:
        #     print(word, freq)
        #     freq = np.log10(freq) * 0.1
        if USE_WC_WORD_FREQ:        
            return findoc_freq, findoc_2gram_freq, wc_freq, wc_2gram_freq
        return findoc_freq, findoc_2gram_freq

    def get_pinyin_list(self, token):
        # return a list of pinyin 
        # ['gong'] / ['chao', 'duan', 'qun']
        pinyins = self.pu.to_pinyin(token)
        if re.match('^[a-z]+$', ''.join(pinyins)):
            return pinyins
        else:
            return None

    def get_pinyin_sequence(self, token, cand_pinyin_num=None, method=None, 
                            pinyin_sample_mode=None, spec_func=None, **args):
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
        if self.debug:
            self.record_time('before pinyin filtering')

        if 'baseline' in method:
            cand_pinyins = self.get_pinyin_sequence_baseline(
                token, first_cand_pinyin_num)
        elif 'two-stage' in method:
            cand_fzimu_num = args.get('cand_fzimu_num', None) or self.cand_fzimu_num
            top_cand = self.first_stage_filtering(token, cand_fzimu_num)
            if self.debug:
                self.record_time('first stage filtering')
            cand_pinyins = self.get_pinyin_sequence_two_stage(
                token, first_cand_pinyin_num, top_cand)
        elif 'dcc' in method:
            cand_dcc_num = args.get('cand_dcc_num', None) or self.cand_dcc_num
            top_cand = self.dcc_filtering(token, cand_dcc_num)
            if self.debug:
                self.record_time('dcc filtering')
            cand_pinyins = self.get_pinyin_sequence_two_stage(
                token, first_cand_pinyin_num, top_cand)
        elif 'beam' in method:
            cand_zi_num = args.get('cand_zi_num', None) or self.cand_zi_num
            keep_num = args.get('keep_num', None) or self.keep_num
            cand_pinyins = self.beam_search_retrieval(
                token, cand_zi_num, keep_num, first_cand_pinyin_num)
        elif 'all-similar' in method:
            cand_pinyins = self.get_pinyin_sequence_all(token)
            if self.debug:
                self.record_time('all-similar filtering')
        else:
            raise ValueError("invalid method: {}".format(method))

        if 'single-freedom' in method:
            more_cand_pinyins = self.get_pinyin_sequence_single_freedom(token)
            cand_pinyins = sorted(list(set(cand_pinyins + more_cand_pinyins)), 
                                  key=lambda x: x[1])
            if self.debug:
                self.record_time('single-freedom filtering')

        if cand_pinyins is None:
            return None
        if self.debug:
            print('candidate pinyins', cand_pinyins)
        cand_pinyins = self.sample_cand_pyseq(
            token=token, 
            pinyin_scores=cand_pinyins, 
            size=cand_pinyin_num, 
            pinyin_sample_mode=pinyin_sample_mode,
            spec_func=spec_func)
        return cand_pinyins

    def pinyin_retrieval_recall_evaluator(self, token, evaluate_num, **args):
        self.timer = []
        self.record_time('start')
        cand_pinyins = self.get_pinyin_sequence(
            token, cand_pinyin_num=evaluate_num, **args)
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
        candpylist = {py: toks['pylist'] 
                      for py, toks in self.pinyin2token_corpus[len(token)].items()}
        if self.debug:
            self.record_time('get pinyin sequence initialization.')
        cand_py = {py: self.compute_RED(pylist, pinyin) 
                   for py, pylist in candpylist.items()}
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
        exist_pys = []
        if self.debug:
            self.record_time('dcc cut')
        for gen_py in product(*candpys):
            gen_pyseq = ''.join(gen_py)
            if gen_pyseq in self.pinyin2token_corpus[len(token)]:
                exist_pys.append(gen_pyseq)
        return exist_pys

    def get_pinyin_sequence_single_freedom(self, token, debug=False):
        """
        Given a list of candidate pinyin sequences, the last word has different pinyins.
        @return: a list of tuples (pinyin, score) sorted by scores.
        """
        ret = []
        pinyin_list = self.get_pinyin_list(token)
        token_length = len(pinyin_list)
        if pinyin_list is None or len(pinyin_list) <= 1:
            return ret
        for _idx, _py in enumerate(pinyin_list):
            if debug:
                print(_py)
            if _idx == 1:
                # if debug:  print(_py, self.fzimu_bucket[_py[0]])
                for fzimu_py in self.fzimu_bucket[_py[0]]: 
                    _candidate_pinyin = pinyin_list[:1] + [fzimu_py] + pinyin_list[2:]
                    if not ''.join(_candidate_pinyin) in self.pinyin2token_corpus[token_length]:
                        continue
                    ret.append((''.join(_candidate_pinyin), 
                                self.compute_RED(_candidate_pinyin, pinyin_list)))
            elif _idx == len(pinyin_list) - 1:
                for fzimu_py in self.fzimu_bucket[_py[0]]: 
                    _candidate_pinyin = pinyin_list[:-1] + [fzimu_py]
                    if not ''.join(_candidate_pinyin) in self.pinyin2token_corpus[token_length]:
                        continue
                    ret.append((''.join(_candidate_pinyin), 
                                self.compute_RED(_candidate_pinyin, pinyin_list)))
        ret = sorted(ret, key=lambda x: x[1])
        # if debug:  print(ret)
        return ret

    def get_pinyin_sequence_two_stage(self, token, cand_pinyin_num, cand_pinyins):
        """
        Given a list of candidate pinyin sequences, return the top candidates.
        @param cand_pinyins: a list of candidate pinyin sequences.
        @return: a list of tuples (pinyin, score) sorted by scores.
        """
        pinyin = self.get_pinyin_list(token)
        if pinyin is None:
            return None
        candpylist = {py: self.pinyin2token_corpus[len(token)][py]['pylist'] for py in cand_pinyins}
        cand_py = {py: self.compute_RED(pylist, pinyin) for py, pylist in candpylist.items()}
        top_cand = sorted(cand_py.items(), key=lambda x: x[1])
        return top_cand[:cand_pinyin_num]

    def get_zi_pinyin_scores_for_beam(self, pinyin):
        """
        Get zi_pinyin_scores for the beam search retrieval. If the zi_pinyin is not in the zi_sim_matrix, update it.
        @returns: List of (other_zi_pinyin, sim_score_with_input) in the increasing order.
        """
        if pinyin not in self.zi_sim_matrix:
            self.zi_sim_matrix, ziREDscore = complete_zi_sim_matrix(
                pinyin, self.ziREDscore, self.score_matrix)
            print("save new zi_sim_matrix to path: {}".format(ZISIMMATRIX))
            pickle.dump(self.zi_sim_matrix, open(ZISIMMATRIX, 'wb'))
            if ziREDscore:
                self.ziREDscore = ziREDscore
                print("Complete ziREDscore either. Save new ziREDscore to path: {}".format(REDSCORE_PATH))
                pickle.dump(self.ziREDscore, open(REDSCORE_PATH, 'wb'))
        return self.zi_sim_matrix[pinyin]

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
        candpys = [self.get_zi_pinyin_scores_for_beam(py)[:cand_zi_num] 
                   for py in token_pinyins]
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
            if pyseq in self.pinyin2token_corpus[tok_len]:
                if pylist == self.pinyin2token_corpus[tok_len][pyseq]['pylist']:
                    cand_pyseq.append((pyseq, score))
        return cand_pyseq[:cand_pinyin_num]

    def get_pinyin_sequence_all(self, token):
        """
        Get all pinyin sequences for the given token.
        @return: a list of tuples (pinyin, score) sorted by scores.
        """
        pinyin = self.get_pinyin_list(token)
        if pinyin is None:
            return None

        cand_pyseq = []
        if self.similar_pinyins is not None:
            similar_pinyins = []
            if len(token) in self.similar_pinyins:
                if tuple(pinyin) in self.similar_pinyins[len(token)]:
                    similar_pinyins = sorted(self.similar_pinyins[len(token)][tuple(pinyin)])
            if len(similar_pinyins):
                if tuple(pinyin) not in similar_pinyins:
                    similar_pinyins.append(tuple(pinyin))
                for pylist in similar_pinyins:
                    score = self.compute_RED(pylist, pinyin)
                    cand_pyseq.append((''.join(pylist), score))
                return sorted(cand_pyseq, key=lambda x: x[1])

        # calculate
        for py_str in self.pinyin2token_corpus[len(token)]:
            pylist = self.pinyin2token_corpus[len(token)][py_str]['pylist']
            if self.is_pinyin_similar(pylist, pinyin):
                score = self.compute_RED(pylist, pinyin)
                cand_pyseq.append((py_str, score))
        return sorted(cand_pyseq, key=lambda x: x[1])

    def sample_cand_pyseq(self, token, pinyin_scores, size, pinyin_sample_mode, 
                          spec_func=None, sort_ratio=0.7, random_state=10, debug=False):
        """
        Sample candidate (pinyin, score) pairs.
        @param token: target word.
        @param pinyin_scores:  list of (pinyin, score) pairs to sample.
        @param size: the number of output pairs.
        @param pinyin_sample_mode: options: {'sort', 'random', 'special', 'wide'}.
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
            # the number of candidates depends on the length of the token
            cand_pyseq = pinyin_scores[:size**len(token)]        
        else:
            raise ValueError("invalid pinyin sampling mode: {}".format(pinyin_sample_mode))
        if debug:
            print(cand_pyseq)
        return cand_pyseq

    def get_confuse_tokens(self, token, pinyin_scores, cos_threshold, token_sample_mode, weight, size, sort_ratio=0.7, debug=False):
        """
        @param pinyin_scores: a list of tuples (pinyin, score) sorted by scores.
        @param token_sample_mode: options: {'sort', 'random'}
        @param weight: final_score depends on 
            weight[0] * pinyin_score, 
            weight[1] * cosine_similarity, and
            weight[2] * frequency_score.
        @size: the number of output tokens.
        """

        _src_py_case = self.pu.to_pinyin(token)
        _src_py = ''.join(_src_py_case)
        
        # whether we need the token itself in its confusion set.
        cand_tokens = [token]  # []

        if pinyin_scores is None:
            return cand_tokens
        weight_py, weight_cos, weight_freq = weight
        candpy2score = {p[0]: p[1] for p in pinyin_scores}
        for pin in candpy2score:
            if self.pinyin2token_corpus[len(token)].get(pin) is None:
                continue
            _tokens = self.pinyin2token_corpus[len(token)][pin]['tokens']
            if _tokens:
                cand_tokens.extend(_tokens)
                # print(pin, _tokens)
        cand_tokens = [tok for tok in cand_tokens if 
                       (len(tok) == 1 and tok in self.bert_vocab) or len(tok) > 1]
        cand_tokens = sorted(set(cand_tokens))
        
        calculate_cos_sim = True  # whether to calculate cosine similarity
        filtered_cand_toks = []
        if cos_threshold is None or cos_threshold == (0., 1.):
            calculate_cos_sim = False
            filtered_cand_toks = cand_tokens
        else:  # load embedding for selected tokens.
            tok2emb = self.load_embeddings(cand_tokens)
            if self.debug:
                self.record_time('load embeddings')
            for tok in cand_tokens:
                if tok == token:
                    continue
                cos_sim = cosine_similarity(tok2emb.get(token), tok2emb.get(tok))
                if cos_threshold[0] <= cos_sim <= cos_threshold[1]:
                    filtered_cand_toks.append(tok)
        # print("{} candidate tokens in total.".format(len(filtered_cand_toks)))

        cand2score, tok_scores = {}, {}
        for tok in filtered_cand_toks:
            cosine_sim = 0
            if weight_cos != 0 and calculate_cos_sim:
                print(weight_cos, calculate_cos_sim)
                cosine_sim = cosine_similarity(tok2emb[token], tok2emb[tok])
            _cand_py_case = self.pu.to_pinyin(tok)
            _cand_py = ''.join(_cand_py_case)
            pinyin_score = candpy2score.get(_cand_py, 
                self.compute_RED(_cand_py_case, _src_py_case))
            # pinyin_score (similarity)
            up_limit = 0.25  # current maximum RED score is 0.25
            pinyin_score = (up_limit - pinyin_score) / up_limit
            # extra_penalty for the first letter/char:
            if _cand_py_case[0] != _src_py_case[0]:
                pinyin_score -= self.FIRST_PINYIN_PENALTY
                if _cand_py[0] != _src_py[0]:
                    pinyin_score -= self.FIRST_PINYIN_PENALTY
            wc_freq, wc_2gram_freq = 0., 0.
            freq_scores = self.word_frequency_score(tok)
            findoc_freq, findoc_2gram_freq = freq_scores[:2]
            if USE_WC_WORD_FREQ:
                wc_freq, wc_2gram_freq = freq_scores[2:]
                if len(token) == 1:
                    freq_score = (wc_freq + wc_2gram_freq) / 2
                    final_score = sum([
                        weight_py * pinyin_score, 
                        # weight_freq * freq_delta, 
                        weight_freq * freq_score])
                else:
                    freq_delta = max(wc_freq, wc_2gram_freq) - \
                                 max(findoc_freq, findoc_2gram_freq)
                    freq_score = max(
                        [findoc_freq + findoc_2gram_freq, 
                         wc_freq + wc_2gram_freq]) / 2
                    if wc_2gram_freq > 0. and findoc_freq + findoc_2gram_freq + wc_freq == 0.:
                        freq_score *= 0.9
                    final_score = sum([
                        weight_py * pinyin_score, 
                        max(weight_freq * freq_delta, 
                            weight_freq * freq_score)])
                if sum(freq_scores) == 0.:
                    final_score -= 1.  # ignore
            else:
                final_score = weight_py * pinyin_score + \
                              weight_freq * (findoc_freq + findoc_2gram_freq)
                              # weight_cos * cosine_sim + \

            # new here: candidates prefer using common characters.
            # final_score += self.common
            cand2score[tok] = final_score
            if debug:
                tok_scores.setdefault(tok, {})
                tok_scores[tok] = {
                    'final_score': final_score,
                    'pinyin_score': pinyin_score,
                    'cosine_sim': cosine_sim,
                    'freq_score': freq_scores,
                }
        sort_cand = sorted(cand2score.items(), 
                           key=lambda x: x[1], reverse=True)
        if debug:
            show_how_many = 30
            for cand, _ in sort_cand[:show_how_many]:
                _final_score = round(tok_scores[cand]['final_score'], 3)
                _pinyin_score = round(tok_scores[cand]['pinyin_score'], 3)
                _freq_score_str = " ".join(
                    [f"{round(freq_score, 4):.04f}" 
                        for freq_score in tok_scores[cand]['freq_score']])
                print_str = f'token: {cand}  ' + \
                    f'| final_score: {_final_score:.04f} ' + \
                    f'| pinyin_score: {_pinyin_score:.04f} '
                print_str += f'| freq_score: {_freq_score_str} '
                _cosine_sim = tok_scores[cand]['cosine_sim']
                if _cosine_sim != 0.:
                    print_str += f'| cos_sim: {round(_cosine_sim, 3):.3f} '
                print(print_str)
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

    def check_ranking(self, cor_word, err_word):
        # recording for later restoring
        debug = self.debug    
        conf_size = self.conf_size

        self.debug = True
        self.conf_size = 5000

        cand_pinyins = self.get_pinyin_sequence(
            token=cor_word, method=self.method, 
            pinyin_sample_mode=self.pinyin_sample_mode)
        err_pinyin_case = self.get_pinyin_list(err_word)
        err_pinyin = ''.join(err_pinyin_case)
        for _idx, (_pinyin, _score) in enumerate(cand_pinyins):
            if _idx < 20 or _pinyin == err_pinyin:
                print(_pinyin, _score)
        confusion_set = self.get_confuse_tokens(
            token=cor_word, 
            pinyin_scores=cand_pinyins, 
            cos_threshold=self.cos_threshold, 
            token_sample_mode=self.token_sample_mode, 
            weight=self.weight, 
            size=self.conf_size, 
            debug=self.debug)
        if err_word in confusion_set:
            rank = confusion_set.index(err_word)
            print(f"{err_word} is the {rank}-th candidate of {cor_word}")

    def __call__(self, word, context=None, word_position=None):
        if self.debug:
            self.timer = []
            self.record_time('start')
        if self.debug:
            print(self.get_pinyin_list(word))
        cand_pinyins = self.get_pinyin_sequence(
            token=word, method=self.method, 
            pinyin_sample_mode=self.pinyin_sample_mode)
        if self.debug:
            # pprint(sorted(cand_pinyins, key=lambda x: x[1]))
            pprint(list(map(lambda x: x[0], 
                   sorted(cand_pinyins, key=lambda x: x[1]))))
            self.record_time('get pinyin sequence')
        confusion_set = self.get_confuse_tokens(
            token=word, 
            pinyin_scores=cand_pinyins, 
            cos_threshold=self.cos_threshold, 
            token_sample_mode=self.token_sample_mode, 
            weight=self.weight, 
            size=self.conf_size, 
            debug=self.debug)
        if self.debug:
            self.record_time('get confusion set')
            phase, start_t = self.timer[0]
            print(phase, time.strftime("%H:%M:%S", time.gmtime(start_t)))
            for phase, t in self.timer[1:]:
                print(phase, '+', t - start_t)
        return confusion_set


def default_confusor():
    return Confusor(
        cand_pinyin_num=100, 
        cos_threshold=(0., 1.), 
        method='all-similar single-freedom', 
        token_sample_mode='sort', 
        pinyin_sample_mode='sort',  # special
        weight=[1., 0, .5],   # pinyin score, word freq score
        conf_size=300, 
        debug=False)


if __name__ == "__main__":
    debug = True
    conf = Confusor(
        cand_pinyin_num=100, 
        cos_threshold=(0., 1.), 
        method='all-similar single-freedom', 
        token_sample_mode='sort', 
        pinyin_sample_mode='sort',  # special
        weight=[1., 0, .2],   # pinyin score, word freq score
        conf_size=300, 
        debug=debug)

    print(conf.compute_RED(['huang','shang','huang'], ['huang','shan','huang']))

    print(conf('公'))
    print(conf('司'))
    print(conf('业'))
    print(conf('工商'))
    print(conf('出生'))
    print(conf('这边'))
    print(conf('短期'))
    print(conf('超短期'))
    print(conf('不确定性'))
    # print(conf('特色旅游'))
    # print(conf('!@#$'))   # test 
    print(conf.get_pinyin_sequence_single_freedom('短期', debug=debug))

