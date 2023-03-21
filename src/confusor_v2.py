# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2023 All rights reserved.
#
#   filename : confusor.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-02-03
#   desc     : return a list of candidate words/phrases
#              for generating better augmented CSC samples
#   comments : a more concise revised version.
# ==========================================================================
import os
import re
import sys
sys.path.append("./")
sys.path.append("../")

# paths
from paths import (
    to_path,
    VOCAB_PATH, CHAR_PY_VOCAB_PATH,
    SCORE_DATA_DIR, EMBEDDING_PATH
)

# requirements
import time
import pypinyin
import Pinyin2Hanzi
from tqdm import tqdm
from pprint import pprint
from src.input_sequence_manager import InputSequenceManager

# utils
from utils import (
    load_vocab,
    Py2HzUtils,
    save_pkl, load_pkl,
    save_kari, load_kari,
    edit_distance, 
    is_chinese_char,
    is_pure_chinese_phrase)
from src.confusor_utils import (
    refined_edit_distance,
)


class Confusor(object):
    RED_THRESHOLD = 0.4
    PUNC_LIST = "，；。？！…"

    def __init__(self):
        # for debugging
        self.debug = True
        self.timer = []

        # cache for faster function call on phrases
        self.red_score_cache = {}  # (str1, str2): float
        self.update_flag_red = False
        self.path_to_red_score_cache = os.path.join(
            SCORE_DATA_DIR, 'red_cache.pkl')

        # data loader
        self._load_confusor_data()

        # to obtain input sequences
        self.ism = InputSequenceManager()
        
    def _load_confusor_data(self):
        self.vocab = load_vocab(VOCAB_PATH)
        self.py_vocab = load_vocab(CHAR_PY_VOCAB_PATH)

        # load char-level REDscore matrix: red_score[str1][str2]
        self.red_score = load_pkl(to_path(SCORE_DATA_DIR, 'red_matrix.pkl'))
        self.del_matrix = load_pkl(to_path(SCORE_DATA_DIR, 'del_matrix.pkl'))
        self.rep_matrix = load_pkl(to_path(SCORE_DATA_DIR, 'rep_matrix.pkl'))

        # load red scores for longer phrases
        if os.path.exists(self.path_to_red_score_cache):
            self.red_score_cache = load_pkl(self.path_to_red_score_cache)

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        if self.debug:
            self.timer.append((information, time.time()))

    def get_pinyin(self, char, heteronym=False):
        """
        @param char: a single Chinese character
        @return: a list of pinyin strings
        """
        if not is_chinese_char(ord(char)):
            return None
        pinyins = pypinyin.pinyin(
            char,  # '了'
            style=pypinyin.NORMAL, 
            heteronym=heteronym)
        return pinyins[0]  # [le] or [le, liao]

    def get_pinyin_list(self, word, context=None, heteronym=True):
        """
        @param word: a Chinese word
        @param context: the sentence which the Chinese word is in
        @return: a list of pinyin strings
        """
        if not is_pure_chinese_phrase(word):
            return None
        pinyins = pypinyin.pinyin(
            word, style=pypinyin.NORMAL, 
            heteronym=heteronym)
        return pinyins

    def edit_distance(self, seq1, seq2, rate=True):
        """
        @param seq1: a list of pinyin strings
        @param seq2: a list of pinyin strings
        @return: the edit_distance of the two sequences
        """
        return edit_distance(seq1, seq2, rate=rate)

    def refined_edit_distance(self, seq1, seq2, rate=False):
        """
        @param seq1: a list of pinyin strings
        @param seq2: a list of pinyin strings
        @return: the red score of the two sequences
        """
        if (seq1 in self.py_vocab) and (seq2 in self.py_vocab):
            try:
                return self.red_score[seq1][seq2]
            except:
                print(f"({seq1}, {seq2}) not in red_score matrix.")
        if (seq1, seq2) in self.red_score_cache:
            return self.red_score_cache[(seq1, seq2)]
        red_score = refined_edit_distance(
            seq1, seq2, 
            del_matrix=self.del_matrix, 
            rep_matrix=self.rep_matrix, 
            rate=False)
        self.red_score_cache[(seq1, seq2)] = red_score
        if rate:
            return red_score / max(len(seq1), len(seq2))
        return red_score

    def get_similar_pinyins(self, pinyin, has_mistyped=True):
        """
        @param pinyins: a list of pinyin strings
        @return: a list of candidate input_sequences, tuple(pinyin string, ime_rank).
        """
        ret = {}
        simpy = self.ism.simpy(pinyin)
        pinyin, input_sequences = self.ism(word=None, pinyin=pinyin)
        for py1 in simpy:
            for py2 in [''.join(pinyin)] + input_sequences:
                red_score = self.refined_edit_distance(py1, py2, rate=True)
                ret[py2] = min(red_score, ret.get(py2, 1.0))
        ret = [(k, v) for k, v in ret.items() if v < self.RED_THRESHOLD]
        return sorted(ret, key=lambda x: x[1])

    def get_confusion_set(self, cand_pinyins, sort=False):
        """
        @param pinyins: a list of (pinyin string, score) pairs
        @return: a list of (phrase string, score) pairs.
        """
        if not sort:
            cfs = {}
            for py, score in cand_pinyins:
                for cand in self.ism.to_candidates(py):
                    cfs[cand] = min(score, cfs.get(cand, 999))
        return sorted(cfs.items(), key=lambda x: x[1])

    def warmup_ism_memory(self):
        for _py1 in tqdm(self.py_vocab):
            if _py1.startswith('['):
                continue
            simpy = self.ism.simpy([_py1])
            similiar_input_sequences = self.ism.get_input_sequence(
                word=None, pinyin=[_py1], simp_candidates=True)
            for _sp in simpy:
                red = self.refined_edit_distance(_py1, _sp, rate=False)
                candidates = self.ism.to_candidates(_sp, ngram=1)
        self.save_memory()

        for _py1 in tqdm(self.py_vocab):
            if _py1.startswith('['):
                continue
            for _py2 in tqdm(self.py_vocab):
                if _py2.startswith('['):
                    continue
                red = self.refined_edit_distance(_py1, _py2, rate=False)
                similiar_input_sequences = self.ism.get_input_sequence(
                    word=None, pinyin=[_py1, _py2], simp_candidates=True)
                simpy = self.ism.simpy([_py1, _py2])
                for _sp in simpy:
                    candidates = self.ism.to_candidates(_sp, ngram=2)
            self.save_memory()

    def save_memory(self):
        self.ism.save_memory()
        save_pkl(
            self.red_score_cache, 
            self.path_to_red_score_cache)

    def __del__(self):
        if self.update_flag_red:
            self.save_memory()
        self.update_flag_red = False

    def __call__(self, word, context=None, debug=None):
        """
        input a word, return its word-level confusion (a list of words).
        """

        # for debugging
        if debug is not None:
            self.debug = debug
        self.record_time('start')
        
        # get the pinyin list of the word (maybe heteronym)
        pinyins = self.get_pinyin_list(
            word, context=context, heteronym=False)  
        self.record_time('get pinyin list')
        if self.debug:
            pprint(pinyins)
        if pinyins is None:
            return [word]  # return itself if not a Chinese word

        # get simliar pinyin sequences of the word
        cand_pinyins = self.get_similar_pinyins(pinyins)
        self.record_time('get pinyin sequence')
        if self.debug:
            pprint(cand_pinyins)

        # get the confusion set with the same/similar pinyin sequence.
        confusion_set = self.get_confusion_set(cand_pinyins)
        self.record_time('get confusion set')
        if self.debug:
            pprint(confusion_set)
        
        # select some of them as the output.
        return confusion_set


if __name__ == "__main__":
    cfs = Confusor()
    # cfs('陈点')
    # cfs.ism.update_memory_from_tmp()
    cfs.warmup_ism_memory()
