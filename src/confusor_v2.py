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

# requirements
import time
import pypinyin
import Pinyin2Hanzi
from tqdm import tqdm
from pprint import pprint
from src.input_sequence_manager import InputSequenceManager

# paths
from paths import (
    to_path,
    CONFUSION_PATH, RED_SCORE_PATH,
    VOCAB_PATH, CHAR_PY_VOCAB_PATH,
    PATH_FREQ_GNR_CHAR, PATH_FREQ_GNR_WORD,
    PATH_FREQ_FIN_CHAR, PATH_FREQ_FIN_WORD,
    SCORE_DATA_DIR, EMBEDDING_PATH)

# utils
from utils import (
    load_vocab,
    Py2HzUtils,
    save_pkl, load_pkl,
    edit_distance, 
    is_chinese_char,
    is_pure_chinese_phrase)

from src.confusor_utils import (
    refined_edit_distance,
)


class Confusor(object):
    IME_PAGE_SIZE = 8
    RED_THRESHOLD = 0.4
    PUNC_LIST = "，；。？！…"

    def __init__(self, debug=False):
        # for debugging
        self.debug = debug
        self.timer = []

        # cache for faster function call on phrases
        self.confusor_cache = {}  # input_str: {word_size: list of words}
        self.save_flag_confusor = False
        self.red_score_cache = {}  # (str1, str2): float
        self.save_flag_red = False
        self.path_to_confusor_cache = CONFUSION_PATH
        self.path_to_red_score_cache = RED_SCORE_PATH

        # data loader
        self._load_confusor_data()
        self._load_frequency_data()

        # to obtain input sequences
        self.ism = InputSequenceManager()
        
    def _load_confusor_data(self):
        self.vocab = load_vocab(VOCAB_PATH)
        self.py_vocab = load_vocab(CHAR_PY_VOCAB_PATH)

        # load char-level REDscore matrix: red_score[str1][str2]
        self.red_score = load_pkl(to_path(SCORE_DATA_DIR, 'red_matrix.pkl'))
        self.del_matrix = load_pkl(to_path(SCORE_DATA_DIR, 'del_matrix.pkl'))
        self.rep_matrix = load_pkl(to_path(SCORE_DATA_DIR, 'rep_matrix.pkl'))

        # load red scores & confusor for longer phrases
        if os.path.exists(self.path_to_red_score_cache):
            self.red_score_cache = load_pkl(self.path_to_red_score_cache)
            print(f"Loaded {len(self.red_score_cache)} items from {self.path_to_red_score_cache}")
        if os.path.exists(self.path_to_confusor_cache):
            self.confusor_cache = load_pkl(self.path_to_confusor_cache)
            print(f"Loaded {len(self.confusor_cache)} items from {self.path_to_confusor_cache}")

    def _load_frequency_data(self):
        self.freq_general_char = load_pkl(PATH_FREQ_GNR_CHAR)
        self.freq_general_word = load_pkl(PATH_FREQ_GNR_WORD)
        self.freq_findoc_char = load_pkl(PATH_FREQ_FIN_CHAR)
        self.freq_findoc_word = load_pkl(PATH_FREQ_FIN_WORD)
        if None in [self.freq_general_char, self.freq_general_word, 
                    self.freq_findoc_char, self.freq_findoc_word]:
            self.freq_general_char = self.freq_general_char or {}
            self.freq_general_word = self.freq_general_word or {}
            self.freq_findoc_char = self.freq_findoc_char or {}
            self.freq_findoc_word = self.freq_findoc_word or {}
            self.frequency_loaded = False
        else:
            self.frequency_loaded = True

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        if self.debug:
            self.timer.append((information, time.time()))

    def print(self, *args):
        if self.debug:
            pprint(*args)

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
        # lower is better candidate
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
        self.save_flag_red = True
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
        for py1 in [''.join(pinyin)] + simpy:
            if has_mistyped:
                for py2 in [''.join(pinyin)] + input_sequences:
                    red_score = self.refined_edit_distance(py1, py2, rate=True)
                    ret[py2] = min(red_score, ret.get(py2, 1.0))
            else:  # people only type in simplifed pinyin, without typos.
                ret[py1] = 0.0  
        ret = [(k, v) for k, v in ret.items() if v < self.RED_THRESHOLD]
        # return a list of (pinyin string, score) pairs
        return sorted(ret, key=lambda x: x[1])

    def get_candidates(self, input_sequence, ngram=None):
        """
        @param input_sequence: a string of pinyin / simplified pinyin
        @param ngram: the ngram of the output phrase
        """
        _inp = input_sequence
        self.confusor_cache.setdefault(_inp, {})
        if self.confusor_cache[_inp].get(ngram) is None:
            candidates = self.ism.to_candidates(_inp, ngram=ngram)
            # save in cache 
            self.confusor_cache[_inp][ngram] = candidates
            self.save_flag_confusor = True
        else:
            # candidates = self.ism.to_candidates(_inp, ngram=ngram)  # for warm-up
            candidates = self.confusor_cache[_inp][ngram]
        return candidates
    
    def get_frequency_score(self, phrase, source='general'):
        # higher is better candidate
        score = 0.
        if not self.frequency_loaded:
            return score
        if len(phrase) == 1:
            gnr_score = self.freq_general_char.get(phrase, 0.)
            fin_score = self.freq_findoc_char.get(phrase, 0.)
        else:
            gnr_score_char = self.freq_general_char.get(phrase, 0.)
            gnr_score_word = self.freq_general_word.get(phrase, 0.)
            if gnr_score_char < 0.5 or gnr_score_word < 0.5:
                gnr_score = max(gnr_score_char, gnr_score_word)
            else:  # harmonic mean
                gnr_score = 1./ (1. / gnr_score_char + 1./ gnr_score_word)
            fin_score_char = self.freq_findoc_char.get(phrase, 0.)
            fin_score_word = self.freq_findoc_word.get(phrase, 0.)
            if fin_score_char < 0.5 or fin_score_word < 0.5:
                fin_score = max(fin_score_char, fin_score_word)
            else:  # harmonic mean
                fin_score = 1./ (1. / fin_score_char + 1./ fin_score_word)
        # return score value for ranking, from these scores        
        if source in ['general', 'gnr']:
            return gnr_score
        if source in ['findoc', 'fin']:
            return fin_score
        if source in ['delta']:
            if gnr_score > fin_score > 0.: 
                score = gnr_score - fin_score  # delta
            elif fin_score > .8 and gnr_score > .8:
                score = max(gnr_score, fin_score)  # max
        return score

    def get_ime_rank_score(self, rank=None, rank_str=None, old_version=True):
        # lower is better candidate
        if rank_str is not None:
            rank = list(map(int, rank_str.split('-')))
        if old_version:
            return sum(rank) * 0.1
        if len(rank) == 1:  # with one selection
            rank_penity = rank[0] // self.IME_PAGE_SIZE * 0.1
        else:
            rank_penity = sum([1. - (i / (r+1)) for i, r in enumerate(rank)]) * 0.01
        return -rank_penity
    
    def get_confusion_set(self, cand_pinyins, ngram=None, sort=True):
        """
        @param pinyins: a list of (pinyin string, score) pairs
        @return: a list of (phrase string, score) pairs.
        """
        cfs = {}
        for py, pinyin_distance in cand_pinyins:
            # cand_list = self.ism.to_candidates(py, ngram=ngram)
            cand_list = self.get_candidates(py, ngram=ngram)
            for cand, rank_str in cand_list:
                rank_penity = self.get_ime_rank_score(rank_str=rank_str)  # negative value
                if rank_penity > 0.9: continue
                # if self.frequency_loaded:
                #     freq_score = self.get_frequency_score(cand)
                cfs[cand] = min(pinyin_distance + rank_penity, 
                                cfs.get(cand, 999))
        if sort:
            ret = sorted(cfs.items(), key=lambda x: x[1])
        else:
            ret = [(k, v) for k, v in cfs.items()]
        return ret

    def warmup_ism_memory(self):
        # pre-calculate, save in memory files.

        for _py1 in tqdm(self.py_vocab):
            if _py1.startswith('['):
                continue
            simpy = self.ism.simpy([_py1])
            similiar_input_sequences = self.ism.get_input_sequence(
                word=None, pinyin=[_py1], simp_candidates=True)
            for _sp in simpy:
                red = self.refined_edit_distance(_py1, _sp, rate=False)
                candidates = self.get_candidates(_sp, ngram=1)
        self.save_memory()

        for _py1 in tqdm(self.py_vocab):
            if _py1.startswith('['):
                continue
            for _py2 in self.py_vocab:
                if _py2.startswith('['):
                    continue
                red = self.refined_edit_distance(_py1, _py2, rate=False)
                similiar_input_sequences = self.ism.get_input_sequence(
                    word=None, pinyin=[_py1, _py2], simp_candidates=True)
                simpy = self.ism.simpy([_py1, _py2])
                for _sp in [_py1 + _py2] + simpy:
                    candidates = self.get_candidates(_sp, ngram=2)
            self.save_memory()

    def save_memory(self):
        self.ism.save_memory()
        if self.save_flag_confusor:
            save_pkl(
                self.confusor_cache, 
                self.path_to_confusor_cache)
            print(f"Saved confusor memory in {self.path_to_confusor_cache}")
            self.save_flag_confusor = False
        if self.save_flag_red:
            save_pkl(
                self.red_score_cache, 
                self.path_to_red_score_cache)
            print(f"Saved red memory in {self.path_to_red_score_cache}")
            self.save_flag_confusor = False

    def __call__(self, word, context=None, has_mistyped=False, debug=None, return_score=False):
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
        self.print(pinyins)
        if pinyins is None:
            return [word]  # return itself if not a Chinese word

        # get simliar pinyin sequences of the word
        cand_pinyins = self.get_similar_pinyins(
            pinyins, has_mistyped=has_mistyped)
        self.record_time('get pinyin sequence')
        self.print(cand_pinyins)

        # get the confusion set with the same/similar pinyin sequence.
        confusion_set = self.get_confusion_set(
            cand_pinyins=cand_pinyins, 
            ngram=len(word), sort=True)
        self.record_time('get confusion set')
        self.print(confusion_set)
        
        # select some of them as the output.
        # self.save_memory()
        if return_score:
            return confusion_set
        return [k for k, v in confusion_set]


if __name__ == "__main__":
    cfs = Confusor()
    ret = cfs('短裙', return_score=True)
    print(cfs.timer)
    # print(ret)
    # cfs.warmup_ism_memory()
    # cfs.save_memory()
    # cfs.ism.update_memory_from_tmp()
