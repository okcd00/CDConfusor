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
    REPO_DIR, CONFUSOR_DATA_DIR, 
    SCORE_DATA_DIR,
    EMBEDDING_PATH
)

# requirements
import time
import pypinyin
from tqdm import tqdm
from pprint import pprint

# utils
from utils import (
    edit_distance, 
    is_chinese_char,
    is_pure_chinese_phrase,
    load_pkl)
from src.confusor_utils import (
    char2idx, 
    refined_edit_distance,
    amb_del_mat, amb_rep_mat, ins_rep_mat
)


class Confusor(object):
    PUNC_LIST = "，；。？！…"

    def __init__(self):
        # for debugging
        self.debug = True
        self.timer = []

        # data loader
        self._load_confusor_data()

        # cache for faster function call
        self.red_score_cache = {}
        
    def _load_confusor_data(self):
        # load char-level REDscore matrix
        self.red_score = load_pkl(to_path(SCORE_DATA_DIR, 'red_data.pkl'))
        # self.del_matrix = load_pkl(to_path(SCORE_DATA_DIR, 'amb_data.pkl'))['del_mat']
        # self.rep_matrix = load_pkl(to_path(SCORE_DATA_DIR, 'amb_data.pkl'))['rep_mat']
        self.del_matrix, self.rep_matrix = amb_del_mat, amb_rep_mat

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        if self.debug:
            self.timer.append((information, time.time()))

    def get_pinyin(self, char, heteronym=False):
        """
        @param char: a Chinese character
        @return: a list of pinyin strings
        """
        if not is_chinese_char(ord(char)):
            return None
        pinyins = pypinyin.pinyin(
            char, style=pypinyin.NORMAL, 
            heteronym=heteronym)
        return pinyins[0]

    def get_pinyin_list(self, word, heteronym=False):
        """
        @param word: a Chinese word
        @return: a list of pinyin strings
        """
        if not is_pure_chinese_phrase(word):
            return None
        pinyins = pypinyin.pinyin(
            word, style=pypinyin.NORMAL, 
            heteronym=heteronym)
        return pinyins

    def get_similar_pinyins(self, pinyins):
        """
        @param pinyins: a list of pinyin strings
        @return: a list of (pinyin string, score) pairs.
        """
        return []

    def get_confusion_set(self, pinyins):
        """
        @param pinyins: a list of (pinyin string, score) pairs
        @return: a list of (phrase string, score) pairs.
        """
        return []

    def refined_edit_distance(self, seq1, seq2):
        """
        @param seq1: a list of pinyin strings
        @param seq2: a list of pinyin strings
        @return: the red score of the two sequences
        """
        return refined_edit_distance(
            seq1, seq2, self.del_matrix, self.rep_matrix)

    def __call__(self, word, debug=None):
        """
        input a word, return its word-level confusion (a list of words).
        """

        # for debugging
        if debug is not None:
            self.debug = debug
        self.record_time('start')
        
        # get the pinyin list of the word (maybe hypernyms)
        pinyins = self.get_pinyin_list(word, True)        
        self.record_time('get pinyin list')

        # get the pinyin sequence of the word
        cand_pinyins = self.get_similar_pinyins(pinyins)
        if self.debug:
            # pprint(sorted(cand_pinyins, key=lambda x: x[1]))
            pprint(list(map(lambda x: x[0], 
                   sorted(cand_pinyins, key=lambda x: x[1]))))
            self.record_time('get pinyin sequence')

        # get the confusion set with the same/similar pinyin sequence.
        confusion_set = self.get_confusion_set(cand_pinyins)
        if self.debug:
            self.record_time('get confusion set')
        
        # select some of them as the output.
        return confusion_set


if __name__ == "__main__":
    cfs = Confusor()
    print(cfs.get_pinyin_list('传说', True))
    pass
