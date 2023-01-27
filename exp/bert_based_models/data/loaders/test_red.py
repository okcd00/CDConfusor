import os
import sys
sys.path.append("/home/chendian/Confusor")  # project path

import re
import time
import pickle
from bbcm.utils import edit_distance
from bbcm.utils.pinyin_utils import PinyinUtils
from bbcm.utils.confuser_utils import (
    generate_score_matrix,
    complete_ziRED,
    complete_zi_sim_matrix
)

 
CONFUSOR_DATA_DIR = '/data/chendian/'  # c10

SCORE_MAT_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/score_data/'
EMBEDDING_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/sound_tokens/'

SIMILAR_PINYIN_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/similar_pinyins.pkl'
FZIMU2PINYIN_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/fzimu2pinyin.pkl'
REDSCORE_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/ziREDscore.pkl'
FZIMURED_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/fzimuREDscore.pkl'
ZISIMMATRIX = f'{CONFUSOR_DATA_DIR}/tencent_embedding/zi_sim_matrix.pkl'

BERT_VOCAB_PATH = f'{CONFUSOR_DATA_DIR}/pretrained_bert_models/chinese-macbert-base/vocab.txt'
SIGHAN_CFS_PATH = f'{CONFUSOR_DATA_DIR}/sighan_confusion.txt'
PINYIN_MAPPING_PATH = f'{CONFUSOR_DATA_DIR}/pinyin_mapping.json'


def get_filesize(fp):
    # '''获取文件的大小,结果保留两位小数，单位为MB'''
    # fp = unicode(fp, 'utf8')
    fsize = os.path.getsize(fp)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


class TesterRED(object):
    PUNC_LIST = "，；。？！…"
    FIRST_PINYIN_PENALTY = 0.05

    def __init__(self, amb_score=0.5, inp_score=0.25, cos_threshold=(0.1, 0.5), method='beam', cand_pinyin_num=10,
                 cand_fzimu_num=100, cand_dcc_num=50, cand_zi_num=50, keep_num=500, weight=(1, 0.5, 1), conf_size=10,
                 pinyin_sample_mode='sort', token_sample_mode='special', debug=False):
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

        print("Using {} method.".format(method))
        print("Pinyin sampling mode: {}.".format(pinyin_sample_mode))
        print("Token sampling mode: {}.".format(token_sample_mode))

        self.bert_vocab = [line.strip() for line in open(BERT_VOCAB_PATH, 'r')]
        self.char_confusion_set = {}
        self.word_confusion_set = {}  # a function is better than a dict
        self.external_word_set = {}

        # self.load_sighan_confusion_set()
        self.load_word_confusion_set()
        if 'two-stage' in self.method:
            self.log("Now loading fzimu2pinyin corpus.")
            self.fzimu2pinyin = self.load_pickle(FZIMU2PINYIN_PATH)

        
        self.log("Now loading REDscore:")
        self.ziREDscore = self.load_pickle(REDSCORE_PATH)
        if 'two-stage' in self.method:
            self.log("Now loading fzimuREDscore:")
            self.fzimuREDscore = self.load_pickle(FZIMURED_PATH)

        # load and generate the score matrix
        self.log("Now generating score matrix.")
        self.score_matrix = self.load_score_matrix()

        print("Instance initial ends.\n")

    def record_time(self, information):
        """
        @param information: information that describes the time point.
        """
        self.timer.append((information, time.time()))

    def log(self, text):
        if self.debug:
            print(text)

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

    def compute_RED(self, pylist1, pylist2, detail=False):
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
        
        red_score = sum(scores) / max([len(''.join(pylist1)), len(''.join(pylist2))])

        if detail:
            return scores, red_score
        return red_score

    def get_pinyin_list(self, token):
        # return a list of pinyin 
        # ['gong'] / ['chao', 'duan', 'qun']
        pinyins = self.pu.to_pinyin(token)
        if re.match('^[a-z]+$', ''.join(pinyins)):
            return pinyins
        else:
            return None

    def pinyin_retrieval_recall_evaluator(self, token, evaluate_num, **args):
        self.timer = []
        self.record_time('start')
        cand_pinyins = self.get_pinyin_sequence(
            token, cand_pinyin_num=evaluate_num, **args)
        self.record_time('pinyin retrieval')
        return self.timer, cand_pinyins

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

    def word_pinyin_similarity(self, word_1, word_2):
        return 1. - self.compute_RED(
            self.pu.to_pinyin(word_1), 
            self.pu.to_pinyin(word_2))

    def __call__(self, word_1, word_2):
        if self.debug:
            self.timer = []
            self.record_time('start')

        pinyin_1 = self.pu.to_pinyin(word_1) 
        pinyin_2 = self.pu.to_pinyin(word_2)

        
        scores, red_score = self.compute_RED(
            pinyin_1, pinyin_2, detail=True)
        similarity_value = 1. - red_score

        print(f"{word_1}\t{pinyin_1}")
        print(f"{word_2}\t{pinyin_2}")
        print(f"Similarity:", similarity_value)
        print(f"char-level detail:", scores)
        return similarity_value


if __name__ == "__main__":
    conf = TesterRED(debug=False)
    conf('黄山黄', '煌上煌')
