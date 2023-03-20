# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2023 All rights reserved.
#
#   filename : preprocess_red_embeddings.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-03-18
#   desc     : generate intermediate files
#              for input sequences mapping
# ==========================================================================
import os
import sys
sys.path.append('./')
sys.path.append('../')
import json
import time
import httpx
import requests
import pypinyin
import Pinyin2Hanzi


# PATH (path to custom-files)
from paths import (
    SCORE_DATA_DIR, TMP_DIR,
    PY_MAPPING_PATH, 
    MEMORY_PATH, IME_MEMORY_PATH,
    CHAR_PY_VOCAB_PATH,)

# utils
from utils import (
    load_json, dump_json,
    load_kari, save_kari, 
    load_vocab, flatten,
    Py2HzUtils)


class InputSequenceManager(object):
    # interval network routine, invalid for outside
    proxies = 'socks5://100.64.0.15:11081'  
    memory_dir = SCORE_DATA_DIR
    URL_GOOGLE_IME_API = "https://inputtools.google.com/request?" + \
        "text={}&itc=zh-t-i0-pinyin&num=20&cp=1&cs=1&ie=utf-8&oe=utf-8&app=demopage"
    
    def __init__(self):
        self.char_vocab = load_vocab(CHAR_PY_VOCAB_PATH)
        self.py_mapping = load_json(f"{PY_MAPPING_PATH}")

        self.memory = {}  # pinyin -> [input_sequence1, input_sequence2, ...]
        self.save_flag = False
        self.ime_memory = {}  # input_sequence -> [word1, word2, ...]
        self.ime_save_flag = False
        self.init_memory()  
        self.init_google_ime()

        self.ph_util = Py2HzUtils()
        self.is_chinese = Pinyin2Hanzi.is_chinese

    def load_memory(self, memory_path=None):
        # ime_memory.google.json
        if os.path.exists(memory_path):
            if MEMORY_PATH.endswith('.json'):
                dic = load_json(memory_path)
            elif MEMORY_PATH.endswith('.kari'): 
                dic = load_kari(memory_path)
            for k, v in dic.items():
                self.memory[k] = sorted(set(v + self.memory[k]))

    def init_memory(self):
        # load recorded google IME memory
        if os.path.exists(MEMORY_PATH):  
            self.load_memory(MEMORY_PATH)

    def load_ime_memory(self, ime_memory_path=None):
        dic = load_json(ime_memory_path)
        self.ime_memory.update(dic)

    def init_google_ime(self):
        # load recorded google IME memory
        if os.path.exists(IME_MEMORY_PATH):  # ime_memory.google.json
            self.load_ime_memory(IME_MEMORY_PATH)
        # online google IME initialization.
        from httpx_socks import SyncProxyTransport, AsyncProxyTransport
        self.transport = SyncProxyTransport.from_url(self.proxies)

    def save_memory(self, force=False):
        if not os.path.exists(f"{TMP_DIR}"):
            os.system(f"mkdir -p {TMP_DIR}")
        if self.save_flag or force:
            fp = MEMORY_PATH.split('/')[-1]
            if fp.endswith('.json'):
                dump_json(self.memory, f"{TMP_DIR}/{fp}")
            elif fp.endswith('.kari'):
                save_kari(self.memory, f"{TMP_DIR}/{fp}")
        if self.ime_save_flag or force:
            fp = IME_MEMORY_PATH.split('/')[-1]
            if fp.endswith('.json'):
                dump_json(self.ime_memory, f"{TMP_DIR}/{fp}")
            elif fp.endswith('.kari'):
                save_kari(self.ime_memory, f"{TMP_DIR}/{fp}")
        print("Saved ISM memory.", time.ctime())
        
    def update_memory_from_tmp(self):
        if os.path.exists(f"{TMP_DIR}/memory.json"):
            os.system(f"mv {MEMORY_PATH} {MEMORY_PATH}.bak")
            os.system(f"cp {TMP_DIR}/memory.json {MEMORY_PATH}")
        if os.path.exists(f"{TMP_DIR}/ime_memory.json"):
            os.system(f"mv {IME_MEMORY_PATH} {IME_MEMORY_PATH}.bak")
            os.system(f"cp {TMP_DIR}/ime_memory.google.json {IME_MEMORY_PATH}")

    def py2phrase(self, pinyins):
        """
        pinyins: list of pinyin
        return: list of Chinese words
        """
        res = self.ph_util.to_hanzi(
            pinyin_tuple=pinyins, 
            n_candidates=20, 
            with_rank=False)
        return [w for w in res 
                if all([c in self.char_vocab for c in w])]

    def simpy(self, pinyin):
        # simplified input sequences from the pinyin
        # simplify inputs with first char: xujiali -> [xjl, xujl, xujial]
        if isinstance(pinyin[0], list):
            pinyin = [p[0] for p in pinyin]
        input_sequences = []
        for simp_idx, _ in enumerate(pinyin):
            input_sequences.append(''.join(
                [py[0] if idx >= simp_idx else py for idx, py in enumerate(pinyin)]))
        return input_sequences

    def get_similar_input_sequences(self, pinyin):
        # get_similar_input_sequences
        if isinstance(pinyin[0], list):
            pinyin = [p[0] for p in pinyin]
        if pinyin.__len__() == 1:
            if pinyin[0] in self.py_mapping:
                return self.py_mapping[pinyin]
            else:
                print("Invalid pinyin", pinyin)
                return pinyin
        input_sequences = set()
        input_sequences.add(''.join(pinyin))
        # errors in single char
        for err_idx, err_py in enumerate(pinyin):
            for py_cand in self.py_mapping[err_py]:
                input_sequences.add(
                    f''.join([py_cand if idx == err_idx else py 
                              for idx, py in enumerate(pinyin)]))
        for _sp in self.simpy(pinyin):
            input_sequences.add(_sp)
        return sorted(input_sequences)

    def get_input_sequence(self, word=None, pinyin=None, simp_candidates=True):
        if pinyin is None:
            pinyin = pypinyin.pinyin(word, style=pypinyin.NORMAL)
            # Luo: only take the first pinyin candidate in sentence
            # TODO: adapt for heteronym
        if isinstance(pinyin[0], list):
            pinyin = [p[0] for p in pinyin]
        pinyin_str = ''.join(pinyin)
        if simp_candidates:
            if pinyin_str in self.memory:
                input_sequences = self.memory[pinyin_str]
            else:
                input_sequences = self.get_similar_input_sequences(pinyin)
                self.memory[pinyin_str] = input_sequences
                self.save_flag = True
        else:
            input_sequences = [pinyin_str]
        return pinyin, input_sequences

    def request_google_ime(self, url):
        candidates = []
        for i in range(10):
            try:
                # proxies={'http':'http://127.0.0.1:1081'}
                result = requests.get(url, proxies=self.proxies)
                # print(result)
            except:
                if i >= 9:
                    return candidates
                else:
                    time.sleep(0.5)
            else:
                time.sleep(0.1)
                break
        
        rjson = result.json()
        # print(rjson)
        return rjson

    def client_post_google_ime(self, url):
        try:
            with httpx.Client(transport=self.transport) as client:
                resp = client.post(url, timeout=5 * 60)                
                return resp.json()
        except Exception as e:
            import logging
            logging.exception(e)
            return "请求失败，请重试"

    def _from_online_ime(self, input_sequence, method='client'):
        url = self.URL_GOOGLE_IME_API.format(input_sequence)
        if method in ['request']:
            rjson = self.request_google_ime(url)
        elif method in ['client']:
            rjson = self.client_post_google_ime(url)
        if rjson[0] == 'SUCCESS':
            # 424it [03:43,  1.90it/s]
            candidates = rjson[1][0][1]
            cn_tags = rjson[1][0][3]["lc"]
            candidates = [c for i, c in enumerate(candidates) 
                            if list(set(map(int, cn_tags[i].split()))) == [16]]
        return candidates

    def to_candidates(self, input_sequence):
        if input_sequence in self.ime_memory:
            return self.ime_memory[input_sequence]
        # self.py2word() only takes complete pinyin lists.
        candidates = self._from_online_ime(input_sequence)
        self.ime_memory[input_sequence] = candidates  # update
        self.ime_save_flag = True
        return candidates

    def __del__(self):
        if self.save_flag or self.ime_save_flag:
            self.save_memory()

    def __call__(self, word=None, pinyin=None):
        # return similar input sequence with the input string
        pinyin, input_sequences = self.get_input_sequence(
            word=word, pinyin=pinyin, 
            simp_candidates=True)
        return pinyin, input_sequences


if __name__ == "__main__":
    ism = InputSequenceManager()
    # print(ism.to_candidates('chend'))
    print(ism.get_input_sequence('陈点'))
    pass
