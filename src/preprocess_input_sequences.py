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
import pickle
import requests
import pypinyin
import numpy as np

# PATH (path to custom-files)
from paths import (
    SCORE_DATA_DIR, TMP_DIR,
    PY_MAPPING_PATH, 
    MEMORY_PATH, IME_MEMORY_PATH,
    CHAR_PY_VOCAB_PATH,
)


def flatten(l):
    return [item for sublist in l for item in sublist]


class InputSequenceManager(object):
    # interval network routine, invalid for outside
    proxies = 'socks5://100.64.0.15:11081'  
    memory_dir = SCORE_DATA_DIR
    URL_GOOGLE_IME_API = "https://inputtools.google.com/request?" + \
        "text={}&itc=zh-t-i0-pinyin&num=20&cp=1&cs=1&ie=utf-8&oe=utf-8&app=demopage"
    
    def __init__(self):
        self.char_vocab = [line.strip() for line in open(CHAR_PY_VOCAB_PATH, 'r')]
        self.py_mapping = json.load(open(f"{PY_MAPPING_PATH}", 'r'))

        self.memory = {}  # 
        self.ime_memory = {}  #
        self.init_memory()
        self.init_google_ime()

    def load_memory(self, memory_path=None):
        dic = json.load(open(memory_path, 'r'))
        self.memory.update(dic)

    def init_memory(self):
        # load recorded google IME memory
        if os.path.exists(MEMORY_PATH):  # ime_memory.google.json
            self.load_memory(MEMORY_PATH)

    def load_ime_memory(self, ime_memory_path=None):
        dic = json.load(open(ime_memory_path, 'r'))
        self.ime_memory.update(dic)

    def init_google_ime(self):
        # load recorded google IME memory
        if os.path.exists(IME_MEMORY_PATH):  # ime_memory.google.json
            self.load_ime_memory(IME_MEMORY_PATH)
        # online google IME initialization.
        from httpx_socks import SyncProxyTransport, AsyncProxyTransport
        self.transport = SyncProxyTransport.from_url(self.proxies)

    def save_memory(self):
        os.system(f"mkdir -p {TMP_DIR}")
        json.dump(self.memory, 
                  open(f'{TMP_DIR}/memory.json', 'w'))
        json.dump(self.ime_memory, 
                  open(f'{TMP_DIR}/ime_memory.google.json', 'w'))
        
    def update_memory_from_tmp(self):
        os.system(f"mv -p {self.memory_dir}; cp {TMP_DIR}/memory.json {MEMORY_PATH}")
        os.system(f"mv -p {self.memory_dir}; cp {TMP_DIR}/ime_memory.google.json {IME_MEMORY_PATH}")

    def simpy(self, pinyin):
        # get_similar_input_sequences
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
        # simplify inputs with first char
        for err_idx, err_py in enumerate(pinyin):
            input_sequences.add(''.join(
                [py[0] if idx >= err_idx else py for idx, py in enumerate(pinyin)]))
        return sorted(input_sequences)

    def get_input_sequence(self, word, simp_candidates=True):
        pinyin = pypinyin.pinyin(word, style=pypinyin.NORMAL)
        # Luo: only take the first pinyin candidate in sentence
        pinyin = [p[0] for p in pinyin]  
        if simp_candidates:
            input_sequences = self.simpy(pinyin)
        else:
            input_sequences = [''.join(pinyin)]
        return input_sequences

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

    def to_candidates(self, input_sequence):
        url = self.URL_GOOGLE_IME_API.format(input_sequence)
        if input_sequence in self.ime_memory:
            return self.ime_memory[input_sequence]
        rjson = self.request_google_ime(url)
        # rjson = self.client_post_google_ime(url)
        if rjson[0] == 'SUCCESS':
            # 424it [03:43,  1.90it/s]
            candidates = rjson[1][0][1]
            cn_tags = rjson[1][0][3]["lc"]
            candidates = [c for i, c in enumerate(candidates) 
                            if list(set(map(int, cn_tags[i].split()))) == [16]]
        self.ime_memory[input_sequence] = candidates  # update
        return candidates

    def __call__(self, pinyins):
        # return similar input sequence with the input string
        if pinyins.__len__() == 1:
            pass
        return 


if __name__ == "__main__":
    ism = InputSequenceManager()
    # print(ism.to_candidates('chend'))
    print(ism.get_input_sequence('陈点'))
    pass
