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
from pprint import pprint


# PATH (path to custom-files)
from paths import (
    SCORE_DATA_DIR, TMP_DIR,
    PY_MAPPING_PATH, 
    IS_MEMORY_PATH, IME_MEMORY_PATH,
    VOCAB_PATH, CHAR_PY_VOCAB_PATH,)

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
        self.vocab = load_vocab(VOCAB_PATH)
        self.vocab_set = set(self.vocab)  # for O(1) finding
        self.char_vocab = load_vocab(CHAR_PY_VOCAB_PATH)
        self.py_mapping = load_json(f"{PY_MAPPING_PATH}")

        self.is_memory = {}  # pinyin_str -> similar [input_sequence, ...] 
        self.is_save_flag = False
        self.ime_memory = {}  # input_sequence -> [(word, rank-chain), ...]
        self.ime_save_flag = False
        self.ime_update_count = 0
        self.update_to_save = 100
        self.ime_candidate_count = 20
        self.ime_max_selection_count = 3
        self.init_is_memory()  
        self.init_ime_memory()

        self.ph_util = Py2HzUtils()
        self.is_chinese = Pinyin2Hanzi.is_chinese

    def load_is_memory(self, memory_path=None):
        if os.path.exists(memory_path):
            if IS_MEMORY_PATH.endswith('.json'):
                dic = load_json(memory_path, show_time=True)
            elif IS_MEMORY_PATH.endswith('.kari'): 
                dic = load_kari(memory_path, show_time=True)
            for k, v in dic.items():
                self.is_memory[k] = sorted(set(v + self.is_memory.get(k, [])))
            print(f"Loaded {len(dic.items())} items from {memory_path}")

    def init_is_memory(self):
        # load recorded google IME memory
        if os.path.exists(IS_MEMORY_PATH):  
            self.load_is_memory(IS_MEMORY_PATH)
        
        # mkdir for tmp files
        os.system(f"mkdir -p {TMP_DIR}")

    def load_ime_memory(self, ime_memory_path=None):
        dic = load_json(ime_memory_path, show_time=True)
        self.ime_memory.update(dic)
        print(f"Loaded {len(dic.items())} items from {ime_memory_path}")

    def filter_ime_memory(self):
        dic = {k: [(cand, rank_str) for cand, rank_str in v 
                   if len(rank_str.split('-')) <= self.ime_max_selection_count] 
               for k, v in self.ime_memory.items()}
        self.ime_memory = dic
        self.ime_save_flag = True
        self.ime_update_count += 1
        self.save_memory()

    def init_ime_memory(self):
        # load recorded google IME memory
        if os.path.exists(IME_MEMORY_PATH):  # ime_memory.google.json
            self.load_ime_memory(IME_MEMORY_PATH)
        # online google IME initialization.
        from httpx_socks import SyncProxyTransport, AsyncProxyTransport
        self.transport = SyncProxyTransport.from_url(self.proxies)

    def save_memory(self, force=False):
        # print(self.save_flag, self.ime_save_flag)
        fp = '[not-saved]'
        if self.is_save_flag or force:
            fp = IS_MEMORY_PATH.split('/')[-1]
            if fp.endswith('.json'):
                dump_json(self.is_memory, f"{TMP_DIR}/{fp}")
            elif fp.endswith('.kari'):
                save_kari(self.is_memory, f"{TMP_DIR}/{fp}")
            else:
                raise ValueError(f"Unknown file format: {fp}")
            print(f"Saved IS memory in {TMP_DIR}/{fp}.", time.ctime())
            self.is_save_flag = False
        
        if self.ime_save_flag or force:
            fp = IME_MEMORY_PATH.split('/')[-1]
            if fp.endswith('.json'):
                with open(f"{TMP_DIR}/{fp}", 'w') as f:
                    f.write('{\n')
                    f.write(',\n'.join([
                        f' "{inp}": {json.dumps(v, ensure_ascii=False)}' 
                        for inp, v in sorted(
                            self.ime_memory.items(), 
                            key=lambda x: x[0])]))
                    f.write('\n}\n')
            elif fp.endswith('.kari'):
                save_kari(self.ime_memory, f"{TMP_DIR}/{fp}")
            else:
                raise ValueError(f"Unknown file format: {fp}")
            print(f"Saved IME memory in {TMP_DIR}/{fp}.", time.ctime())
            self.ime_save_flag = False
            self.ime_update_count = 0
        
    def update_memory_from_tmp(self):
        fp = IS_MEMORY_PATH.split('/')[-1]
        if os.path.exists(f"{TMP_DIR}/{fp}"):
            if os.path.exists(IS_MEMORY_PATH):
                os.system(f"mv {IS_MEMORY_PATH} {IS_MEMORY_PATH}.bak")
            os.system(f"cp {TMP_DIR}/{fp} {IS_MEMORY_PATH}")
        fp = IME_MEMORY_PATH.split('/')[-1]
        if os.path.exists(f"{TMP_DIR}/{fp}"):
            if os.path.exists(IME_MEMORY_PATH):
                os.system(f"mv {IME_MEMORY_PATH} {IME_MEMORY_PATH}.bak")
            os.system(f"cp {TMP_DIR}/{fp} {IME_MEMORY_PATH}")

    def py2phrase(self, pinyins):
        """
        pinyins: list of complete pinyins
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
        # not that the original pinyin is not in the returned list
        if isinstance(pinyin[0], list):
            pinyin = [p[0] for p in pinyin]
        input_sequences = []
        for simp_idx, _ in enumerate(pinyin):
            input_sequences.append(''.join(
                [py[0] if idx >= simp_idx else py 
                 for idx, py in enumerate(pinyin)]))
        return input_sequences

    def get_similar_input_sequences(self, pinyin):
        # get_similar_input_sequences
        if isinstance(pinyin[0], list):
            pinyin = [p[0] for p in pinyin]
        if pinyin.__len__() == 1:
            if pinyin[0] in self.py_mapping:
                return self.py_mapping[pinyin[0]]
            else:
                print("Invalid pinyin", pinyin)
                return pinyin
        input_sequences = set()
        input_sequences.add(''.join(pinyin))
        # errors in single Chinese character
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
            if pinyin_str in self.is_memory:
                input_sequences = self.is_memory[pinyin_str]
            else:
                input_sequences = self.get_similar_input_sequences(pinyin)
                self.is_memory[pinyin_str] = input_sequences
                self.is_save_flag = True
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

    def _analysis_from_ime_json(self, input_sequence, rjson, 
                                method='client', heuristic=True, ngram=None):
        _ret = []
        i_size = input_sequence.__len__()
        candidates, info = rjson[1][0][1], rjson[1][0][3]
        matched_length = info.get("matched_length")

        for i, c in enumerate(candidates[:self.ime_candidate_count]):
            t = map(int, info["lc"][i].split())
            if list(set(t)) != [16]:  
                continue  # not Chinese phrase
            if not heuristic:
                if i_size == len(c):
                    _ret.append((c, f'{i}'))
                continue
            if i_size == 1:
                _ret.append((c, f'{i}'))
                continue
            if (matched_length is None) or matched_length[i] == i_size:
                _ret.append((c, f'{i}'))
                continue
            _rest = input_sequence[matched_length[i]:]

            candidates_with_rank = self._from_online_ime(
                _rest, method=method, heuristic=heuristic,
                ngram=(ngram-1 if ngram is not None else ngram))
            
            for cand, rank_str in candidates_with_rank:
                if len(rank_str.split('-')) >= self.ime_max_selection_count:
                    continue  # not recording too-long-selections such as 1-14-5-2
                _ret.append((c + cand, f"{i}-{rank_str}"))
        
        return _ret

    def _from_online_ime(self, input_sequence, method='client', 
                         heuristic=True, ngram=None):
        # if ngram is None, return all length of word candidates.
        if ngram == 0:
            return []
        
        _ret = []
        if input_sequence in self.ime_memory:
            # a list of (cand_str, rank_str) items
            _ret = self.ime_memory[input_sequence] 
        else:
            # generate url for online IME
            url = self.URL_GOOGLE_IME_API.format(input_sequence)
            if method in ['request']:
                rjson = self.request_google_ime(url)
            elif method in ['client']:
                rjson = self.client_post_google_ime(url)
            # network error
            if rjson[0] != 'SUCCESS':  # 424it [03:43,  1.90it/s]
                print("Error: ", rjson)
                candidates = _ret
                return candidates
            _ret = self._analysis_from_ime_json(
                input_sequence, rjson, 
                method=method, heuristic=heuristic)
        
            # update memory
            self.ime_memory[input_sequence] = _ret
            self.ime_save_flag = True
            self.ime_update_count += 1
            if self.ime_update_count > self.update_to_save:
                self.save_memory()

        # output candidates with n-gram
        if ngram is not None:
            _ret = [(cand_str, rank_str) for cand_str, rank_str in _ret 
                    if cand_str.__len__() == ngram \
                    and all([c in self.vocab_set for c in cand_str])]
        return _ret

    def to_candidates(self, input_sequence, ngram=None):
        if input_sequence in self.ime_memory:
            return self.ime_memory[input_sequence]
        # self.py2word() only takes complete pinyin lists.
        print(f"Warning: [{input_sequence}] not in memory, ")
        candidates = self._from_online_ime(input_sequence, ngram=ngram)
        return candidates

    def __call__(self, word=None, pinyin=None):
        # return similar input sequence with the input string
        pinyin, input_sequences = self.get_input_sequence(
            word=word, pinyin=pinyin, 
            simp_candidates=True)
        # self.save_memory()
        return pinyin, input_sequences


if __name__ == "__main__":
    ism = InputSequenceManager()
    ret = ism._from_online_ime('chendian', ngram=2)
    print(ret)
    # ism.save_memory()
    # ism.filter_ime_memory()
    ism.update_memory_from_tmp()
