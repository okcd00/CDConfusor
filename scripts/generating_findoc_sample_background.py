import sys
sys.path.append('../')

import random
from tqdm import tqdm
from pprint import pprint
from pypinyin import lazy_pinyin

mapping = {}
used_conf = {}

from src.confusor_v2 import Confusor as ConfusorV2
cfs = ConfusorV2()


def word_confusor_cfs(word, random_select=True, ignore_word=None):
    if (word not in used_conf) or len(used_conf.get(word, [])) == 0:
        used_conf[word] = [] 
        res = []
        
        candidates = cfs(word, return_score=True)
        _extra = []
        for idx, item in enumerate(candidates):
            if len(item) == 1:
                print(f"Single-item [{idx}] {item} in candidates:\n")
                pprint(candidates)
                continue
        # print(candidates)
        res.extend(
            [(c, score) for idx, (c, score) in enumerate(candidates) 
             if len(c) == len(word)])

        for w, score in res:
            used_conf[word].extend([w] * int( (1. - score) ** 2 // 0.01 + 1))
    
    if word in used_conf[word]:
        used_conf[word].remove(word)
    if ignore_word is not None:
        if ignore_word in used_conf[word]:
            used_conf[word].remove(ignore_word)

    def random_augment_single_char(_word):
        try:  # sample single char in it.
            char_idx = random.choice(list(range(len(_word))))
            char = _word[char_idx]
            _c = word_confusor_cfs(char, random_select=True)
            ret = ''.join([_c if i == char_idx else c for i, c in enumerate(_word)])
            return ret
        except Exception as e:
            print(e)
            return _word

    if len(used_conf[word]) == 0:
        return random_augment_single_char(word)
    elif random_select:
        ret = random.choice(used_conf[word])
        used_conf[word].remove(ret)
    else:
        ret = used_conf[word][0]
        used_conf[word] = used_conf[word][1:]

    return ret


import Pinyin2Hanzi
from tqdm import tqdm

DATE_STAMP = '230405'
PATH_TO_CORPUS = '/data/chendian/csc_findoc_corpus/unique_text_lines.220803.txt'
lines = [line.strip() for line in open(PATH_TO_CORPUS, 'r')]

generated_n_lines = 0
sampled_candidates = {}
with open(f'../exp/data/fin/findoc_augw.{DATE_STAMP}.tsv', 'w') as f:
    for idx, line in tqdm(enumerate(lines)):
        words = []
        available = []
        line = line.strip()
        for i, wt in enumerate(line.split('\x01')):
            w = wt.split('\x02')[0]
            words.append(w)
            if len(wt.split('\x02')) == 1:
                # if sampled more than 10 times, skip this word
                if not Pinyin2Hanzi.is_chinese(w):
                    continue
                # each word is sampled at most 10 times
                if sampled_candidates.get(w, {}).__len__() < 10:  
                    available.append(i)
        if len(available) == 0:
            continue
        cor = ''.join(words)
        if not 10 < len(cor) < 192:
            continue
        _i = random.choice(available)
        _w = words[_i]
        if len(_w) > 2:
            end_pos = random.randint(2, len(_w))
            piece = _w[end_pos-2:end_pos]
            piece = word_confusor_cfs(piece, random_select=True)
            _c = f"{_w[:end_pos-2]}{piece}{_w[end_pos:]}"
        else:
            _c = word_confusor_cfs(_w, random_select=True)
        sampled_candidates.setdefault(_w, set())
        sampled_candidates[_w].add(_c)
        words[_i] = _c
        err = ''.join(words)
        f.write(f"{err}\t{cor}\n")
        generated_n_lines += 1

        if idx % 10000 == 0:
            with open(f'../exp/data/fin/findoc_augw.{DATE_STAMP}.sampled_candidates.txt', 'w') as f2:
                f2.write(f"Generating Info:  Index-{idx}, TargetWord-{len(sampled_candidates)}, Sample-{generated_n_lines}.")
                for k, v in sampled_candidates.items():
                    f2.write(f"{k}\t{' '.join(v)}\n")
            print(len(sampled_candidates), 'kinds of words are saved.')
