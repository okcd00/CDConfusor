import sys
sys.path.append('../')

import os
import time
import random
from tqdm import tqdm
from pprint import pprint
from pypinyin import lazy_pinyin
from collections import defaultdict
from src.confusor_v2 import Confusor as ConfusorV2

cfs = ConfusorV2()

char_cfs = {line.strip().split('\t')[0].strip() : line.strip().split('\t')[1].strip() + line.strip().split('\t')[2].strip()
            for line in open('../data/same_pinyin.txt') if line.strip() and line[0] != '#' and len(line.split('\t'))==3}
print(len(char_cfs))

spellgcn_cfs = defaultdict(list)
for line in open('../data/spellGraphs_spellgcn.txt'):
    if not line.strip():
        continue
    l, r, tag = line.strip().split('|')
    if tag[1] == '音' and tag[-1] == '调':
        spellgcn_cfs[l].append(r)
print(len(spellgcn_cfs))

# custom size for confusion set
ROUND = 3
TAKE_CHAR_RATE = 0.11  # still confuse on char-level in a ratio
# CONFUSION_SIZE = 10  # each word take at most K kinds of candidates
WORD_SAMPLE_TIMES = 50  # each word take at most m times for generating samples

used_conf = {}  # word: a list of tuple(candidate, int(weight))
generated_n_lines = 0


sampled_words = {}  # word: int(times)
sampled_candidates = {}  # word: set(candidates)
path_to_sampled_history = None  # '../exp/data/fin/findoc_augw.230406.sampled_candidates.txt'
if path_to_sampled_history and os.path.exists(path_to_sampled_history):
    for idx, line in enumerate(open(path_to_sampled_history, 'r')):
        if idx == 0:
            continue
        items = line.split('\t')
        key = items[0]
        candidates = items[1]
        counts = None
        if len(items) > 2:
            counts = items[2]
        if counts:
            for _c in counts.split(' '):
                sampled_words[key] = int(_c)
        sampled_candidates.setdefault(key, set())
        for _c in candidates.split(' '):
            sampled_candidates[key].add(_c)


def scoring_with_distance(distance_score, min_weight=1):
    version_01 = (1. - distance_score) ** 2 // 0.01 + 1
    return max(min_weight, version_01)


def update_used_conf(word):
    used_conf[word] = [] 
    candidates = cfs(word, return_score=True)
    # deprecated checking
    """
    for idx, item in enumerate(candidates):
        if len(item) == 1:
            print(f"Single-item [{idx}] {item} in candidates:\n")
            pprint(candidates)
            continue
    """
    # print(candidates)    
    res = [  # different candidate words with the same length
        (c, score) for idx, (c, score) in enumerate(candidates) 
        if len(c) == len(word) and c != word \
        and all([_c in cfs.vocab_set for _c in c])]
    # update used_conf dict
    used_conf[word].extend(
        [(w, scoring_with_distance(score)) for w, score in res
         if w != word])

def char_confusor_cfs(word, random_select=True, _cfs=char_cfs, piece_size=1):
    assert len(word) >= piece_size
    _w = f"{word}"
    candidates = [(end_pos, _w[end_pos-piece_size:end_pos]) 
                  for end_pos in range(piece_size, len(_w)+1)
                  if sampled_words.get(_w[end_pos-piece_size:end_pos], 0) < WORD_SAMPLE_TIMES // piece_size]
    if len(candidates) == 0:
        return word
    end_pos, piece = random.choice(candidates)
    # piece = _w[end_pos-piece_size:end_pos]
    piece_candidate = random.choice(_cfs.get(piece, [piece]))
    ret = f"{_w[:end_pos-piece_size]}{piece_candidate}{_w[end_pos:]}"
    return ret
    

def word_confusor_cfs(word, random_select=True, ignore_word=None):
    if (word not in used_conf) or len(used_conf.get(word, [])) == 0:
        update_used_conf(word)
    if len(used_conf[word]) == 0:
        if word[-1] not in ['额']:
            print(f"Word with empty confusion: {word}")
        ret = word
    elif random_select:
        candidates, weights = list(zip(*used_conf[word]))
        if len(sampled_candidates.get(word, [])) >= len(candidates):
            ret = random.choices(candidates, weights=weights, k=1)[0]
        else:
            keep_index = [i for i, c in enumerate(candidates) 
                          if c not in sampled_candidates.get(word, [])]
            candidates = [c for i, c in enumerate(candidates)
                          if i in keep_index]
            weights = [w for i, w in enumerate(weights)
                       if i in keep_index]
            ret = random.choices(candidates, weights=weights, k=1)[0]
    else:
        candidates, weights = list(zip(*used_conf[word]))
        if len(sampled_candidates.get(word, [])) >= len(candidates):
            ret = candidates[0]
        else:
            keep_index = [i for i, c in enumerate(candidates) 
                          if c not in sampled_candidates.get(word, [])]
            candidates = [c for i, c in enumerate(candidates)
                          if i in keep_index]
            ret = candidates[0]
    sampled_candidates.setdefault(word, set())
    sampled_candidates[word].add(ret)
    sampled_words.setdefault(word, 0)
    sampled_words[word] += 1
    return ret


def piece_confusor_cfs(word, random_select=True, piece_size=2):
    assert len(word) >= piece_size
    _w = f"{word}"
    candidates = [(end_pos, _w[end_pos-piece_size:end_pos]) 
                  for end_pos in range(piece_size, len(_w)+1)
                  if sampled_words.get(_w[end_pos-piece_size:end_pos], 0) < WORD_SAMPLE_TIMES // piece_size]
    if len(candidates) == 0:
        return word
    end_pos, piece = random.choice(candidates)
    # piece = _w[end_pos-piece_size:end_pos]
    piece_candidate = word_confusor_cfs(
        piece, random_select=random_select)
    ret = f"{_w[:end_pos-piece_size]}{piece_candidate}{_w[end_pos:]}"
    return ret


import Pinyin2Hanzi
from tqdm import tqdm

DATE_STAMP = 'W271k.ours'
PATH_TO_CORPUS = '../exp/data/cn/Wang271k/dcn_train.tsv'
lines = [line.strip().split('\t')[1].strip() 
         for line in open(PATH_TO_CORPUS, 'r') 
         if line.strip() and len(line.split('\t')) > 1 and line.strip().split('\t')[1].strip()]
print(len(lines))


import jieba
import jieba.posseg as pseg


def isname(single_word_string):
    # print(single_word_string)
    pair_word_list = pseg.lcut(single_word_string)
    for eve_word, cixing in pair_word_list:
        if cixing in ["nr", "ns", "nt", "nw", "nz", "t", "n", "PER"]:
            return True
    return False


generated_err = set()
with open(f'../exp/data/fin/findoc_augw.{DATE_STAMP}.tsv', 'w') as f:
    for _ in tqdm(range(ROUND)):
        for idx, line in tqdm(enumerate(lines)):
            line = line.strip()
            words = jieba.lcut(line)
            available = []
            for i, w in enumerate(words):
                # if sampled more than 10 times, skip this word
                if not Pinyin2Hanzi.is_chinese(w):
                    continue
                if isname(w):  # skip names
                    continue
                # each word is sampled at most K/len(w) times
                # if sampled_candidates.get(w, set()).__len__() < CONFUSION_SIZE:  
                if sampled_words.get(w, 0) < WORD_SAMPLE_TIMES // len(w):
                    available.append(i)
            
            if len(available) == 0:
                continue
            cor = ''.join(words)
            if not 8 < len(cor) < 192:
                continue
            _i = random.choice(available)
            _w = words[_i]
            if False:  # True:
                _c = char_confusor_cfs(_w, _cfs=char_cfs) 
            elif len(_w) > 1 and random.random() < TAKE_CHAR_RATE:
                _c = piece_confusor_cfs(_w, piece_size=1)
            elif len(_w) > 2:  # allow at most 2 char-level in one word.
                _c = piece_confusor_cfs(_w, piece_size=2)
            elif len(_w) <= 2:  # len(_w) <= 2
                _c = word_confusor_cfs(_w, random_select=True)
            else:
                _c = piece_confusor_cfs(_w, piece_size=1)
            words[_i] = _c
            err = ''.join(words)
            if err in generated_err:
                continue
            f.write(f"{err}\t{cor}\n")
            generated_n_lines += 1
            generated_err.add(err)

            if idx % 50000 == 0:
                # save the entire record per 50k steps
                with open(f'../exp/data/fin/findoc_augw.{DATE_STAMP}.sampled_candidates.txt', 'w') as f2:
                    f2.write(f"Generating Info:  Index-{idx},"
                            f"TargetWord-{len(sampled_candidates)},"
                            f"Sample-{generated_n_lines}.\n")
                    for k, v in sampled_candidates.items():
                        c = str(sampled_words.get(k, 0))
                        f2.write(f"{k}\t{' '.join(v)}\t{c}\n")
                    print(len(sampled_candidates), 'kinds of words are saved.')


print(f"{generated_n_lines} lines are generated in this loop.")

# finished
with open(f'../exp/data/fin/findoc_augw.{DATE_STAMP}.sampled_candidates.txt', 'w') as f2:
    f2.write(f"Generating Info:  Index-{idx},"
                f"TargetWord-{len(sampled_candidates)},"
                f"Sample-{generated_n_lines}.\n")
    for k, v in sorted(sampled_candidates.items()):  # sort them at the last time.
        c = str(sampled_words.get(k, 0))
        f2.write(f"{k}\t{' '.join(v)}\t{c}\n")
    print(len(sampled_candidates), 'kinds of words are saved.')

cfs.save_memory()