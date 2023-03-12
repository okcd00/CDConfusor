# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2023 All rights reserved.
#
#   filename : preprocess_tx_embeddings.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-02-03
#   desc     : generate intermediate files
#              for boosting confusor
# ==========================================================================
import os
import re
import opencc
import pickle
import pypinyin
import numpy as np
from tqdm import tqdm
from paths import (
    REPO_DIR, CONFUSOR_DATA_DIR,
    TX_PREFIX, FILE_VERSION, TX_DIR,  # for untar.
    EMBEDDING_PATH  # for embedding bucketing.
)


# other resources
CONFUSOR_KEYBOARD_DATA = [
    '            ',
    ' qwertyuiop ',
    ' asdfghjkl  ',
    '  zxcvbnm   ',
    '            ',
]


def untar():
    # the tar.gz file should be put in CONFUSOR_DATA_DIR
    ret = os.system(
        f"cd {CONFUSOR_DATA_DIR} ;" + \
        f"tar -zxvf {CONFUSOR_DATA_DIR}/{TX_PREFIX}{FILE_VERSION}.tar.gz -C ./ ;" + \
        f"mv {CONFUSOR_DATA_DIR}/{TX_PREFIX}{FILE_VERSION} ./{TX_DIR} ;" + \
        f"mv ./{TX_DIR}/{TX_PREFIX}{FILE_VERSION}.txt ./{TX_DIR}/origin.txt"
    )
    return ret


def get_pinyin(word):
    return pypinyin.lazy_pinyin(word)


def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return ch
    return False


def bucket_by_length(max_length=7):
    """
    12287937it [04:45, 43098.73it/s]
    Now saving 4894 1-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//1gram.pkl
    Now saving 7678 2-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//2gram.pkl
    Now saving 62637 3-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//3gram.pkl
    Now saving 132083 4-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//4gram.pkl
    Now saving 171060 5-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//5gram.pkl
    Now saving 126791 6-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//6gram.pkl
    Now saving 93209 7-gram embeddings in /data/chendian/CDConfusor//tx_embeddings//7gram.pkl
    """
    fp = open(f'{REPO_DIR}/data/vocab.txt', 'r', encoding='utf-8')
    vocab = [line.strip().lstrip('#') for line in fp 
             if not line.startswith('[unused')]
    vocab_set = set([char for char in vocab if len(char)==1 and check_contain_chinese(char)])
    char_vocab = [c for c in vocab if c in vocab_set]
    convertor = opencc.OpenCC('tw2sp.json')

    tw_vocab = []
    oov_vocab = []
    word_vocab = []
    ascii_vocab = []
    long_gram_vocab = []
    word_embedding = {i: dict() for i in range(1, max_length+1)}
    for idx, line in tqdm(enumerate(open(f'{EMBEDDING_PATH}/origin.txt', 'r'))):
        items = line.split()
        if idx == 0:
            print("{} lines, {}-dim embeddings.".format(*items))
            continue
        word = items[0].strip() 
        if re.match('^[a-zA-Z0-9\.\-]+$', word):
            ascii_vocab.append(word)
            continue  # pure alphabet or pure number
        if convertor.convert(word) != word:
            tw_vocab.append(word)
            continue  # traditional words
        if len(word) > 1 and len([c for c in word if c in vocab_set]) == 0:
            oov_vocab.append(word)
            continue  # out-of-BERTvocab
        if len(word) > max_length:
            long_gram_vocab.append(word)
            continue  # too-long-gram
        try:
            vec = list(map(float, items[1:]))
        except Exception as e:
            # print(f"{e}\n{items}")
            continue
        word_vocab.append(word)
        word_embedding[len(word)][word] = np.array(vec, dtype=np.float16)
    for tag, vocab in [("tw", tw_vocab),
                       ("oov", oov_vocab), 
                       ("char", char_vocab),
                       ('word', word_vocab),
                       ('ascii', ascii_vocab),
                       ('long_gram', long_gram_vocab)]:
        with open(f"{EMBEDDING_PATH}/{tag}_vocab.txt", 'w') as f:
            for word in vocab:
                f.write(f"{word}\n")
    for i in range(1, max_length+1):
        dump_file = f"{EMBEDDING_PATH}/{i}gram.pkl"
        print(f"Now saving {len(word_embedding[i])} {i}-gram embeddings in {dump_file}")
        pickle.dump(word_embedding[i], open(dump_file, 'wb'))


if __name__ == "__main__":
    # untar()
    bucket_by_length()
    pass