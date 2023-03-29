# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2023 All rights reserved.
#
#   filename : paths.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-03-18
#   desc     : record paths to files and directories
#              for registering and easy-modification
# ==========================================================================
import os

# PATH (path to custom-files)
REPO_DIR = '/home/chendian/CDConfusor/'
CONFUSOR_DATA_DIR = '/data/chendian/CDConfusor/'
TMP_DIR = f"{CONFUSOR_DATA_DIR}/tmp"


# ---------------------------------------------------------
#          Tencent Embeddings (optional, can be downloaded)
# ---------------------------------------------------------

# custom-version (v0.2.0 recommended)
# https://ai.tencent.com/ailab/nlp/en/download.html
# other available versions:
#   d100-v0.2.0-s (vocab=2M, tar_size=0.8G, size=1.8G)
#   d200-v0.2.0-s (vocab=2M, tar_size=1.5G, size=3.6G)
#   d100-v0.2.0 (vocab=12M, tar_size=4.7G, size=12G)
#   d200-v0.2.0 (vocab=12M, tar_size=9.0G, size=22G)
FILE_VERSION = "d200-v0.2.0"  

# origin embeddings directory (untarred directory)
TX_PREFIX = "tencent-ailab-embedding-zh-"  
TX_EMBEDDING_PATH = f"{CONFUSOR_DATA_DIR}/{TX_PREFIX}{FILE_VERSION}"  

# pre-processed embeddings directory (store intermediate files)
TX_DIR = "tx_embeddings"  
EMBEDDING_PATH = f"{CONFUSOR_DATA_DIR}/{TX_DIR}/"

# ---------------------------------------------------------
#          Word Frequency (optional, collect it yourself)
# ---------------------------------------------------------

# for generating frequency records
# in pickle files, there is a dict as {word: frequency_score} 
FREQ_DIR = f"{CONFUSOR_DATA_DIR}/frequency/" 

# The frequency of words in the general corpus of network texts.
PATH_FREQ_GNR_CHAR = f"{FREQ_DIR}/word_freq.gnr01.pkl"  # 1char & 1word
PATH_FREQ_GNR_WORD = f"{FREQ_DIR}/phrase_freq.gnr2.pkl"  # 2words

# The frequency of words in financial texts.
PATH_FREQ_FIN_CHAR = f"{FREQ_DIR}/word_freq.fin01.pkl"  # 1char & 1word
PATH_FREQ_FIN_WORD = f"{FREQ_DIR}/phrase_freq.fin2.pkl"  # 2words 

# ---------------------------------------------------------
#          Candidate Scoring 
# ---------------------------------------------------------

# for generating score matrix
SCORE_DATA_DIR = f"{CONFUSOR_DATA_DIR}/score_data/"

# vocabulary/mapping files in current repo
VOCAB_PATH = f"{REPO_DIR}/data/vocab.txt"
CHAR_PY_VOCAB_PATH = f"{REPO_DIR}/data/vocab_pinyin.txt"
PY_MAPPING_PATH = f"{REPO_DIR}/data/pinyin_mapping.json"

# pre-processed memory files
IS_MEMORY_PATH = f"{REPO_DIR}/data/is_memory.json"
IME_MEMORY_PATH = f"{REPO_DIR}/data/ime_memory.json"

# generated cache files
CONFUSION_PATH = f"{REPO_DIR}/data/cfs_cache.pkl"
RED_SCORE_PATH = f"{REPO_DIR}/data/red_cache.pkl"


def to_path(*args):
    return os.path.join(*args)


if __name__ == "__main__":
    pass
