import os

# PATH (path to custom-files)
REPO_DIR = '/home/chendian/CDConfusor/'
CONFUSOR_DATA_DIR = '/data/chendian/CDConfusor/'

# custom-version (v0.2.0 recommended)
# https://ai.tencent.com/ailab/nlp/en/download.html
# other available versions:
#   d100-v0.2.0-s (vocab=2M, tar_size=0.8G, size=1.8G)
#   d200-v0.2.0-s (vocab=2M, tar_size=1.5G, size=3.6G)
#   d100-v0.2.0 (vocab=12M, tar_size=4.7G, size=12G)
#   d200-v0.2.0 (vocab=12M, tar_size=9.0G, size=22G)
FILE_VERSION = "d200-v0.2.0"  
# origin embeddings directory
TX_PREFIX = "tencent-ailab-embedding-zh-"  
TX_EMBEDDING_PATH = f"{CONFUSOR_DATA_DIR}/{TX_PREFIX}{FILE_VERSION}"  
# pre-processed embeddings directory
TX_DIR = "tx_embeddings"  
EMBEDDING_PATH = f"{CONFUSOR_DATA_DIR}/{TX_DIR}/"

# for generating score matrix
SCORE_DATA_DIR = CONFUSOR_DATA_DIR + 'score_data/'  

# vocabulary files
VOCAB_PATH = f"{REPO_DIR}/data/vocab.txt"
CHAR_PY_VOCAB_PATH = f"{REPO_DIR}/data/vocab_pinyin.txt"


def to_path(*args):
    return os.path.join(*args)


if __name__ == "__main__":
    pass
