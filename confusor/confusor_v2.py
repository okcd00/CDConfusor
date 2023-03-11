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
import sys
sys.path.append("./")
sys.path.append("../")

# PATH (path to custom-files)
CONFUSOR_DATA_DIR = '/data/chendian/CDConfusor/'

# https://ai.tencent.com/ailab/nlp/en/download.html
EMBEDDING_PATH = f"{CONFUSOR_DATA_DIR}/tencent-ailab-embedding-zh-d200-v0.2.0/"

# requirements
from tqdm import tqdm
from pprint import pprint

# utils
from utils import edit_distance
from utils.file_io import get_filesize


