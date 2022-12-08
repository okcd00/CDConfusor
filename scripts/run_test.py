import sys
sys.path.append("../")

import time
from data.sqlite_db import SQLiteDB
start_time = time.time()
test_db_path = './tmp/confusionset_sighan.221110.db'

dataset = SQLiteDB(
    test_db_path, 
    load_now=False)

dataset.write({'[UNK]': ['[UNK]']})

print("Init SQLite Ends.", time.time() - start_time)
print("The first sample is:", dataset[0], dataset['[UNK]'])


from confusor.confusor import default_confusor
cfs = default_confusor()
cfs.debug = False
cfs.keep_num = 1000

from tqdm import tqdm

# check original-word cover rate.
from utils.file_io import load_json
from utils.csc_utils import get_faulty_pair
from utils.text_utils import clean_text, is_chinese_char, is_pure_chinese_phrase

src_path = "./tmp/sighan15_train.json"
train_pairs = [get_faulty_pair(line) 
               for line in load_json("./tmp/sighan_train.json") + load_json("./tmp/sighan_dev.json")]

from utils.file_io import load_json, dump_json
rec = load_json('./tmp/word_confusion.221111.json')

for ori_word, cor_word in tqdm(train_pairs):
    if is_pure_chinese_phrase(ori_word) and len(ori_word) <= 4:
        if ori_word not in rec:
            rec[ori_word] = cfs(ori_word)
    if is_pure_chinese_phrase(cor_word) and len(cor_word) <= 4:
        if cor_word not in rec:
            rec[cor_word] = cfs(cor_word)

dump_json(rec, './tmp/word_confusion.221111.json')
dataset.write(rec)