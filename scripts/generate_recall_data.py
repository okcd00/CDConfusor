import sys
sys.path.append("../")

import json
from tqdm import tqdm
from confusor import Confusor

SAVEPATH = '/data/pangchaoxu/tencent_embedding/test_data/'

# initialize
time_test_list = json.load(open(SAVEPATH + 'time_test_list.json', 'r'))
conf = Confusor(cos_threshold=(0.1, 0.5), method='baseline', mode='sort', debug=False)
# generate data
recall_test_data = []
for word in tqdm(time_test_list):
    sorted_res = conf.get_pinyin_sequence(word, cand_pinyin_num=1000)
    recall_test_data.append((word, sorted_res))
json.dump(recall_test_data, open(SAVEPATH + 'recall_test_data_1000.json', 'w'))

