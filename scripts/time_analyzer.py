import sys
sys.path.append("../")

from src import Confusor
from tqdm import tqdm
import json

# CONFUSOR_DATA_DIR = '/data/chendian/'
CONFUSOR_DATA_DIR = '/data/pangchaoxu/'
DATA_PATH = f'{CONFUSOR_DATA_DIR}/tencent_embedding/time_test_list.json'
LOG_SAVE_PATH = '/home/pangchaoxu/confusionset/logs/time_analysis/'
test_list = json.load(open(DATA_PATH, 'r'))

"""
CONFIGURATIONS
config confusor arguments (only for the params that result in time changes)
"""
log_name = 'baseline'
exper_info = "baseline with the new embedding storing and loading strategy. test num: {}".format(len(test_list))
args = {
    'method': 'baseline',
    'cand_fzimu_num': 100,
    'cand_pinyin_num': 10,
    'cand_zi_num': 50,
    'keep_num': 500,
    'cos_threshold': (0.1, 0.5),
    'dcc_threshold': 2,
    'mode': 'sort',
    'debug': True
}

# initialize the confusor
conf = Confusor(method=args['method'], cand_fzimu_num=args['cand_fzimu_num'], cand_pinyin_num=args['cand_pinyin_num'],
                cand_zi_num=args['cand_zi_num'], keep_num=args['keep_num'],
                dcc_threshold=args['dcc_threshold'], cos_threshold=args['cos_threshold'], mode=args['mode'], debug=args['debug'])

# time analysis
print('start time analysis.')
# all_record = {info: {token_len: [time_point]}}
all_record = {}
for token in tqdm(test_list):
    timer = conf(token)
    start_time = timer[0][1]
    for info, tpoint in timer[1:]:
        time = tpoint - start_time
        all_record.setdefault(info, {})
        all_record[info].setdefault(len(token), [])
        all_record[info][len(token)].append(time)

average_time = {}
for info, records in all_record.items():
    average_time.setdefault(info, {})
    all_info_rec = []
    for tok_len, len_records in records.items():
        average_time[info][tok_len] = sum(len_records) / len(len_records)
        all_info_rec.extend(len_records)
    average_time[info]['total'] = sum(all_info_rec) / len(all_info_rec)

log = dict(info=exper_info, args=args, average_time=average_time)
json.dump(log, open(LOG_SAVE_PATH + log_name + '.json', 'w'))