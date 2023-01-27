"""
@Time   :   2021-01-21 11:24:00
@File   :   csc.py
@Author :   Abtion, okcd00
@Email  :   abtion{at}outlook.com
"""
import os
import time
from copy import deepcopy
from glob import glob
from torch.utils.data import Dataset
from bbcm.data.datasets.sqlite_db import SQLiteDB
from bbcm.utils import load_json, dump_json, lower_bound, highlight_positions
from bbcm.utils.text_utils import clean_text


class CscDataset(Dataset):
    def __init__(self, fp):
        self.skip_clean = True
        self.skip_wrong_ids = True
        self.data_postfix = fp.split('.')[-1]

        if self.data_postfix in ['json']:
            self.data = load_json(fp)
        elif self.data_postfix in ['txt']:
            self.data = [line.strip() for line in open(fp, 'r')]
        elif self.data_postfix in ['pkl']:
            import pickle
            self.data = pickle.load(open(fp, 'rb'))
        elif self.data_postfix in ['db']:
            self.data = SQLiteDB(db_path=fp, load_now=True)
        else:
            raise ValueError(f"Invalid fp postfix: {self.data_postfix}")

    def __len__(self):
        return self.data.__len__()

    def generate_wrong_ids(self, ot, ct):
        # 这里采用的是字符粒度
        # 我们在使用BERT模型时通常需要重建为 tokenizer 分词后的粒度
        return [_i for _i, (_o, _c) 
                in enumerate(zip(ot, ct)) if _o != _c]

    def show_item(self, tup):
        if isinstance(tup, int):
            tup = self[tup]
        if isinstance(tup, list):
            if len(tup) < 3:
                tup.append(self.generate_wrong_ids(tup[0], tup[1]))
            highlight_positions(
                text=tup[0], positions=tup[2], color='blue')
            highlight_positions(
                text=tup[1], positions=tup[2], color='blue')
        elif isinstance(tup, dict):
            if 'wrong_ids' not in tup:
                tup['wrong_ids'] = self.generate_wrong_ids(
                    tup['original_text'], tup['correct_text'])
            highlight_positions(
                text=tup['original_text'], positions=tup['wrong_ids'], color='blue')
            highlight_positions(
                text=tup['correct_text'], positions=tup['wrong_ids'], color='blue')

    def __getitem__(self, index):
        wr_ids = None
        if index >= self.data.__len__():
            return None
        if self.data_postfix in ['json', 'pkl']:
            ot = self.data[index]['original_text']
            ct = self.data[index]['correct_text']
            wr_ids = self.data[index].get('wrong_ids')
        elif self.data_postfix in ['txt']:
            t_case = self.data[index].split('\t')
            ot = t_case[0].strip()
            if len(t_case) > 1:
                ct = t_case[1].strip()
            else:
                ct = ot
        elif self.data_postfix in ['db']:
            t_case = deepcopy(self.data[index]).split('\t')
            ot, ct = t_case[0].strip(), t_case[1].strip()
        else:
            raise ValueError(f"Invalid fp postfix: {self.data_postfix}")
        
        if not self.skip_clean:
            ot = clean_text(ot)  
            ct = clean_text(ct)  
        if wr_ids is None and not self.skip_wrong_ids:
            wr_ids = self.generate_wrong_ids(ot, ct)
        return ot, ct, wr_ids


class PureTextDataset(Dataset):
    def __init__(self, fp):
        self.fp = fp
        self.file_list = sorted(glob(f"{fp}/*.txt"))
        self.file_sample_count = []
        self.file_offset = [0]
        self.sample_counts = self.count_samples()
        self.current_file_index = -1
        self.current_file_samples = []
        print(f"Loaded {self.file_list.__len__()} files from {fp}.")

    @staticmethod
    def read_text_file(path):
        return [line.strip() for line in open(path, 'r') if line.strip()]

    def remove_dataset_info(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        if os.path.exists(fp_log_path):
            os.remove(fp_log_path)

    def dump_dataset_info(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        dump_json({
            'file_offset': self.file_offset,
            'file_sample_count': self.file_sample_count,
            'sample_counts': self.sample_counts,
        }, fp_log_path)

    def reset_dataset_info(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        self.remove_dataset_info()
        self.count_samples()

    def count_samples(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        start_time = time.time()
        if os.path.exists(fp_log_path):
            dataset_info = load_json(fp_log_path)
            self.file_offset = list(map(int, dataset_info['file_offset']))
            self.file_sample_count = dataset_info['file_sample_count']
            self.sample_counts = dataset_info['sample_counts']
        else:
            for file_name in self.file_list:
                samples = self.read_text_file(file_name)
                s_len = len(samples)
                self.file_sample_count.append(s_len)
                self.file_offset.append(self.file_offset[-1] + s_len)
            self.sample_counts = sum(self.file_sample_count)
            self.dump_dataset_info()
        print(f"Init indexing ends in {time.time()-start_time} seconds")
        return self.sample_counts

    def load_from_dir(self, dir_path):
        self.__init__(dir_path)

    @staticmethod
    def binary_search_file_index(a, x):
        # the index of the file for x-th sample.
        return lower_bound(a, x + 1) - 1

    def show_item(self, tup):
        if isinstance(tup, int):
            tup = self[tup]
        if isinstance(tup, list):
            highlight_positions(
                text=tup[0], positions=tup[2], color='blue')
            highlight_positions(
                text=tup[1], positions=tup[2], color='blue')
        elif isinstance(tup, dict):
            highlight_positions(
                text=tup['original_text'], positions=tup['wrong_ids'], color='blue')
            highlight_positions(
                text=tup['correct_text'], positions=tup['wrong_ids'], color='blue')

    def __len__(self):
        return self.sample_counts

    def __getitem__(self, index):
        # for a large text corpus, shuffle is not recommended.
        file_index = self.binary_search_file_index(self.file_offset, index)
        if file_index == self.file_list.__len__():
            raise ValueError(f"Invalid index {file_index} with offset {index}")
        if file_index != self.current_file_index:
            file_path = self.file_list[file_index]
            self.current_file_samples = self.read_text_file(file_path)
            self.current_file_index = file_index
        index_in_file = index - self.file_offset[file_index]
        target_text = clean_text(self.current_file_samples[index_in_file])[:500]
        # return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']
        return target_text, target_text, []


if __name__ == "__main__":
    ptd = PureTextDataset("/data/chendian/clean_pretrain_data/")
    print(ptd.sample_counts)
