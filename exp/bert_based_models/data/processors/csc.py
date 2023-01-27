"""
@Time   :   2021-01-29 18:27:21
@File   :   csc.py
@Author :   Abtion, okcd00
@Email  :   abtion{at}outlook.com
"""
import gc
import os
import json
import random

import opencc
from lxml import etree
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from bbcm.utils import flatten, dump_json, load_json, get_abs_path


def proc_item(item, convertor):
    """
    处理 sighan 数据集 (SIGHAN13-15)
    Args:
        item:
    Returns:
    """
    root = etree.XML(item)
    passages = dict()
    mistakes = []
    for passage in root.xpath('/ESSAY/TEXT/PASSAGE'):
        passages[passage.get('id')] = convertor.convert(passage.text)
    for mistake in root.xpath('/ESSAY/MISTAKE'):
        mistakes.append({'id': mistake.get('id'),
                         'location': int(mistake.get('location')) - 1,
                         'wrong': convertor.convert(mistake.xpath('./WRONG/text()')[0].strip()),
                         'correction': convertor.convert(mistake.xpath('./CORRECTION/text()')[0].strip())})

    rst_items = dict()

    def get_passages_by_id(pgs, _id):
        p = pgs.get(_id)
        if p:
            return p
        _id = _id[:-1] + str(int(_id[-1]) + 1)
        p = pgs.get(_id)
        if p:
            return p
        raise ValueError(f'passage not found by {_id}')

    for mistake in mistakes:
        if mistake['id'] not in rst_items.keys():
            rst_items[mistake['id']] = {'original_text': get_passages_by_id(passages, mistake['id']),
                                        'wrong_ids': [],
                                        'correct_text': get_passages_by_id(passages, mistake['id'])}

        # todo 繁体转简体字符数量或位置发生改变校验

        ori_text = rst_items[mistake['id']]['original_text']
        cor_text = rst_items[mistake['id']]['correct_text']
        if len(ori_text) == len(cor_text):
            if ori_text[mistake['location']] in mistake['wrong']:
                rst_items[mistake['id']]['wrong_ids'].append(mistake['location'])
                wrong_char_idx = mistake['wrong'].index(ori_text[mistake['location']])
                start = mistake['location'] - wrong_char_idx
                end = start + len(mistake['wrong'])
                rst_items[mistake['id']][
                    'correct_text'] = f'{cor_text[:start]}{mistake["correction"]}{cor_text[end:]}'
        else:
            print(f'{mistake["id"]}\n{ori_text}\n{cor_text}')
    rst = []
    for k in rst_items.keys():
        if len(rst_items[k]['correct_text']) == len(rst_items[k]['original_text']):
            rst.append({'id': k, **rst_items[k]})
        else:
            text = rst_items[k]['correct_text']
            rst.append({'id': k, 'correct_text': text, 'original_text': text, 'wrong_ids': []})
    return rst


def proc_confusion_item(item, id_prefix="", id_postfix=""):
    """
    处理 confusion set 数据集 (AutoCorpusGeneration)
    Args:
        item:
    Returns:
    """
    root = etree.XML(item)
    text = root.xpath('/SENTENCE/TEXT/text()')[0]
    mistakes = []
    for mistake in root.xpath('/SENTENCE/MISTAKE'):
        mistakes.append({'location': int(mistake.xpath('./LOCATION/text()')[0]) - 1,
                         'wrong': mistake.xpath('./WRONG/text()')[0].strip(),
                         'correction': mistake.xpath('./CORRECTION/text()')[0].strip()})

    cor_text = text
    wrong_ids = []

    for mis in mistakes:
        cor_text = f'{cor_text[:mis["location"]]}{mis["correction"]}{cor_text[mis["location"] + 1:]}'
        wrong_ids.append(mis['location'])

    rst = [{
        'id': f'{id_prefix}-{id_postfix}',
        'original_text': text,
        'wrong_ids': wrong_ids,
        'correct_text': cor_text
    }]
    if len(text) != len(cor_text):
        return [{'id': f'{id_prefix}--{id_postfix}',
                 'original_text': cor_text,
                 'wrong_ids': [],
                 'correct_text': cor_text}]
    # 取一定概率保留原文本
    if random.random() < 0.01:
        rst.append({'id': f'{id_prefix}--{id_postfix}',
                    'original_text': cor_text,
                    'wrong_ids': [],
                    'correct_text': cor_text})
    return rst


def proc_test_set(fp, convertor):
    """
    采用 SIGHAN15_CSC_Test 以生成测试集
    Args:
        fp:
        convertor:
    Returns:
    """
    inputs = dict()
    with open(os.path.join(fp, 'SIGHAN15_CSC_TestInput.txt'), 'r', encoding='utf8') as f:
        for line in f:
            pid = line[5:14]
            text = line[16:].strip()
            inputs[pid] = text

    rst = []
    with open(os.path.join(fp, 'SIGHAN15_CSC_TestTruth.txt'), 'r', encoding='utf8') as f:
        for line in f:
            pid = line[0:9]
            mistakes = line[11:].strip().split(', ')
            if len(mistakes) <= 1:
                text = convertor.convert(inputs[pid])
                rst.append({'id': pid,
                            'original_text': text,
                            'wrong_ids': [],
                            'correct_text': text})
            else:
                wrong_ids = []
                original_text = inputs[pid]
                cor_text = inputs[pid]
                for i in range(len(mistakes) // 2):
                    idx = int(mistakes[2 * i]) - 1
                    cor_char = mistakes[2 * i + 1]
                    wrong_ids.append(idx)
                    cor_text = f'{cor_text[:idx]}{cor_char}{cor_text[idx + 1:]}'
                original_text = convertor.convert(original_text)
                cor_text = convertor.convert(cor_text)
                if len(original_text) != len(cor_text):
                    print(pid)
                    print(original_text)
                    print(cor_text)
                    continue
                rst.append({'id': pid,
                            'original_text': original_text,
                            'wrong_ids': wrong_ids,
                            'correct_text': cor_text})
    return rst


def read_data(fp):
    # read corpus from SIGHAN13-15 (*ing.sgml)
    for fn in os.listdir(fp):
        if fn.endswith('ing.sgml'):
            with open(os.path.join(fp, fn), 'r', encoding='utf-8', errors='ignore') as f:
                item = []
                for line in f:
                    if line.strip().startswith('<ESSAY') and len(item) > 0:
                        yield ''.join(item)
                        item = [line.strip()]
                    elif line.strip().startswith('<'):
                        item.append(line.strip())


def read_confusion_data(fp):
    # read AutoCorpusGeneration corpus released from the EMNLP2018 paper
    fn = os.path.join(fp, 'train.sgml')
    with open(fn, 'r', encoding='utf8') as f:
        item = []
        for line in tqdm(f):
            if line.strip().startswith('<SENT') and len(item) > 0:
                yield ''.join(item)
                item = [line.strip()]
            elif line.strip().startswith('<'):
                item.append(line.strip())


def read_cd_data(fp, file_postfix='_cd.json', ignore_files=list()):
    # other pre-processed csc datasets
    # a list of items with keys ['id', 'original_text', 'wrong_ids', 'correct_text']
    for fn in os.listdir(fp):
        ign_flag = False
        for ign_fn in ignore_files:
            if fn in ign_fn:
                ign_flag = True
        if ign_flag:
            continue
        if fn.endswith(file_postfix):
            with open(os.path.join(fp, fn), 'r',
                      encoding='utf-8', errors='ignore') as f:
                samples = json.load(f)
                yield samples


def parse_cged_file(file_dir):
    from xml.dom import minidom
    # from shibing624/pycorrector
    rst = []
    for fn in os.listdir(file_dir):
        if fn.endswith('.xml'):
            path = os.path.join(file_dir, fn)
            print('Parse data from %s' % path)

            dom_tree = minidom.parse(path)
            docs = dom_tree.documentElement.getElementsByTagName('DOC')
            for doc in docs:
                id = ''
                text = ''
                texts = doc.getElementsByTagName('TEXT')
                for i in texts:
                    id = i.getAttribute('id')
                    # Input the text
                    text = i.childNodes[0].data.strip()
                # Input the correct text
                correction = doc.getElementsByTagName('CORRECTION')[0]. \
                    childNodes[0].data.strip()
                wrong_ids = []
                for error in doc.getElementsByTagName('ERROR'):
                    start_off = error.getAttribute('start_off')
                    end_off = error.getAttribute('end_off')
                    if start_off and end_off:
                        for i in range(int(start_off), int(end_off)+1):
                            wrong_ids.append(i)
                source = text.strip()
                target = correction.strip()

                pair = [source, target]
                if pair not in rst:
                    rst.append({'id': id,
                                'original_text': source,
                                'wrong_ids': wrong_ids,
                                'correct_text': target
                                })
    dump_json(rst, get_abs_path('datasets', 'csc', 'cged.json'))
    return rst


def preproc():
    rst_items = []
    convertor = opencc.OpenCC('tw2sp.json')


    # generate samples from SIGHAN15-Test as test-data.
    test_items = proc_test_set(get_abs_path('datasets', 'csc'), convertor)

    # generate samples from "*ing.sgml" files (SIGHAN sgml files)
    sighan_samples = [proc_item(item, convertor)
                      for item in read_data(get_abs_path('datasets', 'csc'))]
    rst_items += flatten(sighan_samples)

    # generate samples from "*train.sgml" files (the ACG sgml file)
    confusion_samples = [proc_confusion_item(item, id_prefix='cf', id_postfix=str(_i))
                         for _i, item in enumerate(read_confusion_data(get_abs_path('datasets', 'csc')))]
    rst_items += flatten(confusion_samples)

    # extend samples from "*cd.json" files (custom csc_data json files)
    for custom_data in read_cd_data(get_abs_path('datasets', 'csc')):
        rst_items += custom_data

    # print("sighan samples count:", len(sighan_samples))
    # print("confusion samples count:", len(confusion_samples))

    # 拆分训练集与测试集
    dev_set_len = len(rst_items) // 10
    print(len(rst_items))

    random.seed(666)
    random.shuffle(rst_items)
    dump_json(rst_items[:dev_set_len], get_abs_path('datasets', 'csc', 'dev.json'))
    dump_json(rst_items[dev_set_len:], get_abs_path('datasets', 'csc', 'train.json'))
    dump_json(test_items, get_abs_path('datasets', 'csc', 'test.json'))
    gc.collect()


def preproc_cd():
    rst_items = []
    dir_path = get_abs_path('datasets', 'csc')

    # generate test samples from ys_data or SIGHAN-15Test.
    test_file_path = get_abs_path(dir_path, '15test_cd.json')  # '15test_cd.json'
    test_items = load_json(test_file_path)

    # generate samples from AutoCorpusGeneration dataset (train.sgml).
    confusion_samples = [proc_confusion_item(item, id_prefix='cf', id_postfix=str(_i))
                         for _i, item in enumerate(read_confusion_data(dir_path))]
    rst_items += flatten(confusion_samples)

    # generate samples from pre-processed samples (*_cd.json).
    ignore_files = ['15test_cd.json']
    for custom_samples in read_cd_data(dir_path, ignore_files=ignore_files):
        rst_items += custom_samples

    # Split into train dataset and valid dataset.
    n_dev = len(rst_items) // 10
    n_trn = len(rst_items) - n_dev
    train_items, dev_items = random_split(rst_items, [n_trn, n_dev],  # absolute size of samples
                                          generator=torch.Generator().manual_seed(666))
    print(f"Dump {n_trn} samples as train and {n_dev} samples as dev.")
    dump_json(list(test_items), get_abs_path('datasets', 'csc', 'test.json'))
    dump_json(list(dev_items), get_abs_path('datasets', 'csc', 'dev.json'))
    dump_json(list(train_items), get_abs_path('datasets', 'csc', 'train.json'))
    gc.collect()
