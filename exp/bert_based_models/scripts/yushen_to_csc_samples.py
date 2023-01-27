# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : yushen_to_csc_samples.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-07-13
#   desc     : yushen_hidden_to_csc_data, for each item in json files:
#              keys are {'sid', 'type', 'tips', }
# ==========================================================================
import json
from tqdm import tqdm
from collections import defaultdict, Counter


error_dict = {
    1000: "自定义错别字",
    1010: "可疑错字",
    10001: "值中含有不当逗号",
    10002: "英文逗号作为断句",
    10003: "连续的值需要分隔",
    10004: "doc文档中上下标号使用错误",
    10005: "连续的时间需要分隔",
    10006: "值中包含多个小数点",
    10007: '连续的值应该用顿号或"和"连接，而不是用逗号',
    10008: '文中含有重复标点',

    20001: '同音异字',
    20002: '近音异字',
    20003: '类形异字',
    20005: '低频错字',
    20006: '分词扩展',
    20007: '多字情况',
    20008: '漏字情况',

    30001: "主承销商名称错误",
}


def flatten(nested_list):
    return [elem for sub_list in nested_list for elem in sub_list]


def show_each_error_type(tr):
    done_list = []
    for sample in tqdm(tr):
        if sample['type'] not in done_list:
            done_list.append(sample['type'])
            print(sample['type'], error_dict.get(sample['type'], '?'))
            print(sample)


def check_for_negative_position(tr):
    ct = Counter([1 if sp['position'][0] < 0 else 0 for sp in tr])
    print(sorted(ct.items()))  # => [(0, 228544), (1, 169)]

    negative_position_samples = [sp for sp in tr if sp['position'][0] < 0]
    json.dump(negative_position_samples,
              open('./negative_position_samples.json', 'w'))


def check_for_oversize_position(tr):
    ct = Counter([1 if sp['position'][0] > len(sp['source_text']) else 0 for sp in tr])
    print(sorted(ct.items()))  # => [(0, 153804), (1, 74909)]

    oversize_position_samples = [sp for sp in tr if sp['position'][0] > len(sp['source_text'])]
    json.dump(oversize_position_samples,
              open('./oversize_position_samples.json', 'w'))


def disconnect_replace(text, src, dst):
    ret = []
    pivot = 0
    if len(src) != len(dst):
        print(src, dst)
        return None
    for c in text:
        if pivot < len(src) and c == src[pivot]:
            ret.append(dst[pivot])
            pivot += 1
        else:
            ret.append(c)
    return ''.join(ret)


def generate_csc_pair(original_text, correction_text, wrong_ids=None):
    pairs = []
    if wrong_ids is None:
        wrong_ids = [_i for _i, (_o, _c) in
                     enumerate(zip(original_text, correction_text)) if _o != _c]
    pivot = -2
    for _i, wid in enumerate(wrong_ids):
        if wid - pivot != 1:
            if pivot > 0:
                pairs.append((original_text[pivot: wrong_ids[_i - 1] + 1],
                              correction_text[pivot: wrong_ids[_i - 1] + 1]))
            pivot = wid + 0
    else:
        if wrong_ids:  # the last one
            pairs.append((original_text[pivot: wrong_ids[-1] + 1],
                          correction_text[pivot: wrong_ids[-1] + 1]))
    return pairs


def ner_crossover(lef, rig, range_list):
    # [e['span'] for e in entities if e['type'] != '文件']
    for l, r in range_list:
        if r <= lef or l >= rig:
            continue
        return True
    return False


def generate_csc_samples(sid_case, sid2ner):
    """

    :param sid_case: {sid: sample}
    :param sid2ner: {sid: ner_results}
    :return:

    PATHS:
    '/home/chendian/download/csc_data/faulty_wordings_with_hidden.json'
    '/home/chendian/BertBasedCorrectionModels/datasets/faulty_wordings_with_hidden_ner_cd.json'
    """

    rst = []
    type_errors = []
    related_errors = [1010, 10001, 20001, 20002, 20003, 20005, 20006, 20007]
    for sid in tqdm(sid_case):
        related_samples = [sp for sp in sid_case[sid]
                           if sp['type'] in related_errors]
        if len(related_samples) == 0:
            continue

        # generate a sample on single sentence
        original_text = f"{related_samples[0]['source_text']}"
        correction_text = f'{original_text}'

        for sample in related_samples:
            if sample['hidden']:
                continue  # it is not a faulty wording

            _lef, _rig = sample['position']
            entities = sid2ner.get(sid)
            if entities:
                range_list = [e['span'] for e in entities if e['type'] != '文件']
                if ner_crossover(_lef, _rig, range_list):
                    continue  # faulty labelled data, (found with NER)

            if _lef < 0:
                continue  # bad case
            if len(sample['tips']) == 1:
                if sample['type'] == 10001:  # wrong commas
                    _f, _t = '，', ','
            else:
                _f, _t = sample['tips']
                if sample['type'] == 20007:  # redundant words
                    _t = ''.join([_c if _c in _t else '✪' for _c in _f])
            if len(_f) != len(_t):
                # pprint(sample)
                # print(original_text, correction_text)
                type_errors.append(sid)
                break
            mid_substr = disconnect_replace(correction_text[_lef: _rig], _f, _t)
            if not mid_substr:
                print(sid)
                print(original_text)
                print(f'[{_lef}, {_rig}] / {len(original_text)}')
                print('related_tokens', original_text[_lef:_rig])
                print(f'tips_tokens: {_f}->{_t}')
                raise ValueError()
            correction_text = f"{correction_text[:_lef]}{mid_substr}{correction_text[_rig:]}"
        else:
            ot, ct = original_text.replace(' ', ''), correction_text.replace(' ', '')
            wrong_ids = [_i for _i, (_o, _c) in enumerate(zip(ot, ct)) if _o != _c]
            csc_pair = generate_csc_pair(original_text, correction_text, wrong_ids)
            rst.append({
                'id': sid,
                'original_text': ot,
                'wrong_ids': wrong_ids,
                'correct_text': ct,
                'csc_pair': csc_pair,
                'ner': sid2ner.get(sid, [])
            })
    # a list of samples in form of csc_data.
    return rst


def check_faulty_labelled_with_ner():
    tr = json.load(
        open('/home/chendian/download/csc_data/' +
             'faulty_wordings_with_hidden.json', 'r'))
    tr_ner = json.load(
        open('/home/chendian/BertBasedCorrectionModels/datasets/csc/' +
             'faulty_wordings_with_hidden_ner_cd.json', 'r'))

    sid2ner = {sample['id']: sample['ner'] for sample in tr_ner}

    related_errors = [1010, 10001, 20001, 20002, 20003, 20005, 20006, 20007]
    check_samples = []

    def crossover(lef, rig, range_list):
        for l, r in range_list:
            if r <= lef or l >= rig:
                continue
            return True
        return False

    solve = 0
    total = 0
    for sample in tr:
        # if not sample['hidden']  # check for NER solving false alarms
        if sample['hidden']:  # check for faulty labelling
            continue
        if sample['type'] not in related_errors:
            continue
        sid = f"{sample['doc_id']}-{sample['doclet_type']}-{sample['sid']}"
        entities = sid2ner.get(sid)
        lef, rig = sample['position']
        if entities and crossover(lef, rig, [e['span'] for e in entities if e['type'] != '文件']):
            solve += 1
            check_samples.append(sample)
            check_samples[-1].update({'ner': entities})
        total += 1

    # show sample counts and ratio
    print(solve, total, solve / total)

    json.dump(check_samples, open(
        '/home/chendian/download/csc_data/' +
        'faulty_labelled_ys_data_with_ner.json', 'w'))


if __name__ == "__main__":
    dir_path = '/home/chendian/download/csc_data/'
    file_name = 'faulty_wordings_with_hidden.json'
    tr = json.load(open(dir_path + file_name, 'r'))
    # show_each_error_type(tr)

    doc_case = defaultdict(list)
    doc_hidden_counts = defaultdict(int)
    for sample in tqdm(tr):
        id_str = f"{sample['doc_id']}-{sample['doclet_type']}"
        doc_case[id_str].append(sample)
        doc_hidden_counts[id_str] += int(sample['hidden'])

    tr_with_hidden = flatten([samples for doc_id, samples in doc_case.items()
                              if doc_hidden_counts[doc_id] > 0])
    sid_case = defaultdict(list)
    for sample in tqdm(tr_with_hidden):
        id_str = f"{sample['doc_id']}-{sample['doclet_type']}-{sample['sid']}"
        sid_case[id_str].append(sample)

    ner_file_name = 'faulty_wordings_with_hidden_ner_cd.json'
    sid2ner = json.load(open(dir_path + ner_file_name, 'r'))

    results = generate_csc_samples(sid_case, sid2ner)
    fp = dir_path + file_name.replace('.json', '_cd.json')
    with open(fp, 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, separators=(',', ':'))
