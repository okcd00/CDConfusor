# ==========================================================================
#   Copyright (C) since 2020 All rights reserved.
#
#   filename : grpc_client.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-04-16
#   desc     : client in grpc service  (version 0.0.2)
# ==========================================================================
import os
import re
import sys
import grpc
import json
from tqdm import tqdm
from pprint import pprint

"""
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

# generate it at first
python -m grpc_tools.protoc -I ./ --python_out=. --grpc_python_out=. keyvaluestore.proto
# then you can get keyvaluestore_pb2_grpc and keyvaluestore_pb2
"""

from grpc_csc import correction_pb2, correction_pb2_grpc

"""
20006: generate_miss_mirror,
20007: generate_expand_mirror,
20008: generate_reduce_mirror,
20009: generate_unknown_mirror,
20010: generate_cloze_mirror,
"""

def test():
    with grpc.insecure_channel('localhost:20416') as channel:
        stub = correction_pb2_grpc.CorrectionStub(channel)
                
        test_json_str = json.dumps([u'今天天气不错？', u'假设这是一句有错误的橘子。', '我司决定发行超短裙融资券'])
        ask_item = correction_pb2.Item(key=test_json_str)
        print(json.loads(test_json_str))
        response = stub.ask(ask_item)
        print(json.loads(response.value))
        # response = stub.remember(correction_pb2.Key(key='record', value=test_json_str))
        # print(response)


def test_speed():
    with grpc.insecure_channel('localhost:20416') as channel:
        stub = correction_pb2_grpc.CorrectionStub(channel)
        
        res = [line for line in open('/data/chendian/cleaned_findoc_samples/autodoc_test.220424.txt', 'r')]
        batch_size = 16

        from tqdm import tqdm
        for i in tqdm(range(len(res) // batch_size)):
            samples = res[batch_size*i: batch_size*i+batch_size]
            test_json_str = json.dumps([s.split('\t')[0] for s in samples])
            ask_item = correction_pb2.Item(key=test_json_str)
            # print(json.loads(test_json_str))
            response = stub.ask(ask_item)
            # print(json.loads(response.value))

        # response = stub.remember(correction_pb2.Key(key='record', value=test_json_str))
        # print(response)


def pick_keys(autodoc_sentence_list, target_keys):
    ret = []
    for sample in autodoc_sentence_list:
        _sample = {k: sample.get(k) for k in target_keys}
        ret.append(_sample)
    return ret


def pre_process_grpc_items(autodoc_sentence_list):
    ret = []
    for sample in autodoc_sentence_list:
        _sample = {
            'words': sample['words'],
            'sentence': sample.get('sentence', sample.get('text', "")),
            'ignore_mask': [0 for _ in range(len(sample['sentence']))]}
        for key in ['times', 'attributes', 'values', 'preattributes']:
            for _, dic in sample.get(key, {}).items():
                if dic.get('tag'):
                    if dic['tag'] in ['sector', 'project', 'document']:
                        continue  # we also detect errors in these mentions.
                _text = dic['value']
                _position = dic['position']
                # _word_index = dic['word_index']
                for _i in range(_position, _position + len(_text)):
                    _sample['ignore_mask'][_i] = 1
        ret.append(_sample)
    return ret

def extract_faulty_positions(ori_sent, pred_sent, confidence, predict_tokens, words):
    ret = []
    char_index = 0  # char-level index (for pure text)
    token_index = 0  # token-level index (for BertTokenizer)
    for word_index, w in enumerate(words):  # word-level index (for `words``)
        delta_token_index = 0  # how many tokens the word takes
        rest_chars = len(w)  # how many chars retained for the current word
        while rest_chars > 0:
            rest_chars -= len(predict_tokens[token_index])
            delta_token_index += 1

        ori = ori_sent[char_index: char_index+len(w)].lower()
        pred = pred_sent[char_index: char_index+len(w)].lower()
        if ori != pred:
            cv = 1.  #  confidence value
            _conf_case = []
            if confidence is not None:
                # print(confidence)
                for _i, (_o, _p) in enumerate(zip(ori, pred)):
                    if _o != _p:
                        _conf_case.append(confidence[token_index+_i])
                if len(_conf_case):
                    cv = sum(_conf_case) / len(_conf_case)
            if cv < 0.001:
                continue
            tup = ( 
                (char_index, char_index+len(w)),  # token-level position
                (word_index, word_index+1),  # word-level position
                ori, pred,  # original token and predict token
                _conf_case  # confidence value for each faulty token
            )
            ret.append(tup)
        
        char_index += len(w)
        token_index += delta_token_index

    return ret


def draft_token_idx_to_word_idx(tokens, words):
    # update word index
    word_index = 0
    for w in tokens:
        _l = len(w)
        while _l > 0:
            if rest_tokens >= _l:
                rest_tokens -= _l
                break
            _l -= rest_tokens
            word_index += 1
            rest_tokens = len(words[word_index])
        print(w, word_index)


def special_judge_for_texts(text):
    if len(text) == 1:  # issue: 过滤单字句子
        return True
    if '..........' in text:  # issue: 目录
        return True
    return False


def special_judge_for_pairs(o, p):
    if re.match("^[一二三四五六七八九十]+$", o):  # issue: 纯数字文本
        return True
    pinyin_flag = False  # judge_exactly_different_pinyins
    if pinyin_flag and len(o) == len(p):
        from pypinyin import lazy_pinyin
        for _pyo, _pyp in zip(lazy_pinyin(o), lazy_pinyin(p)):
            if len(set(_pyo).intersection(set(_pyp))) == 0:
                return True
    return False


def correction(address, autodoc_package, batch_size=16):
    output_target_keys = [
        'sid', 'file_id', 'sentence', 'source_text',
        'times', 'preattributes', 'attributes', 'values',
        'complete_attributes', 'quadruples', 'words']

    # pre-processing for all samples
    autodoc_package = pick_keys(autodoc_package, target_keys=output_target_keys)

    def post_process_grpc_items(response_list, offset=0, ignore_mask=None):
        # each item in response list now is a dict.
        for i, res in enumerate(response_list):
            fw = []
            if ignore_mask:
                white_mask = ignore_mask[i]
            ori_sent = autodoc_package[offset+i]['sentence']
            if special_judge_for_texts(ori_sent):
                autodoc_package[offset+i]['faulty_wordings'] = fw
                continue
            words = autodoc_package[offset+i]['words']
            if isinstance(res, dict):
                pred_sent = res['text']
                pred_cfd = res.get('detection_prob')
                pred_tokens = res.get('text')
            else:  # for version 0.0.1
                pred_sent = res
                pred_cfd = None
                pred_tokens = None
            # print(res['detection_prob'])
            # print(len(ori_sent), len(pred_sent), len(pred_cfd))
            for _results in extract_faulty_positions(
                    ori_sent, pred_sent, pred_cfd, pred_tokens, words):
                (_l, _r), (l, r), o, p, cv = _results
                if ignore_mask is not None and sum(white_mask[_l: _r]) > 0:
                    continue
                if special_judge_for_pairs(o, p):
                    continue
                probability = 1.
                if len(cv) > 0:                    
                    probability = round(sum(cv) / len(cv), 5)
                tip_code = 20001
                tips = '近音异字: {}->{}'.format(o, p)
                if len(cv) > 1:
                    # 当前模型连续错字做不好，暂设为可疑
                    tip_code = 20009
                    tips = '可疑错字: {}->{}'.format(o, p)
                fw.append({
                    'position': [l, r],
                    'position_token': [_l, _r],
                    'type': tip_code,
                    'tips': tips,
                    'probability': probability,
                    'prob_detail': cv,
                })
            autodoc_package[offset+i]['faulty_wordings'] = fw
            # autodoc_sentence_list[offset+i]['faulty_filters'] = []
        # return autodoc_sentence_list
        
    with grpc.insecure_channel(address) as channel:
        stub = correction_pb2_grpc.CorrectionStub(channel)
        for i in tqdm(range(len(autodoc_package) // batch_size + 1)):
            
            samples = pre_process_grpc_items(
                autodoc_package[batch_size*i: batch_size*i+batch_size])
            test_json_str = json.dumps(samples)
            ask_item = correction_pb2.Item(key=test_json_str)
            response_list = json.loads(stub.ask(ask_item).value)
            # print('response', response_list)
            post_process_grpc_items(
                response_list=response_list,
                ignore_mask=[s['ignore_mask'] for s in samples],
                offset=i*batch_size
            )
        # response = stub.remember(correction_pb2.Key(key='record', value=test_json_str))
        # print(response)
    return autodoc_package


if __name__ == '__main__':
    import sys
    sys.path.append('./')

    autodoc_package = json.load(
        open('./faulty_wording_input.json', 'r'))[1000:1500]
    # pprint(autodoc_package)

    autodoc_package = correction('localhost:20416', autodoc_package, batch_size=16)
    for i, item in enumerate(autodoc_package):
        if item['faulty_wordings']:
            print(item['sentence'])
            pprint(item['faulty_wordings'])
