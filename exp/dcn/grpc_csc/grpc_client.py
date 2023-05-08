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
20001: 近音错误
20005: 未知错误
20006: 漏字错误
20007: 多字错误
29999: 疑似错误（模型不确定）
"""

def test():
    with grpc.insecure_channel('localhost:20416') as channel:
        stub = correction_pb2_grpc.CorrectionStub(channel)
                
        test_json_str = json.dumps([u'今天天气不错？', u'假设这是一句有错误的橘子。', '他今天的行为橘子很奇怪', '我司决定发行超短裙融资券'])
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


def B2Q(uchar):
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e: # 不是半角字符就返回原来的字符
        return uchar 
    if inside_code == 0x0020: # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code).upper()


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
                if dic.get('tag') and False:  # now we ignore errors in these.
                    if dic['tag'] in ['sector', 'project', 'document']:
                        continue  # we also detect errors in these mentions.
                _text = dic['value']
                _position = dic['position']
                # _word_index = dic['word_index']
                for _i in range(_position, _position + len(_text)):
                    _sample['ignore_mask'][_i] = 1  # 1 for ignore
        ret.append(_sample)
    return ret

def extract_faulty_positions(ori_sent, pred_sent, confidence, predict_tokens, words):
    ret = []
    char_index = 0  # char-level index (for pure text)
    token_index = 0  # token-level index (for BertTokenizer)
    # print(ori_sent)
    # print("".join(predict_tokens))
    # print(len(confidence), len(predict_tokens), len(words))
    if confidence is None:
        if predict_tokens is None or words is None:
            for idx, (_o, _p) in enumerate(zip(ori_sent, pred_sent)):
                if _o != _p and _p.upper() not in ['[UNK]', '✿']:
                    # TODO: add confidence value for prediction
                    _conf_case = [1. for _ in range(len(_o))]
                    tup = ( 
                        (idx, idx+len(_o)),  # char-level position
                        (-1, -1),  # word-level position
                        _o, _p,  # original token and predict token
                        _conf_case  # confidence value for each faulty token
                    )
                    ret.append(tup)
            return ret
    elif len(confidence) < len(predict_tokens):
        return ret  # failed in streak arrangement.
    
    rest_chars = 0
    # print(len(''.join(words)), words)
    for word_index, w in enumerate(words):  # word-level index (for `words``)
        delta_token_index = 0  # how many tokens the word takes
        rest_chars += len(w)  # how many chars retained for the current word
        while rest_chars > 0:
            # print(predict_tokens[token_index + delta_token_index], len(predict_tokens))
            rest_chars -= len(predict_tokens[token_index + delta_token_index])
            # print(predict_tokens[token_index + delta_token_index], end=', ')
            delta_token_index += 1
            if token_index + delta_token_index >= len(predict_tokens):
                break
        # print("")
        pred_tokens = predict_tokens[token_index: token_index + delta_token_index]
        pred = ''.join(pred_tokens).lower()
        ori = ori_sent[char_index: char_index+len(pred)].lower()

        if ori != pred and (''.join(B2Q(_ori) for _ori in ori) != pred.upper()):
            cv = 1.  #  confidence value 
            _conf_case = []
            if confidence is not None:
                # print(confidence)
                _offset = 0
                for _i, _p in enumerate(pred_tokens):
                    _o = ''.join(ori[_offset: _offset+len(_p)])
                    _offset += len(_p)
                    # print(f"[{_o}:{_offset}:{_offset+len(_p)}], [{_p}]")
                    if _o != _p:
                        _conf_case.append(confidence[token_index+_i])
                if len(_conf_case):
                    cv = sum(_conf_case) / len(_conf_case)
            if '✿' in pred:  # UNK in prediction
                continue
            if cv > 0.001:  # 当前模型自信度过低
                tup = ( 
                    (char_index, char_index+len(w)),  # char-level position
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
    if len(text) < 5:  # issue: 过滤单字句子
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


def correction(address, autodoc_package, batch_size=16, word_level='word'):
    output_target_keys = [
        'sid', 'file_id', 'sentence', 'source_text',
        'times', 'preattributes', 'attributes', 'values',
        'complete_attributes', 'quadruples', 'words']

    # pre-processing for all samples
    autodoc_package = pick_keys(
        autodoc_package, target_keys=output_target_keys)

    def post_process_grpc_items(response_list, offset=0, ignore_mask=None):
        # each item in response list now is a dict.
        for i, res in enumerate(response_list):
            fw = []
            if ignore_mask:
                white_mask = ignore_mask[i]
            cur_item = autodoc_package[offset+i]
            ori_sent = cur_item['sentence']

            if special_judge_for_texts(ori_sent):
                autodoc_package[offset+i]['faulty_wordings'] = fw
                continue
                
            if isinstance(res, dict):
                pred_sent = res['text']
                pred_cfd = res.get('detection_prob')
                pred_tokens = res.get('text')
            else:  # for version 0.0.1
                pred_sent = res
                pred_cfd = None
                pred_tokens = None
            # print(res['detection_prob'])
            # print(len(ori_sent), len(pred_sent), len(pred_cfd), len(pred_tokens))

            if 'char' in word_level:
                words = [c for c in cur_item['source_text']]
            elif 'dcn' in word_level:
                # words = res.get('dcn_items')  # a list
                words = cur_item.get('words', res.get('dcn_items'))
                pred_sent = res.get('text')  # a string
                if '✿' in pred_sent:  # drop [UNK]s
                    for i, c in pred_sent:
                        if c in ['✿']:
                            pred_sent = f"{pred_sent[:i]}{ori_sent[i]}{pred_sent[i+1:]}"
                pred_tokens = res.get('result_items')  # a list
            else:  # 'word'
                words = cur_item.get('words', cur_item['source_text'])

            try:
                faulty_results = extract_faulty_positions(
                    ori_sent, pred_sent, pred_cfd, pred_tokens, words)
            except Exception as e:
                # print("Error in extract_faulty_positions()")
                print(str(e))
                autodoc_package[offset+i]['faulty_wordings'] = fw
                # return
                raise e
            for _results in faulty_results:
                (lpos_char, rpos_char), (lpos_word, rpos_word), o, p, cv = _results
                if ignore_mask is not None and sum(white_mask[lpos_char: rpos_char]) > 0:
                    continue
                if special_judge_for_pairs(o, p):
                    continue
                
                # adapt for version 0.0.1
                probability = round(sum(cv) / len(cv), 5) if len(cv) > 0 else 1.
                
                # generate tip codes.
                tip_code = 20001  # as default
                tips = '近音异字: {}->{}'.format(o, p)
                
                if len(cv) > 1 or probability < 0.1:  # 当前模型连续错字做不好，暂设为可疑
                    tip_code = 29999
                    tips = '可疑错字: {}->{}'.format(o, p)
                # append items.
                fw.append({
                    'position': [lpos_char, rpos_char],
                    'position_word': [lpos_word, rpos_word],
                    'position_char': [lpos_char, rpos_char],
                    'type': tip_code,
                    'tips': tips,
                    'probability': probability,
                    'prob_detail': cv,
                })
            # print(f"{offset+i}-th sentence marked")
            autodoc_package[offset+i]['faulty_wordings'] = fw
            # autodoc_sentence_list[offset+i]['faulty_filters'] = []
        # return autodoc_sentence_list
        
    with grpc.insecure_channel(address) as channel:
        stub = correction_pb2_grpc.CorrectionStub(channel)
        for i in tqdm(range(len(autodoc_package) // batch_size + 1)):
            # char-level ignore mask from client
            samples = pre_process_grpc_items(
                autodoc_package[batch_size*i: batch_size*i+batch_size])
            if len(samples) == 0:
                continue
            test_json_str = json.dumps(samples)
            # print(type(test_json_str))
            # print(test_json_str)
            ask_item = correction_pb2.Item(key=test_json_str)
            try:  # try to obtain the response.
                response_list = json.loads(stub.ask(ask_item).value)
            except Exception as e:
                print(str(e))  # catch exception from server.
                # pprint(samples)
                continue
            # add key 'faulty_wordings' here
            # print(response_list)
            post_process_grpc_items(  
                response_list=response_list,
                ignore_mask=[s['ignore_mask'] for s in samples],
                offset=i*batch_size)
        # response = stub.remember(correction_pb2.Key(key='record', value=test_json_str))
        # print(response)
    return autodoc_package


def test_port20416(autodoc_package, batch_size=4):
    autodoc_package = correction(
        'localhost:20416', autodoc_package, 
        batch_size=batch_size)
    for i, item in enumerate(autodoc_package):
        if item['faulty_wordings']:
            print(item['sentence'])
            # pprint(item['faulty_wordings'])
            for dic in item['faulty_wordings']:
                print(f"{dic['tips']}, confidence: {dic['probability']}")
                l, r = dic['position']
                print(item['sentence'][l:r], l, r, end=' | ')
                l, r = dic['position_word']
                print(item['words'][l:r], l, r)
            print("")


def test_port20417(autodoc_package, batch_size=4):
    autodoc_package = correction(
        'localhost:20417', autodoc_package, 
        batch_size=batch_size, word_level='dcn')
    n_pred = 0
    for i, item in enumerate(autodoc_package):
        if item.get('faulty_wordings') is None:  # no add-key
            print(f"No faulty_wordings key in {i}-th sample.")
            print(item['sentence'])
            continue
        if item['faulty_wordings'] == []:
            continue  # empty list
        print(item['sentence'])
        n_pred += 1
        # pprint(item['faulty_wordings'])
        for dic in item['faulty_wordings']:
            print(f"{dic['tips']}, confidence: {dic['probability']}")
            l, r = dic['position']
            print(item['sentence'][l:r], l, r, end=' | ')
            l, r = dic['position_word']
            print(item['words'][l:r], l, r)
        print("")
    print(f"find {n_pred} errors.")


if __name__ == '__main__':
    import sys
    sys.path.append('./')

    autodoc_package = json.load(
        open('./faulty_wording_input.json', 'r'))[:]  # [96*16:]
    print(len(autodoc_package))
    # pprint(autodoc_package)
    
    test_port20417(autodoc_package, batch_size=4)