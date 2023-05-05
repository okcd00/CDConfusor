# ==========================================================================
#   Copyright (C) since 2020 All rights reserved.
# 
#   filename : grpc_server.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-04-16
#   desc     : server in grpc service
# ==========================================================================
import sys
import time
import grpc
from urllib import response
from concurrent import futures
from multiprocessing import Pool
from collections import OrderedDict, defaultdict
from grpc._cython.cygrpc import CompressionAlgorithm, CompressionLevel


"""
if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

# generate it at first
python -m grpc_tools.protoc -I ./ --python_out=. --grpc_python_out=. correction.proto
# then you can get correction_pb2_grpc and correction_pb2
"""


import correction_pb2, correction_pb2_grpc
import re
re_han = re.compile("([\u4E00-\u9Fa5a-zA-Z0-9+#&]+)", re.U)

import json
import torch
import numpy as np
from tqdm import tqdm

from predict_DCN import result_predict
from inference import load_model


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def is_pure_chinese_phrase(phrase_str):
    # all tokens are Chinese
    return False not in list(map(is_chinese_char, map(ord, phrase_str)))


def B2Q(uchar):
    if len(uchar) > 1:
        print(f"B2Q can not solve the {len(uchar)}-gram: {uchar}")
        return uchar
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e: # 不是半角字符就返回原来的字符
        return uchar 
    if inside_code == 0x0020: # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code).upper()


def clean_texts(text):
    if isinstance(text, list):
        return [clean_texts(t) for t in text]
    # B2Q + upper
    return ''.join([B2Q(c) for c in text])


def split_2_short_text(text, include_symbol=False):
    """
    长句切分为短句
    :param text: str
    :param include_symbol: bool
    :return: (sentence, idx)

    Examples:
    [..., ('然而', 200), ('，', 202), ...]
    """
    result = []
    blocks = re_han.split(text)
    start_idx = 0
    for blk in blocks:
        if not blk:
            continue
        if include_symbol:
            result.append((blk, start_idx))
        else:
            if re_han.match(blk):
                result.append((blk, start_idx))
        start_idx += len(blk)
    return result


def predict_on_texts(input_lines, model, tokenizer,
                     batch_size=4, max_len=192, return_fo=False):
    # pre-process texts:
    dcn_lines = clean_texts(input_lines)
    dcn_items = [[w[2:] if w.startswith('##') else w
                  for w in tokenizer.tokenize(text)] 
                 for text in dcn_lines]
    
    # record missing blanks
    blank_indexes = [
        [i for i, c in enumerate(line) if c == B2Q(' ')]
        for line in enumerate(dcn_lines)]
    
    # predict:
    result_items = result_predict(
        sentence_list=dcn_lines, 
        tokenizer=tokenizer, model=model, device='cuda', 
        batch_size=batch_size, max_seq_length=max_len)
    # [CLS] and [SEP]/[PAD] are removed
    result_items = [[w[2:] if w.startswith('##') else w
                     for w in res] 
                    for res in result_items]
    # compare
    outputs = []
    for idx, (inp, out) in enumerate(zip(dcn_items, result_items)):
        # print(len(inp), inp)
        # print(len(out), out)
        offset = 0
        blanks = blank_indexes[idx]
        corrected_line = f"{input_lines[idx]}"
        # print(corrected_line)
        if inp != out:    
            for i, (c1, c2) in enumerate(zip(inp, out)):
                if offset in blanks:
                    offset += 1
                elif c1.startswith('[') and c1.endswith(']'):
                    # [UNK] and [PAD]
                    offset += 1
                else:
                    offset += len(c1)
                if c1 == c2:
                    continue
                if is_pure_chinese_phrase(c1) and is_pure_chinese_phrase(c2):
                    # print(i, (c1, c2))
                    # print(len(corrected_line), offset, offset-len(c1))
                    corrected_line = f"{corrected_line[:offset-len(c1)]}{c2}{corrected_line[offset:]}"
        outputs.append(corrected_line)
    if return_fo: # return full outputs
        return outputs, dcn_items, result_items
    return outputs


class CSCServicer(correction_pb2_grpc.CorrectionServicer):
    def __init__(self, debug=False, return_detail=False):
        self.debug = debug
        self.return_detail = False  # temporarily disabled confidence value.
        self.max_len = 180
        self.records = OrderedDict()

        model_path = '/home/chendian/CDConfusor/exp/dcn/cd_models/findoc_finetuned_230410_multigpu/'

        model, tokenizer = load_model(model_path=model_path)
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_ignore_mask_for_predict(self, text, ignore_mask):
        # the input ignore_mask is char-level from client.
        if isinstance(text, list):
            assert len(text) == len(ignore_mask)
            mask_case = [self.generate_ignore_mask_for_predict(text, im) 
                         for text, im in zip(text, ignore_mask)]

            # 1 for ignore, has [CLS] and [SEP] 
            max_len = max(map(len, mask_case))
            mask_case = [np.pad(msk, (0, (max_len-len(msk))), constant_values=1) 
                         for msk in mask_case]
            # the ignore_mask consists of 0(normal) and 1(ignore)
            # return torch.from_numpy(np.stack(mask_case))
            return np.stack(mask_case)

        def judge_valid_token_for_csc(token):
            if token.startswith('##'):
                token = token[2:]
            if len(token) > 1:
                return 1  # invalid (ignore)
            if is_chinese_char(ord(token)):
                return 0  # valid
            return 1  # invalid (ignore)
        
        def mark_entities(ignore_mask, tok_case, _text):
            _valid = [judge_valid_token_for_csc(tok) for tok in tok_case]

            char_to_word_idx = []  # mapping characters to tokenizer-level words
            for tok_idx, tok in enumerate(tok_case):
                if tok.startswith('##'):
                    tok = tok[2:]
                for c in tok:
                    char_to_word_idx.append(tok_idx)
            char_to_word_idx.append(len(tok_case))  # end
            # rest_str = f"{_text}"
            
            for i, d in enumerate(ignore_mask):
                if d == 1:
                    _valid[char_to_word_idx[i]] = 1
            # the single ignore_mask consists of 0(ignore) and 1(normal)
            # the flag will be reversed in batch.
            ignore_case = np.array([1] + _valid + [1])
            return ignore_case            
        
        try:
            # `text` reach here is a single sentence
            # list w/o [CLS] ... [SEP] 
            tok_case = self.tokenizer.tokenize(
                clean_texts(convert_to_unicode(text)))
            # nparray w/ [CLS] ... [SEP] 
            ignore_case = mark_entities(ignore_mask, tok_case, text)
            assert len(tok_case) + 2 == len(ignore_case)
        except Exception as e:
            for t, m in zip(tok_case, ignore_case[1:-1]):
                print(t, m)
            print(len(tok_case))
            print(tok_case)
            print(len(ignore_case))
            print(ignore_case)     
            raise e       
        # tokenizer-level
        return ignore_case

    def get_metric(self, input_lines, truth_lines, outputs):
        results = list(zip(input_lines, truth_lines, outputs))
        from exp.dcn.evaluate import compute_corrector_prf_faspell
        metrics = compute_corrector_prf_faspell(results, strict=True)
        if self.detailed_evaluate:
            return metrics
        p, r, f, acc = metrics['cor_sent_p'], metrics['cor_sent_r'], \
            metrics['cor_sent_f1'], metrics['cor_sent_acc']
        p, r, f, acc = list(map(lambda x: round(x*100, 2), [p, r, f, acc]))
        print(f"{p}/{r}/{f} | {acc}")
        return p, r, f, acc

    def evaluate(self, input_lines, truth_lines=None, 
                 det_mask=None, batch_size=8, detail=False):
        # predict on splitted segments.
        _lines = []
        segment_id = 0
        segment_mapping = []  # line_id: [segment_id1, segment_id2]
        for line_idx, line in enumerate(input_lines):
            segment_mapping.append([])
            if len(line) < 179:  # short sentences
                _lines.append(line)
                segment_mapping[line_idx].append(segment_id)
                segment_id += 1
                continue
            head, buffer = 1, ""  
            split_cases = split_2_short_text(line, include_symbol=True)
            split_cases = [(split_cases[i][0] + (split_cases[i+1][0] if i+1 < len(split_cases) else ""), 
                            split_cases[i][1]) 
                           for i in range(0, len(split_cases), 2)]
            for subtext, start_idx in split_cases:
                subtext_size = len(subtext)
                if len(buffer) + subtext_size >= self.max_len:
                    _lines.append(buffer)  # char-level tokens
                    segment_mapping[line_idx].append(segment_id)
                    segment_id += 1
                    buffer = ""
                    head = start_idx
                buffer += subtext
            else:
                if buffer:
                    _lines.append(buffer)
                    segment_mapping[line_idx].append(segment_id)
                    segment_id += 1
                    buffer = ""
        try:
            assert sum(list(map(len, _lines))) == sum(list(map(len, input_lines)))
        except Exception as e:
            print(segment_mapping)
            print(list(map(len, _lines)))
            print(list(map(len, input_lines)))
            raise e

        # outputs may be a 1-item case or 3-item case
        # a list of texts [, dcn_items] [, result_items]
        outputs = predict_on_texts(
            input_lines=_lines, 
            model=self.model, 
            tokenizer=self.tokenizer,
            batch_size=batch_size, 
            max_len=self.max_len,
            return_fo=detail)
        
        prediction_texts = outputs[0] if detail else outputs
        
        if truth_lines is not None:
            p, r, f, acc = self.get_metric(
                input_lines, truth_lines, outputs)
        
        try:
            assert len(det_mask) == len(segment_mapping)
            assert len(prediction_texts) == sum(map(len, segment_mapping))
        except Exception as e:
            for _d in det_mask:
                print('tokenizer-level mask:', _d.shape)
            for _s in segment_mapping:
                segment_size = [len(prediction_texts[segment_id]) for segment_id in _s]
                print('char-level length:', '+'.join(list(map(str, segment_size))))

        _texts = []
        _dcn_items = []
        _result_items = []
        for line_id, segment_ids in enumerate(segment_mapping):
            if detail:
                bucket = ["", [], []]
            for segment_id in segment_ids:
                bucket[0] += prediction_texts[segment_id]
                if detail:
                    bucket[1].extend(outputs[1][segment_id])
                    bucket[2].extend(outputs[2][segment_id])

            # drop ignored positions in result_items.
            mask = det_mask[line_id]  # tokenizer-based token level, w/ [CLS]
            assert bucket[1].__len__() == bucket[2].__len__() 
            if mask.shape[0] < bucket[1].__len__():
                for i, b in enumerate(bucket[1]):
                    m = mask[i+1] if i < mask.shape[0] else 'X'
                    print(i, m, b, bucket[2][i])
            for idx, (o, p) in enumerate(zip(bucket[1], bucket[2])):
                if p.upper() == '[UNK]':
                    if o.upper() == '[UNK]':
                        bucket[2][idx] = '✿'
                    else:
                        bucket[2][idx] = bucket[1][idx]
                    continue
                elif o == p:
                    continue
                elif o != '[UNK]' and B2Q(o) == p.upper():
                    continue  
                try:
                    if mask[idx + 1] == 1:  # considering [CLS]
                        bucket[2][idx] = bucket[1][idx]
                except Exception as e:
                    print(mask.shape)
                    print(idx)  # failed at 257-th item
                    print(bucket[1].__len__(), bucket[2].__len__())
                    for i, (o, p) in enumerate(list(zip(bucket[1], bucket[2]))):                       
                        if -5 < i-idx < 5:
                            print(i, o, p)
                    print(bucket[1][idx], bucket[2][idx])
                    raise e

            # append into return items.
            _texts.append(bucket[0])
            _dcn_items.append(bucket[1])
            _result_items.append(bucket[2])               

        if detail:
            # a list of texts [, dcn_items] [, result_items]
            return _texts, _dcn_items, _result_items
        # a list of texts
        return _texts

    def ask(self, request, context):
        # request is a list of texts (json str)
        # context is a placeholder for other features.
        if self.debug:
            print(request)
        sentence_list = json.loads(request.key)

        if len(sentence_list) == 0:
            outputs = []
        else:
            texts = [s['sentence'] for s in sentence_list]
            ignore_mask = [s['ignore_mask'] for s in sentence_list]
            # with [CLS] and [SEP]
            # the mask consists of 0(normal) and 1(ignore)
            try:
                det_mask = self.generate_ignore_mask_for_predict(
                    texts, ignore_mask)             
            except Exception as e:
                print("Error at generate_ignore_mask_for_predict().")
                print(texts, ignore_mask)
                raise e
            if self.debug:
                print(request)
                print(sentence_list)
            try: 
                outputs, dcn_items, result_items = self.evaluate(
                    input_lines=texts, det_mask=det_mask, detail=True)
                outputs = [
                    {'text': output, 
                     'dcn_items': dcn_item, 
                     'result_items': result_item}
                    for output, dcn_item, result_item 
                    in zip(outputs, dcn_items, result_items)]
            except Exception as e:
                print(f"Error in evaluate(): ", str(e))
                outputs = []
        
        if self.debug:
            print(len(texts), len(outputs))
            for i, o in enumerate(outputs):
                print(f"Input[{i}]:")
                print(texts[i])
                print(f"Output[{i}]:")
                print(o['text'])

        # outputs = self.construct_response_dict_list(outputs)
        response_str = json.dumps(outputs)
        return correction_pb2.Response(value=response_str)

    def remember(self, request, context):
        # dummy function
        key, value = request.key, request.value
        self.records.update({key: value})
        return correction_pb2.Response(
            value="Remembered: the value for {} is {}".format(key, value))


def serve(n_worker=1, port=20417):
    print("starting server...", time.ctime())

    max_receive_message_length = 100 * 1024
    service = CSCServicer(debug=False, return_detail=False)
    service.process_pool = Pool(processes=n_worker)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=n_worker), options=[
        ('grpc.max_receive_message_length', max_receive_message_length),
        ('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
        ('grpc.grpc.default_compression_level', CompressionLevel.high)
    ])
    correction_pb2_grpc.add_CorrectionServicer_to_server(service, server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    print("server started.", time.ctime())
    
    # server.wait_for_termination()
    # return server

    try:
        while True:
            time.sleep(12 * 60 * 60)
            print("[ALIVE] {}".format(time.ctime()))
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
    # line = "作为一个写作和编程助手，我可以为您解答这个问题。适当的时间管理和休息是非常重要的，以避免过度劳累。"
    # print(split_2_short_text(line, include_symbol=True))
    # service = CSCServicer(debug=False, return_detail=False)
    # service.ask()
