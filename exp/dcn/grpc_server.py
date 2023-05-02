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
from collections import OrderedDict
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


import json
import torch
import numpy as np
from tqdm import tqdm

from exp.dcn.predict_DCN import result_predict
from exp.dcn.inference import load_model
from utils import is_chinese_char, convert_to_unicode, is_pure_chinese_phrase


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


def clean_texts(input_lines):
    # B2Q + upper
    return [''.join([B2Q(c) for c in t]) for t in input_lines]


def predict_on_texts(input_lines, model, tokenizer, det_mask=None,
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
        offset = 0
        blanks = blank_indexes[idx]
        corrected_line = f"{input_lines[idx]}"
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
                if det_mask is not None:
                    # [CLS]/[SEP] are removed
                    if det_mask[i+1] == 0:  
                        continue  # ignored indexes
                if is_pure_chinese_phrase(c1) and is_pure_chinese_phrase(c2):
                    # print(i, (c1, c2))
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

        model_path = '/home/chendian/CDConfusor/exp/dcn/cd_models/findoc_finetuned_230410_multigpu/checkpoint-374439/'

        model, tokenizer = load_model(model_path=model_path)
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_ignore_mask_for_predict(self, texts, ignore_mask):
        # the mask consists of 0(normal) and 1(ignore)
        if isinstance(texts, list):
            mask_case = [self.generate_ignore_mask_for_predict(text, im) 
                         for text, im in zip(texts, ignore_mask)]
            max_len = max(map(len, mask_case))
            mask_case = [1 - np.pad(msk, (0, (max_len-len(msk)))) 
                         for msk in mask_case]
            return torch.from_numpy(np.stack(mask_case))

        def judge_valid_token_for_csc(token):
            if token.startswith('##'):
                token = token[2:]
            if len(token) > 1:
                return 0  # invalid
            if is_chinese_char(ord(token)):
                return 1  # valid
            return 0  # invalid
        
        def mark_entities(ignore_mask, tok_case, _text):
            _valid = [judge_valid_token_for_csc(tok) for tok in tok_case]
            ignore_case = np.array([0] + _valid + [0])

            char_to_word_idx = []
            for tok_idx, tok in enumerate(tok_case):
                if tok.startswith('##'):
                    tok = tok[2:]
                for c in tok:
                    char_to_word_idx.append(tok_idx)
            char_to_word_idx.append(len(tok_case))  # end
            # rest_str = f"{_text}"
            
            for i, d in enumerate(ignore_mask):
                if d == 0:
                    continue
                ignore_case[char_to_word_idx[i]] = 1
            return ignore_case            
        
        tok_case = self.tokenizer.tokenize(convert_to_unicode(texts))
        # [CLS] ... [SEP] 
        ignore_case = mark_entities(ignore_mask, tok_case, texts)
        return ignore_case

    def construct_response_dict_list(self, outputs):
        if not self.return_detail:
            return [{'text': t} for t in outputs]

        response = [{'text': t} for t in outputs]
        # TODO: add confidence value outputs later.
        return response

    def evaluate(self, input_lines, truth_lines=None, 
                 det_mask=None, batch_size=4):
        outputs = predict_on_texts(
            input_lines=input_lines, 
            det_mask=det_mask,
            model=self.model, tokenizer=self.tokenizer,
            batch_size=batch_size, max_len=self.max_len)
        
        if truth_lines is not None:
            results = list(zip(input_lines, truth_lines, outputs))
            from exp.dcn.evaluate import compute_corrector_prf_faspell
            metrics = compute_corrector_prf_faspell(results, strict=True)
            if self.detailed_evaluate:
                return metrics
            p, r, f, acc = metrics['cor_sent_p'], metrics['cor_sent_r'], \
                metrics['cor_sent_f1'], metrics['cor_sent_acc']
            p, r, f, acc = list(map(lambda x: round(x*100, 2), [p, r, f, acc]))
            print(f"{p}/{r}/{f} | {acc}")
        
        return outputs  # a list of texts

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
            det_mask = self.generate_ignore_mask_for_predict(
                texts, ignore_mask)             
            if self.debug:
                print(request)
                print(sentence_list)
            outputs = self.evaluate(
                input_lines=texts, 
                det_mask=det_mask, 
                detail=self.return_detail)
        
        response = self.construct_response_dict_list(outputs)
        response_str = json.dumps(response)
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
    service = CSCServicer(debug=False, return_detail=True)
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
