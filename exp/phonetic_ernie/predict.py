# -*- coding: UTF-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import argparse
import numpy as np
from functools import partial

import paddle
import paddlenlp as ppnlp
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.transformers import ErnieTokenizer

from utils import convert_example, parse_decode

# yapf: disable
parser = argparse.ArgumentParser()  # (__doc__)
parser.add_argument("--model_file", default='./infer_model/static_graph_params.pdmodel', type=str, required=True, 
                    help="The path to model info in static graph.")
parser.add_argument("--params_file", default='./infer_model/static_graph_params.pdiparams', type=str, required=True, 
                    help="The path to parameters in static graph.")
parser.add_argument("--test_file", default='../data/cn/SIGHAN15_test/sighan15_test.txt', type=str, required=False, 
                    help="The path to model info in static graph.")
parser.add_argument("--batch_size", type=int, default=4, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path")

args = parser.parse_args()
# yapf: enable


class Predictor(object):
    def __init__(self, model_file, params_file, device, max_seq_length,
                 tokenizer, pinyin_vocab):
        self.max_seq_length = max_seq_length

        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]

        self.det_error_probs_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])
        self.corr_logits_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[1])
        self.tokenizer = tokenizer
        self.pinyin_vocab = pinyin_vocab

    def predict(self, data, batch_size=1):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `seq_len`(sequence length).
            batch_size(obj:`int`, defaults to 1): The number of batch.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []
        texts = []
        trans_func = partial(
            convert_example,
            tokenizer=self.tokenizer,
            pinyin_vocab=self.pinyin_vocab,
            max_seq_length=self.max_seq_length,
            is_test=True)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype='int64'),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype='int64'),  # segment
            Pad(axis=0, pad_val=self.pinyin_vocab.token_to_idx[self.pinyin_vocab.pad_token], dtype='int64'),  # pinyin
            Stack(axis=0, dtype='int64'),  # length
        ): [data for data in fn(samples)]

        for text in data:
            example = {"source": text.strip()}
            input_ids, token_type_ids, pinyin_ids, length = trans_func(example)
            examples.append((input_ids, token_type_ids, pinyin_ids, length))
            texts.append(example["source"])

        batch_examples = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        batch_texts = [
            texts[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        results = []

        for examples, texts in zip(batch_examples, batch_texts):
            token_ids, token_type_ids, pinyin_ids, length = batchify_fn(
                examples)
            self.input_handles[0].copy_from_cpu(token_ids)
            self.input_handles[1].copy_from_cpu(pinyin_ids)
            self.predictor.run()
            det_error_probs = self.det_error_probs_handle.copy_to_cpu()
            corr_logits = self.corr_logits_handle.copy_to_cpu()

            det_pred = det_error_probs.argmax(axis=-1)
            char_preds = corr_logits.argmax(axis=-1)

            for i in range(len(length)):
                pred_result = parse_decode(texts[i], char_preds[i], det_pred[i],
                                           length[i], self.tokenizer,
                                           self.max_seq_length)

                results.append(''.join(pred_result))
        return results


def get_predictor():
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    pinyin_vocab = Vocab.load_vocabulary(
        args.pinyin_vocab_file_path, unk_token='[UNK]', pad_token='[PAD]')
    predictor = Predictor(args.model_file, args.params_file, args.device,
                          args.max_seq_len, tokenizer, pinyin_vocab)
    return predictor


def get_samples_from_testset(path='../data/cn/SIGHAN15_test/sighan15_test.txt'):
    lines = [line for line in open(path, 'r')]
    samples, answers = [], []
    for line in lines:
        err, cor = line.rstrip().split('\t')[:2]
        samples.append(err)
        answers.append(cor)
    return samples, answers


def cd_write_sighan_result_to_file(args, predictor):
    # modified for another kind of output.
    with open(args.test_file, 'r', encoding='utf-8') as fin:
        with open(args.predict_file, 'w', encoding='utf-8') as fout:
            for i, line in enumerate(fin.readlines()):
                items = line.strip('\n').split('\t')
                err, cor = items[0:2]
                pred_result = predictor.predict([err], batch_size=args.batch_size)
                result = pred_result[0]
                fout.write(f"{result}\t{err}\t{cor}\n")


def evaluate_from_file(predict_file):
    # sentence level
    TP, FP, FN = 0, 1, 2
    det_case = [0, 0, 0]  # TP, FP, FN
    cor_case = [0, 0, 0]  # TP, FP, FN

    with open(predict_file, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin.readlines()):
            pred_result, err, cor = line.strip('\n').split('\t')
            pred_result = ()
            if pred_result == err:  # detect no errors
                if err == cor:  # truth is no errors
                    det_case[TP] += 1
                    cor_case[TP] += 1
                else:  # truth has errors
                    det_case[FN] += 1  # not recall
                    cor_case[FN] += 1
            else:  # detect errors
                if err == cor:  # truth is no errors
                    det_case[FP] += 1
                    cor_case[FP] += 1 
                else:  # truth has errors
                    det_case[TP] += 1
                    # pred_err_positions = [i for i, (_p, _t) in enumerate(zip(pred_result, cor)) if _p != _t]
                    # truth_err_positions = [i for i, (_e, _t) in enumerate(zip(err, cor)) if _e != _t]
                    # if pred_err_positions == truth_err_positions:
                    #     det_case[TP] += 1
                    # else:  
                    if pred_result == cor:
                        cor_case[TP] += 1
                    else:
                        cor_case[FN] += 1


def evaluate(srcs, preds, targets):
    from tqdm import tqdm
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    
    for src, tgt_pred, tgt in tqdm(zip(srcs, preds, targets)):
        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                # if verbose: print('right')
            # 预测为正
            else:
                FP += 1
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
            # 预测为负
            else:
                FN += 1
        
        total_num += 1

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return acc, precision, recall, f1


if __name__ == "__main__":
    predictor = get_predictor()
    samples, answers = get_samples_from_testset(args.test_file)

    """
    samples = [
        '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
        '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。',
    ]"""
    results = predictor.predict(samples, batch_size=args.batch_size)
    acc, precision, recall, f1 = evaluate(
        samples, results, answers)
    print(dict(acc=acc, precision=precision, recall=recall, f1=f1))
