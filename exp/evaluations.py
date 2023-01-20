"""
@Time   :   2021-01-21 12:01:32
@File   :   evaluations.py
@Author :   okcd00, Abtion
@Email  :   okcd00@qq.com, abtion{at}outlook.com
"""
import os
import json
import numpy as np
from tqdm import tqdm


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    "./pretrained_models/chinese-roberta-wwm-ext/")


def report_prf(tp, fp, fn, phase, logger=None, return_dict=False):
    # For the detection Precision, Recall and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    if phase and logger:
        logger.info(f"The {phase} result is: "
                    f"{precision:.4f}/{recall:.4f}/{f1_score:.4f} -->\n"
                    # f"precision={precision:.6f}, recall={recall:.6f} and F1={f1_score:.6f}\n"
                    f"support: TP={tp}, FP={fp}, FN={fn}")
    if return_dict:
        ret_dict = {
            f'{phase}_p': precision,
            f'{phase}_r': recall,
            f'{phase}_f1': f1_score}
        return ret_dict
    return precision, recall, f1_score


def compute_detector_prf_as_faspell(results, logger=None, strict=True):
    # calculate based on detector's prob (individually)

    all_sent = 0  # sent count
    accurate_detected_sent = 0  # correctly prediction on sent-level

    TP, FP, FN = 0, 0, 0  # global
    TP_sent, FP_sent, FN_sent = 0, 0, 0  # global sentence-level

    for item in results:
        all_sent += 1
        _, _, _, d_tgt, d_predict = item
        
        if d_tgt == d_predict:
            accurate_detected_sent += 1

        no_error_sent_flag = True
        has_p1 = False  # model_think_this_sent_is_wrong
        has_faulty_p = False
        for _i, (t, p) in enumerate(zip(d_tgt, d_predict)):
            if t == p:  # correct
                if t == p == 1:
                    TP += 1
                    has_p1 = True
                    no_error_sent_flag = False
                # we don't care the TN (t == p == 0)
            else:
                if t == 1:  # p == 0
                    FN += 1  # not recall
                    has_faulty_p = True
                    no_error_sent_flag = False
                else:  # t == 0 and p == 1
                    FP += 1  # faulty precision
                    has_faulty_p = True
                    has_p1 = True
        else:
            if not strict:  # we need to point out it HAS errors.
                if no_error_sent_flag:  # this sentence is correct
                    if has_p1:
                        # model thinks it has errors (faulty prediction)
                        FP_sent += 1  
                    else:
                        # model thinks it is correct
                        TP_sent += 1  
                else:  # this sentence has errors.
                    if has_p1:  
                        # model thinks it has errors
                        TP_sent += 1  
                    else:
                        # model thinks it is correct (not recalled)
                        FN_sent += 1  
            else:  # we need to point out EVERY fault position.
                if no_error_sent_flag:  # this sentence is correct
                    if has_p1:
                        # model thinks it has errors (faulty prediction)
                        FP_sent += 1  
                    else:
                        # model thinks it is correct
                        TP_sent += 1  
                else:  # this sentence has errors.
                    if has_faulty_p:
                        # model failed to perfectly catch all errors
                        FN_sent += 1 
                    else:  # not even one position in prediction is wrong
                        # model correctly predict all faulty positions
                        TP_sent += 1

    counts = {  # TP, FP, TN for each level
        'det_char_counts': [TP, FP, FN],
        'det_sent_counts': [TP_sent, FP_sent, FN_sent],
        'det_sent_acc': accurate_detected_sent / all_sent,
        'cor_sent_acc': TP / all_sent,
        'all_sent_count': all_sent,
    }

    details = {}
    for phase in ['det_char', 'det_sent']:
        dic = report_prf(
            *counts[f'{phase}_counts'], 
            phase=phase, logger=logger,
            return_dict=True)
        details.update(dic)
    details.update(counts)
    return details


def compute_corrector_prf_faspell(results, logger=None, strict=True):
    """
    All-in-one measure function for end-to-end correction by @okcd00.
    re-arrange from FASpell's measure script. (re-name the vars for readablity.)
    :param results: a list of (wrong, correct, predict[, ...])
    both token_ids or characters are fine for the script.
    :param logger: take which logger to print logs.
    :param strict: a more strict evaluation mode (all-char-detected/corrected)

    References:
        sentence-level PRF: https://github.com/iqiyi/FASPell/blob/master/faspell.py
    """

    det_char_TP = 0
    det_sent_TP = 0
    char_pred, char_truth = 0, 0

    cor_char_TP = 0
    cor_sent_TP = 0
    sent_pred, sent_truth = 0, 0
    
    all_sent = 0
    det_sent_acc = 0
    cor_sent_acc = 0

    for item in results:
        if isinstance(item, dict):
            wrong, correct, predict = item['original_text'], item['correct_text'], item['predict_text']
        else:
            # wrong, correct, predict, d_tgt, d_predict = item
            wrong, correct, predict = item[:3]

        all_sent += 1

        _char_pred = 0
        _char_truth = 0
        _det_char_FP = 0
        _det_char_hit = 0

        for _i, (c, w, p) in enumerate(zip(correct, wrong, predict)):
            if c != p:  # prediction error : FP
                _det_char_FP += 1
            if w != p:  # predict is different from the origin word
                _char_pred += 1
                if c == p:  # correction hit: p == c and p != w : TP
                    cor_char_TP += 1
                if w != c:  # detection hit: w != p and w != c : TP
                    det_char_TP += 1
                    _det_char_hit += 1
            if c != w:  # a position needs to be recalled
                # char-level not recalled = _det_char_truth - hit_char : FN
                _char_truth += 1

        char_pred += _char_pred
        char_truth += _char_truth

        # it is a sentence with faulty wordings.
        if _char_truth != 0:
            sent_truth += 1

        # the model think the santence is faulty
        if _char_pred != 0:
            sent_pred += 1

            # if there's no precision errors
            if _det_char_FP == 0:
                cor_sent_TP += 1  # sentence-level TP + 1

        # if a sentence has faulty words.
        if strict:  # find out all faulty wordings' potisions
            # all positions detected (recall) 
            # & all predictions are correct (precision)
            true_detected_flag = (
                _char_truth != 0 \
                and _det_char_hit == _char_truth \
                and _char_pred == _det_char_hit)
        else:  
            # the model thinks the sentence has faulty wordings
            true_detected_flag = (
                _char_pred != 0 and _char_truth != 0)

        # if a sentence has some [or no] errors.
        if true_detected_flag:  # or (correct == wrong == predict):
            det_sent_TP += 1

        if correct == predict:
            cor_sent_acc += 1
            
        if correct == predict or true_detected_flag:
            det_sent_acc += 1  # consider about the correct original sentence.

    counts = {  # TP, FP, FN for each level
        'det_char_counts': [det_char_TP,
                            char_pred-det_char_TP, 
                            char_truth-det_char_TP],
        'cor_char_counts': [cor_char_TP, 
                            char_pred-cor_char_TP, 
                            char_truth-cor_char_TP],
        'det_sent_counts': [det_sent_TP, 
                            sent_pred-det_sent_TP, 
                            sent_truth-det_sent_TP],
        'cor_sent_counts': [cor_sent_TP, 
                            sent_pred-cor_sent_TP, 
                            sent_truth-cor_sent_TP],
        'det_sent_acc': det_sent_acc / all_sent,
        'cor_sent_acc': cor_sent_acc / all_sent,
        'all_sent_count': all_sent,
    }

    details = {}
    for phase in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
        dic = report_prf(
            *counts[f'{phase}_counts'], 
            phase=phase, logger=logger,
            return_dict=True)
        details.update(dic)
    details.update(counts)
    del counts
    return details


def compute_corrector_prf_pnet(results, logger, on_detected=True):
    """
    References:
        character-level PRF: https://github.com/sunnyqiny/
        Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py

    maybe-to-do: implements sentence-level measures for it.
    """

    all_gold_index = []
    all_predict_index = []
    all_predict_true_index = []

    TP, FP, FN = 0, 0, 0
    SENT_TP, SENT_FP, SENT_FN = 0, 0, 0
    for item in results:
        src, tgt, predict, d_tgt, d_predict = item
        src_len = len(list(src))

        # different tokens' indexes between source and target
        gold_index = [i for i in range(src_len) if src[i] != tgt[i]]
        all_gold_index.append(gold_index)

        # different tokens' indexes between source and predict
        predict_index = [i for i in range(src_len) if src[i] != predict[i]]
        all_predict_index.append(predict_index)

        if gold_index == predict_index:
            SENT_TP += 1
        else:
            SENT_FP += 1

        each_true_index = []
        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    dp, dr, detection_f1 = report_prf(TP, FP, FN,
                                      'detection', logger=logger)

    det_counts = {'det_counts': [TP, FP, FN]}        
    
    # store FN counts
    n_misreported = int(FN)

    # <Pointer-Networks work>: 
    # we only detect those correctly detected location, 
    # which is a different from the common metrics since
    # we wanna to see the precision improve by using the confusion set
    TP, FP, FN = 0, 0, 0
    for i in range(len(all_predict_true_index)):
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    cp, cr, correction_f1 = report_prf(TP, FP, FN,
                                       'correction', logger=logger)

    # common metrics to compare with other baseline methods.
    ccp, ccr, correction_cf1 = report_prf(TP, FP, FN + n_misreported,
                                          'correction_common', logger=logger)
    if not on_detected:
        correction_f1 = correction_cf1

    cor_counts = {'cor_counts': [TP, FP, FN]}
    
    details = {
        'det_p': dp,
        'det_r': dr,
        'det_f1': detection_f1,
        'cor_p': cp,
        'cor_r': cr,
        'cor_f1': correction_f1,
        'common_cor_p': ccp,
        'common_cor_r': ccr,
        'common_cor_f1': correction_cf1,
    }

    details.update(det_counts)
    details.update(cor_counts)
    return detection_f1, correction_f1, details


# detection-only prf
compute_detector_prf = compute_detector_prf_as_faspell

# overall correction prf
# compute_corrector_prf = compute_corrector_prf_pnet
compute_corrector_prf = compute_corrector_prf_faspell


def compute_sentence_level_prf(results, logger):
    """
    deprecated: original sentence-level measure function from bbcm 
    自定义的句级prf，设定需要纠错为正样本，无需纠错为负样本
    :param results:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = len(results)

    for item in results:
        if isinstance(item, dict):  # in case of results json
            src, tgt, predict = item['original_text'], item['correct_text'], item['predict_text']
        else:
            src, tgt, predict = item[:3]

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == predict:
                TN += 1
            # 预测为正
            else:
                FP += 1
        # 正样本
        else:
            # 预测也为正
            if tgt == predict:
                TP += 1
            # 预测为负
            else:
                FN += 1

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    logger.info(f'Sentence Level: acc:{acc:.6f}, \n'
                f'precision:{precision:.6f}, recall:{recall:.6f}, f1:{f1:.6f}')
    return acc, precision, recall, f1


def eval_on_txt_list(texts, correct_fn, verbose, dump_path):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    write_lines = []
    
    for line in tqdm(texts):
        line = line.strip()
        if line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        src = parts[0]
        tgt = parts[1]

        result = correct_fn(src)
        if len(result) == 2:
            tgt_pred, pred_detail = result
        else:
            tgt_pred, pred_detail = result, None
        wrong_flag = False

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                # if verbose: print('right')
            # 预测为正
            else:
                FP += 1
                if verbose: 
                    wrong_flag = True
                    print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                # if verbose: print('right')
            # 预测为负
            else:
                FN += 1
                if verbose: 
                    wrong_flag = True
                    print('wrong')
        
        total_num += 1
        if verbose:  # output on stdout
            if wrong_flag:
                print()
                print('input  :', src)
                print('truth  :', tgt)
                print('predict:', tgt_pred, pred_detail)
        
        if dump_path is not None:  # output in file
            if wrong_flag:
                write_lines.extend([
                    f"O: {src}",
                    f"T: {tgt}", 
                    f"P: {tgt_pred}", ""])

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return acc, precision, recall, f1, write_lines


def eval_on_txt_file(sighan_path, correct_fn, verbose, dump_path):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    write_lines = []
    with open(sighan_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            src = parts[0]
            tgt = parts[1]

            result = correct_fn(src)
            if len(result) == 2:
                tgt_pred, pred_detail = result
            else:
                tgt_pred, pred_detail = result, None
            wrong_flag = False

            # 负样本
            if src == tgt:
                # 预测也为负
                if tgt == tgt_pred:
                    TN += 1
                    # if verbose: print('right')
                # 预测为正
                else:
                    FP += 1
                    if verbose: 
                        wrong_flag = True
                        print('wrong')
            # 正样本
            else:
                # 预测也为正
                if tgt == tgt_pred:
                    TP += 1
                    # if verbose: print('right')
                # 预测为负
                else:
                    FN += 1
                    if verbose: 
                        wrong_flag = True
                        print('wrong')
            
            total_num += 1
            if verbose:  # output on stdout
                if wrong_flag:
                    print()
                    print('input  :', src)
                    print('truth  :', tgt)
                    print('predict:', tgt_pred, pred_detail)
            
            if dump_path is not None:  # output in file
                if wrong_flag:
                    write_lines.extend([
                        f"O: {src}",
                        f"T: {tgt}", 
                        f"P: {tgt_pred}", ""])

        acc = (TP + TN) / total_num
        precision = TP / (TP + FP) if TP > 0 else 0.0
        recall = TP / (TP + FN) if TP > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return acc, precision, recall, f1, write_lines


def eval_on_json_file(sighan_path, correct_fn, verbose, dump_path, with_gold_detection=True):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    write_lines = []
    
    from bbcm.data.loaders.collator import DataCollatorForCsc
    evaluation_collator = DataCollatorForCsc(tokenizer=tokenizer)

    ec = evaluation_collator
    samples = json.load(open(sighan_path, 'r'))
    for sample in tqdm(samples):
        src = sample['original_text']
        tgt = sample['correct_text']
        det_labels = None
        if with_gold_detection:
            det_labels = ec.generate_det_labels(
                ec.get_encoded_texts(src)['input_ids'], 
                ec.get_encoded_texts(tgt)['input_ids'])

        result = correct_fn(src, det_mask=det_labels)
        if len(result) == 2:
            tgt_pred, pred_detail = result
        else:
            tgt_pred, pred_detail = result, None
        wrong_flag = False

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                # if verbose: print('right')
            # 预测为正
            else:
                FP += 1
                if verbose: 
                    wrong_flag = True
                    print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                # if verbose: print('right')
            # 预测为负
            else:
                FN += 1
                if verbose: 
                    wrong_flag = True
                    print('wrong')
        
        total_num += 1
        if verbose:  # output on stdout
            if wrong_flag:
                print()
                print('input  :', src)
                print('truth  :', tgt)
                print('predict:', tgt_pred, pred_detail)
        
        if dump_path is not None:  # output in file
            if wrong_flag:
                write_lines.extend([
                    f"O: {src}",
                    f"T: {tgt}", 
                    f"P: {tgt_pred}", ""])

    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return acc, precision, recall, f1, write_lines


def eval_sighan2015_by_model(correct_fn, sighan_path=None, verbose=False, dump_path=None):
    """
    SIGHAN句级评估结果，设定"需要纠错"为正样本，"无需纠错"为负样本
    reference: shibing624/pycorrector:/utils/eval.py#L252
    Args:
        correct_fn:
        input_eval_path:
        output_eval_path:
        verbose:
    Returns:
        Acc, Recall, F1
    """
    import time
    start_time = time.time()

    if sighan_path is None:
        sighan_path = './datasets/sighan15_test.txt'

    if sighan_path.endswith('.txt'):
        acc, precision, recall, f1, write_lines = eval_on_txt_file(
            sighan_path=sighan_path, correct_fn=correct_fn,
            verbose=verbose, dump_path=dump_path)
    elif sighan_path.endswith('.json'):
        acc, precision, recall, f1, write_lines = eval_on_json_file(
            sighan_path=sighan_path, correct_fn=correct_fn,
            verbose=verbose, dump_path=dump_path)
    else:
        raise ValueError("Invalid sighan path {}".format(sighan_path))
    
    spend_time = time.time() - start_time
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, cost time:{spend_time:.2f} s')
        
    if dump_path:
        print(f"Saving predictions in {dump_path}")
        with open(dump_path, 'w') as f:
            for line in write_lines:
                f.write(f"{line.strip()}\n")

    return acc, precision, recall, f1


if __name__ == "__main__":
    # wrong, correct, predict
    items = [["您好", "你好", "你好"] for _ in range(10)] + \
        [["你好", "你好", "你好"] for _ in range(10)] + [["您好", "你好", "您好"]]
    from pprint import pprint
    pprint(compute_corrector_prf_faspell(items))
    # eval_sighan2015_by_model(lambda x: x)
