# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2022 All rights reserved.
#
#   filename : evaluate.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-01-20
#   desc     : scripts for evaluating on DCN.
# ==========================================================================

import sys
from pprint import pprint


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

from six import iteritems


def colored_text_html(text, color='white'):
    # in dark-theme jupyter, we need the default color is white
    return u"<text style=color:{}>{}</text>".format(color, text)


def highlight_positions(text, positions, color='red', p=True, as_html=False, default_color=None):
    """
    :param text: 
    :param positions: 
      for lists: use `color: 'red'`；
      for dicts, use `{color: positions, ...}`
    :param color:
    :param p: print it out or not
    :param as_html: True outputs html with color, False outputs html src.
    :type as_html: bool
    :return: html strings::

        <text style=color:black>10/0.4kV<text style=color:red>智能箱变</text></text>
    """
    if not isinstance(positions, dict):
        positions = {color: positions}
    position_to_color = {}
    for c, ps in iteritems(positions):
        for p in ps:
            position_to_color[p] = c
    html = u''
    for i, char in enumerate(text):
        if i in position_to_color:
            html += colored_text_html(char, position_to_color[i])
        elif default_color:
            html += colored_text_html(char, default_color)
        else:
            html += char
    to_print = colored_text_html(html, color='black')
    if p:
        if as_html:
            print(to_print)
        else:
            from IPython.core.display import display, HTML
            display(HTML(to_print))
    return to_print


def show_diff(err, cor):
    assert len(err) == len(cor)
    faulty_indexes = [i for i in range(len(err)) if err[i] != cor[i]]
    highlight_positions(err, faulty_indexes, 'red', default_color='white')
    highlight_positions(cor, faulty_indexes, 'green', default_color='white')


def compute_corrector_prf_faspell(results, logger=None, strict=True, in_jupyter=False):
    """
    All-in-one measure function for end-to-end correction by @okcd00.
    re-arrange from FASpell's measure script. (re-name the vars for readablity.)
    add some more metrics for outputs.

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

    for idx, item in enumerate(results):
        if isinstance(item, dict):
            wrong, correct, predict = item['original_text'], item['correct_text'], item['predict_text']
        else:
            # wrong, correct, predict, d_tgt, d_predict = item
            wrong, correct, predict = item[:3]
        if idx == 0:
            print("Evaluating. Sample 1 looks like:")
            if in_jupyter:
                show_diff(wrong, correct)
            else:
                print(f"{wrong}\n{correct}")
            print(f"model outputs:\n{predict}")

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
    for phase in ['cor_sent']:  # 'det_char', 'cor_char', 'det_sent', 
        dic = report_prf(
            *counts[f'{phase}_counts'], 
            phase=phase, logger=logger,
            return_dict=True)
        details.update(dic)
    details.update(counts)
    del counts
    return details


def B2Q(uchar):
    if len(uchar) > 1:
        return uchar.upper()
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e: # 不是半角字符就返回原来的字符
        return uchar 
    if inside_code == 0x0020: # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code).upper()


def evaluate(dcn_path, pred_path):
    dcn_items = [line.strip().split('\t')[:2] for line in open(dcn_path, 'r')]
    results = [line.strip() for line in open(pred_path, 'r')]
    try:
        assert len(dcn_items) == len(results)
    except Exception as e:
        print(e)
        print(len(dcn_items), len(results))
    err_sents, cor_sents = list(zip(*dcn_items))
    err_sents = [''.join(
        [B2Q(c[2:].upper() if c.startswith('##') else c.upper()) 
         for c in sent.strip().split()]) for sent in err_sents]
    cor_sents = [''.join(
        [B2Q(c[2:].upper() if c.startswith('##') else c.upper()) 
         for c in sent.strip().split()]) for sent in cor_sents]
    metrics = compute_corrector_prf_faspell(list(zip(err_sents, cor_sents, results)))
    p, r, f, acc = metrics['cor_sent_p'], metrics['cor_sent_r'], metrics['cor_sent_f1'], metrics['cor_sent_acc']
    p, r, f, acc = list(map(lambda x: round(x*100, 2), [p, r, f, acc]))
    print(f"{p}/{r}/{f} | {acc}")
    return metrics


if __name__ == "__main__":
    # output = os.popen(f'python script/eval_spell_for_training_sent.py {result_path} {input_path}')
    print(sys.argv)
    dcn_path = sys.argv[1]
    pred_path = sys.argv[2]
    evaluate(dcn_path, pred_path)
