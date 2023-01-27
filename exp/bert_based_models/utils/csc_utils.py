# coding: utf-8
# ==========================================================================
#   Copyright (C) 2022 All rights reserved.
#
#   filename : csc_utils.py
#   author   : chendian / okcd00@qq.com
#   date     : 2022-02-09
#   desc     : utils for CSC samples
# ==========================================================================

from tqdm import tqdm
from collections import defaultdict, Counter

def error_position_counts(_sample):
    """
    Tell how many wrong positions and their lengths in the sample.
    example: error_position_counts({'wrong_ids': [1,2,4,5,6,8]})
    outputs: (3, [2, 3, 1])
    """
    head = None  # substring head
    last = None  # last wrong id
    count = 0
    elc = []
    for wi in _sample['wrong_ids']:
        if last is not None and wi != last + 1:
            count += 1
            # print(last, head, last - head + 1)
            elc.append(last - head + 1)
            head = wi
        if head is None:
            head = wi
        last = wi
    else:
        if head is not None:
            # print(last, head, last - head + 1)
            elc.append(last - head + 1)
        if last is not None:
            count += 1
    return count, elc


def get_faulty_pair(_sample):
    """
    Tell the faulty pair from the sample.
    Only for samples with only one error position.
    """
    wrong_ids = _sample['wrong_ids']
    original_token = [_sample['original_text'][_wi] for _wi in wrong_ids]
    correct_token = [_sample['correct_text'][_wi] for _wi in wrong_ids]
    original_token, correct_token = ''.join(original_token), ''.join(correct_token)
    return original_token, correct_token


if __name__ == "__main__":
    pass
