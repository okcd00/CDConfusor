"""
SPEC_FUNC
- Only used for 'special' pinyin sampling mode. 
- Receive a source pinyin list (e.g. ['que', 'shi']) and a target pinyin sequence (e.g. 'qushi'), 
  determine whether the pinyin sequence fits a pre-defined pattern. The pattern can be summarized by observing 
  real-world CSC data.
- Example. same_last_hanzi_first_letter: the pinyin sequence is the same as source pinyin list from the start to the
  first letter of the last hanzi.
"""

import re


def same_last_hanzi_first_letter(pylist, pinyin):
    spec_pattern = ''.join([*pylist[:-1], pylist[-1][0]])
    return re.match(spec_pattern, pinyin)
