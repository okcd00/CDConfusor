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

    
def same_one_hanzi_first_letter(pylist, pinyin):
    # [zhebian] => [zbian] + [zheb]
    first_letters = [(_py[0] + "[a-z]{0,5}") for _py in pylist]
    spec_pattern = []
    for _i, _fl in enumerate(first_letters):
        cur_py = [_py if _i != _j else _fl for _j, _py in enumerate(pylist)]
        spec_pattern.append('(' + ''.join(cur_py) + ')')
    spec_pattern = '|'.join(spec_pattern)
    # print(spec_pattern)
    return re.match(spec_pattern, pinyin)


if __name__ == "__main__":
    print(same_one_hanzi_first_letter(
        ['zhe', 'bian'], 'zheibian'))
