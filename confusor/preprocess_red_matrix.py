# coding: utf-8
# ==========================================================================
#   Copyright (C) since 2023 All rights reserved.
#
#   filename : preprocess_red_embeddings.py
#   author   : chendian / okcd00@qq.com
#   date     : 2023-02-03
#   desc     : generate intermediate files
#              for boosting confusor
# ==========================================================================
import os
import copy
import string
import pickle
import numpy as np

# PATH (path to custom-files)
REPO_DIR = '/home/chendian/CDConfusor'
CONFUSOR_DATA_DIR = '/data/chendian/CDConfusor/'
SCORE_DATA_DIR = CONFUSOR_DATA_DIR + 'score_data/'  # for dumping score matrix

if not os.path.exists(SCORE_DATA_DIR):
    os.system(f"mkdir -p {SCORE_DATA_DIR}")

# for other functions' calling
CHAR_PY_VOCAB_PATH = f"{REPO_DIR}/data/vocab_pinyin.txt"

# deletion matrix
amb_del_mat = {
    'h': ['z', 'c', 's'],
    'g': ['n']}

# replacement matrix
amb_rep_mat = {
    'l': ['r', 'n'],
    'n': ['l'],
    'r': ['l'],
    'f': ['h'],
    'h': ['f']}

# insert/replacement matrix
ins_rep_mat = {}
keyboard_distance = {}
offset = [  # six-direction distrub
    (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, 0), (1, 1)]

# keyboard distribution
# target = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
CONFUSOR_KEYBOARD_DATA = [
    '            ',
    ' qwertyuiop ',
    ' asdfghjkl  ',
    '  zxcvbnm   ',
    '            ',]
for r, line in enumerate(CONFUSOR_KEYBOARD_DATA):
    for c, key in enumerate(line):
        if key == ' ':
            continue
        around = []
        keyboard_distance[(key, key)] = 0
        for r_off, c_off in offset:
            other_key = CONFUSOR_KEYBOARD_DATA[r+r_off][c+c_off]
            if other_key == ' ':
                continue
            keyboard_distance[(key, other_key)] = 1
            around.append(other_key)
        ins_rep_mat[key] = around
char_id = {z: i for i, z in enumerate(list(string.ascii_lowercase))}
char_id['0'] = 26


def char_to_id(zimu):
    """
    '0' for the start of the sequence. Only applied in del_matrix.
    """
    return char_id[zimu]


def apply_mat(target_mat, mat_data, score):
    for firz, del_list in mat_data.items():
        for secz in del_list:
            i = char_to_id(firz)
            j = char_to_id(secz)
            target_mat[i][j] -= score
    return target_mat


def generate_score_matrix(amb_data, amb_score, ins_rep_mat, ins_rep_score):
    del_matrix = [[1 for _ in range(27)] for _ in range(27)]
    rep_matrix = copy.deepcopy(del_matrix)
    for i in range(27):
        for j in range(27):
            if i == j or i == 26 or j == 26:
                rep_matrix[i][j] = 0
    del_matrix = apply_mat(del_matrix, amb_data['del_mat'], amb_score)
    rep_matrix = apply_mat(rep_matrix, amb_data['rep_mat'], amb_score)
    rep_matrix = apply_mat(rep_matrix, ins_rep_mat, ins_rep_score)
    return del_matrix, rep_matrix


def refined_edit_distance(str1, str2, score_matrix):
    """
    Given two sequences, return the refined edit distance normalized by the max length.
    """
    del_matrix, rep_matrix = score_matrix
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            ind_i1 = char_to_id(str1[i - 1])
            ind_j1 = char_to_id(str2[j - 1])
            rep_score = rep_matrix[ind_i1][ind_j1]
            pstr1 = '0' if i == 1 else str1[i - 2]
            pstr2 = '0' if j == 1 else str2[j - 2]
            # 删除a_i
            del_score = del_matrix[ind_i1][char_to_id(pstr1)]

            # 在a后插入b_j
            ins_score = del_matrix[ind_j1][char_to_id(pstr2)]

            matrix[i][j] = min(matrix[i - 1][j] + del_score, 
                               matrix[i][j - 1] + ins_score,
                               matrix[i - 1][j - 1] + rep_score)
            # return matrix
    return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0
    # return a value between -1 and 1
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm:
        return np.dot(v1, v2) / norm
    else:
        return 0


def complete_ziRED(newpy, ziREDscore, score_matrix):
    """
    If new pinyin appears, update the ziREDscore and save it to the default path.
    """
    print("new pinyin: {}, update ziREDscore".format(newpy))
    ziREDscore[newpy] = {}
    ziREDscore[newpy][newpy] = 0
    for oldpy in ziREDscore.keys():
        score = refined_edit_distance(newpy, oldpy, score_matrix)
        ziREDscore[newpy][oldpy] = score
        ziREDscore[oldpy][newpy] = score
    return ziREDscore


def complete_zi_sim_matrix(newpy, ziREDscore, score_matrix):
    """
    If new pinyin appears, update the zi_sim_matrix. If the new pinyin is not in ziREDscore, complete it either.
    """
    print("new pinyin: {}, update zi_sim_matrix".format(newpy))
    update_ziRED = False
    if newpy not in ziREDscore:
        ziREDscore = complete_ziRED(newpy, ziREDscore, score_matrix)
        update_ziRED = True
    new_zi_sim_matrix = {}
    for zipy in ziREDscore.keys():
        top_score = sorted(ziREDscore[zipy].items(), key=lambda x: x[1])
        new_zi_sim_matrix[zipy] = top_score
    ziREDscore = ziREDscore if update_ziRED else None
    return new_zi_sim_matrix, ziREDscore


def softmax(x):
    # Compute the softmax in a numerically stable way.
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def uniform_sample(cand_score_pair, num):
    """
    Given a list of pair (tar, score), sample to get some pairs by the uniform distribution from their scores.
    @param cand_score_pair: a list of (tar, score). The tar can be a pinyin sequence or a word token.
    @param num: the number of output pairs.
    @return: list of sampling pairs.
    """
    if len(cand_score_pair) < num:
        print("Warning: num={} is larger than the number of candidate pinyins.".format(num))
        num = len(cand_score_pair)
    scores = np.array([p[1] for p in cand_score_pair])
    normalized_score = softmax(scores)
    indices = np.random.choice(len(cand_score_pair), num, False, normalized_score)
    return [cand_score_pair[i] for i in indices]
    

def calculate_red_matrix(score_matrix, method='char'):
    # calculate red-score between char-level pinyins.
    if 'char' in method:
        char_pinyin_case = [line.strip() for line in open(f"{CHAR_PY_VOCAB_PATH}", "r") if not line.startswith('[')]
        char_red_matrix = {}
        for first_py in char_pinyin_case:
            for second_py in char_pinyin_case:
                char_red_matrix.setdefault(first_py, {})
                char_red_matrix[first_py][second_py] = refined_edit_distance(
                    first_py, second_py, score_matrix)
        return char_red_matrix

    if 'init' in method:
        initial_pinyin_case = [*string.ascii_lowercase, 'sh', 'ch', 'zh']
        initial_red_matrix = {}
        for first_py in initial_pinyin_case:
            for second_py in initial_pinyin_case:
                initial_red_matrix.setdefault(first_py, {})
                initial_red_matrix[first_py][second_py] = refined_edit_distance(
                    first_py, second_py, score_matrix)
        return initial_red_matrix
            
    raise ValueError(f"Invalid method: {method}, should be in [char|init]")


def main():
    amb_data = dict(
        del_mat=amb_del_mat, 
        rep_mat=amb_rep_mat)    
    score_matrix = generate_score_matrix(
        amb_data=amb_data, 
        amb_score=0.5, 
        ins_rep_mat=ins_rep_mat, 
        ins_rep_score=0.25)
    char_red_matrix = calculate_red_matrix(
        score_matrix, method='char')
    pickle.dump(amb_data, open(SCORE_DATA_DIR + 'amb_data.pkl', 'wb'))
    pickle.dump(ins_rep_mat, open(SCORE_DATA_DIR + 'inp_data.pkl', 'wb'))
    pickle.dump(char_red_matrix, open(SCORE_DATA_DIR + 'red_data.pkl', 'wb'))


if __name__ == "__main__":
    main()
    pass
