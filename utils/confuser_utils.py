import re
import copy
import numpy as np
from tqdm import tqdm
from string import ascii_lowercase

zimu2inds = {z: i for i, z in enumerate(list(ascii_lowercase))}
zimu2inds['0'] = 26


def zimu2ind(zimu):
    """
    '0' for the start of the sequence. Only applied in del_matrix.
    """
    return zimu2inds[zimu]


def edit_distance(str1, str2):
    """
    Given two sequences, return the edit distance normalized by the max length.
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, 
                               matrix[i][j - 1] + 1, 
                               matrix[i - 1][j - 1] + d)
            # return matrix
    return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])


def apply_edit_distance(target, corpus):
    """
    Given a target pinyin, and a pinyin list, return the sorted candidates.
    """
    cand_edit_dist = {}
    for cand_py in tqdm(corpus):
        cand_edit_dist[cand_py] = edit_distance(cand_py, target)
    sort_cand = sorted(cand_edit_dist.items(), key=lambda x:x[1])

    return sort_cand


def edit_distance_filtering(pinyin, pinyin_corpus, cand_num=10000):
    sort_cand = apply_edit_distance(pinyin, pinyin_corpus)
    return [p[0] for p in sort_cand[:cand_num]]


def bow_similarity(s1, s2):
    v1 = np.zeros(27)
    for c in s1:
        v1[zimu2ind(c)] += 1
    v2 = np.zeros(27)
    for c in s2:
        v2[zimu2ind(c)] += 1
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


def bow_similarity_filtering(pinyin, pinyin_corpus, cand_num=10000):
    """
    Given a pinyin and a pinyin list, return the filtered pinyin results.
    """
    # candpy2score = {p: bow_similarity(p, pinyin) for p in pinyin_corpus}
    candpy2score = {}
    for p in tqdm(pinyin_corpus):
        candpy2score[p] = bow_similarity(p, pinyin)
    sort_cand = sorted(candpy2score.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sort_cand[:cand_num]]


def bow_similarity_filtering_new(pinyin, pinyin_corpus, cand_num=10000):
    """
    Given a pinyin and a pinyin list, return the filtered pinyin results.
    """
    # candpy2score = {p: bow_similarity(p, pinyin) for p in pinyin_corpus}
    v1 = np.zeros(27)
    for c in pinyin:
        v1[zimu2ind(c)] += 1
    candpy2score = {}
    for p in tqdm(pinyin_corpus):
        v2 = np.zeros(27)
        for c in p:
            v2[zimu2ind(c)] += 1
        candpy2score[p] = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    sort_cand = sorted(candpy2score.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sort_cand[:cand_num]]


def refined_edit_distance(str1, str2, score_matrix, rate=False):
    """
    Given two sequences, return the refined edit distance normalized by the max length.
    """
    del_matrix, rep_matrix = score_matrix
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            ind_i1 = zimu2ind(str1[i - 1])
            ind_j1 = zimu2ind(str2[j - 1])
            rep_score = rep_matrix[ind_i1][ind_j1]
            pstr1 = '0' if i == 1 else str1[i - 2]
            pstr2 = '0' if j == 1 else str2[j - 2]
            # 删除a_i
            del_score = del_matrix[ind_i1][zimu2ind(pstr1)]

            # 在a后插入b_j
            ins_score = del_matrix[ind_j1][zimu2ind(pstr2)]

            matrix[i][j] = min(matrix[i - 1][j] + del_score, 
                               matrix[i][j - 1] + ins_score,
                               matrix[i - 1][j - 1] + rep_score)
            # return matrix
    if rate:
        return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])
    return matrix[len(str1)][len(str2)]


def cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm:
        return np.dot(v1, v2)/norm
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
