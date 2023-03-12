import copy
import string
from string import ascii_lowercase

# ascii character to index
zimu2inds = {z: i for i, z in enumerate(list(ascii_lowercase))}
zimu2inds['0'] = 26


def zimu2ind(zimu):
    """
    '0' for the start of the sequence. Only applied in del_matrix.
    """
    return zimu2inds[zimu]


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


def generate_score_matrix(amb_score, inp_data, inp_score):
    """
    Generate score matrices from pkl files.
    :param amb_data:
    :param amb_score:
    :param inp_data:
    :param inp_score:
    :return:
    """
    def apply_mat(target_mat, mat_data, score):
        for firz, dellist in mat_data.items():
            for secz in dellist:
                i = zimu2ind(firz)
                j = zimu2ind(secz)
                target_mat[i][j] -= score
        return target_mat
    del_matrix = [[1 for _ in range(27)] for _ in range(27)]
    rep_matrix = copy.deepcopy(del_matrix)
    for i in range(27):
        for j in range(27):
            if i == j or i == 26 or j == 26:
                rep_matrix[i][j] = 0
    del_matrix = apply_mat(del_matrix, amb_del_mat, amb_score)
    rep_matrix = apply_mat(rep_matrix, amb_rep_mat, amb_score)
    rep_matrix = apply_mat(rep_matrix, inp_data, inp_score)
    return del_matrix, rep_matrix


def refined_edit_distance(str1, str2, del_matrix, rep_matrix, rate=False):
    """
    Given two sequences, return the refined edit distance normalized by the max length.
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            ind_i1 = zimu2ind(str1[i - 1])
            ind_j1 = zimu2ind(str2[j - 1])
            rep_score = rep_matrix[ind_i1][ind_j1]

            pstr1 = '0' if i == 1 else str1[i - 2]
            pstr2 = '0' if j == 1 else str2[j - 2]
            # delete a_i
            del_score = del_matrix[ind_i1][zimu2ind(pstr1)]

            # insert b_j after a_i, ins/del share the same score matrix
            ins_score = del_matrix[ind_j1][zimu2ind(pstr2)]

            matrix[i][j] = min(matrix[i - 1][j] + del_score, 
                               matrix[i][j - 1] + ins_score,
                               matrix[i - 1][j - 1] + rep_score)
            # return matrix
    if rate:
        return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])
    return matrix[len(str1)][len(str2)]


if __name__ == "__main__":
    pass
