import copy
import string
from string import ascii_lowercase


# keyboard distribution
# target = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
CONFUSOR_KEYBOARD_DATA = [
    '            ',
    ' qwertyuiop ',
    ' asdfghjkl  ',
    ' zxcvbnm    ',
    '            ',]

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
ins_rep_mat = {}  # keyboard-error
keyboard_distance = {}
offset = [  # six-direction distrub
    (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, 0), (1, 1)]

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

# ascii character to index
CHAT2IDX_MAPPING = {z: i for i, z in enumerate(list(string.ascii_lowercase))}
CHAT2IDX_MAPPING['0'] = 26


def char2idx(zimu):
    """
    '0' for the start of the sequence. Only applied in del_matrix.
    """
    return CHAT2IDX_MAPPING[zimu]


def generate_score_matrix(amb_score, inp_score):
    """
    Generate score matrices from pkl files.
    :param amb_score:
    :param inp_score:
    :return:
    """
    rep_score = amb_score  # all come from accent.
    def apply_mat(target_mat, mat_data, score):
        for firz, dellist in mat_data.items():
            for secz in dellist:
                i = char2idx(firz)
                j = char2idx(secz)
                target_mat[i][j] -= score
        return target_mat
    del_matrix = [[1 for _ in range(27)] for _ in range(27)]
    rep_matrix = copy.deepcopy(del_matrix)
    for i in range(27):
        for j in range(27):
            if i == j or i == 26 or j == 26:
                rep_matrix[i][j] = 0
    del_matrix = apply_mat(del_matrix, amb_del_mat, amb_score)
    rep_matrix = apply_mat(rep_matrix, amb_rep_mat, rep_score)
    rep_matrix = apply_mat(rep_matrix, ins_rep_mat, inp_score)
    return del_matrix, rep_matrix


def refined_edit_distance(str1, str2, del_matrix, rep_matrix, rate=False):
    """
    Given two sequences, return the refined edit distance normalized by the max length.
    """
    matrix = [[i + j for j in range(len(str2) + 1)] 
              for i in range(len(str1) + 1)]
    # here ins_matrix is the same as del_matrix
    # because we think the probability of the insert operation
    # is the same as the probability of the delete operation
    ins_matrix = del_matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            ind_i1 = char2idx(str1[i - 1])
            ind_j1 = char2idx(str2[j - 1])

            pstr1 = '0' if i == 1 else str1[i - 2]
            pstr2 = '0' if j == 1 else str2[j - 2]
            # delete a_i
            del_score = del_matrix[ind_i1][char2idx(pstr1)]

            # insert b_j after a_i
            ins_score = ins_matrix[ind_j1][char2idx(pstr2)]

            # replace a_i with b_j, the score equals to 0 if a_i == b_j
            rep_score = rep_matrix[ind_i1][ind_j1]

            matrix[i][j] = min(matrix[i - 1][j] + del_score, 
                               matrix[i][j - 1] + ins_score,
                               matrix[i - 1][j - 1] + rep_score)
            # return matrix
    if rate:
        return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])
    return matrix[len(str1)][len(str2)]


if __name__ == "__main__":
    del_matrix, rep_matrix = generate_score_matrix(
        amb_score=0.5,  # error with accents
        inp_score=0.5  # error with keyboard mistyping
    )
    print(
        refined_edit_distance(
            'hello', 'hello', 
            del_matrix, rep_matrix, rate=True))

