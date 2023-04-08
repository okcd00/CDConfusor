from transformers import BertTokenizer
tokenizer = BertTokenizer('../vocab/vocab.txt')

from tqdm import tqdm
from pypinyin import lazy_pinyin
# err, cor, 111, pinyin_indexes


vocab = [line.strip() for line in open('../vocab/vocab.txt', 'r')]
pinyin_vocab = [line.strip() for line in open('../vocab/pinyin_vocab.txt', 'r')]


def B2Q(uchar):
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e: # 不是半角字符就返回原来的字符
        return uchar 
    if inside_code == 0x0020: # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code).upper()


def text_to_dcn_form(text, tokenizer):    
    tokens = tokenizer.tokenize(''.join([B2Q(c) for c in text]))
    return tokens


def pinyin_index(pinyin_str):
    try:
        res = pinyin_vocab.index(pinyin_str)
        return res
    except Exception as e:
        return 0  # [UNK]


src_path = '../../data/fin/findoc_augw.230408.tsv'
tgt_path = '../../data/fin/findoc_train.230408.dcn.txt'


same = 0
lines = 0


def find_last_occurrences(s, chars):
    res = []
    for c in chars:
        idx = s.rfind(c)
        if idx != -1:
            res.append(idx)
    return max(res)

    
# with open(tsv_path, 'w') as f2:
with open(tgt_path, 'w') as f:
    for idx, line in tqdm(enumerate(open(src_path, 'r'))):
        err, cor = line.strip().split('\t')
        err = ''.join([B2Q(c) for c in err.strip()])
        cor = ''.join([B2Q(c) for c in cor.strip()])
        tokens = tokenizer.tokenize(err)
        cor_tokens = tokenizer.tokenize(cor)
        try:
            err_mask = ''.join(['1' if o != c else '0' for o, c in zip(tokens, cor_tokens)])
            assert '11111' not in err_mask
            assert len(err) == len(cor)
            assert len(tokens) == len(cor_tokens)
        except Exception as e:
            print(idx, e)
            print(err)
            print(cor)
            print(len(tokens) == len(cor_tokens))
            print(list(zip(tokens, cor_tokens)))
            continue
        len_indexes = [str(1) for _ in range(len(tokens))]
        py_indexes = [str(pinyin_index(_py)) for _py in lazy_pinyin(tokens)]
        faulty_indexes = [i for i, (e, c) in enumerate(zip(tokens, cor_tokens)) if e != c]
        
        last_index = 0  # cut short for longer sentences.
        if len(faulty_indexes) > 0 and faulty_indexes[0] > 192:
            try:
                last_index = find_last_occurrences(
                    ''.join(tokens[:faulty_indexes[0]]), 
                    ['，', '。', '！', '；'])
            except Exception as e:
                last_index = max(0, faulty_indexes[0] - 20)
            tokens = tokens[last_index:]
            cor_tokens = cor_tokens[last_index:]
            len_index = len_index[last_index:]
            py_indexes = py_indexes[last_index:]
        if len(tokens) == len(cor_tokens) == len(len_indexes) == len(py_indexes):
            res = f"{' '.join(tokens)}\t{' '.join(cor_tokens)}\t{' '.join(len_indexes)}\t{' '.join(py_indexes)}\n"
            f.write(res)
            if tokens == cor_tokens:
                same += 1
            lines += 1
        if '.mp' in tgt_path and (tokens != cor_tokens):  # more positive samples.
            len_indexes = ' '.join([str(1) for _ in range(len(cor_tokens))])
            py_indexes = [str(pinyin_index(_py)) for _py in lazy_pinyin(cor_tokens)]
            if len(tokens) == len(cor_tokens) == len(py_indexes):
                res_positive = f"{' '.join(tokens)}\t{' '.join(cor_tokens)}\t{' '.join(len_indexes)}\t{' '.join(py_indexes)}\n"
                f.write(res_positive)
                same += 1
                lines += 1


print(same, lines, same/lines)