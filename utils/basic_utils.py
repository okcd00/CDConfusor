import time, datetime
from bisect import bisect_left, bisect_right


def get_date_str():
    return '{}'.format(time.strftime("%y%m%d"))


def get_cur_time(delta=0):
    # C8 has no delta, C14 has 8 hours delta.
    cur_time = (datetime.datetime.now() +
                datetime.timedelta(hours=delta)).strftime("%m-%d %H:%M:%S")
    return cur_time


def range_to_list(tuple_str):
    return [int(x.strip()) for x in tuple_str.strip('()').split(',')]


def flatten(nested_list, unique=False):
    ret = [elem for sub_list in nested_list for elem in sub_list]
    if unique:
        return list(set(ret))
    return ret


def lower_bound(arr, x):
    """
    the index of the first number not smaller than x
    :param arr: [0, 2, 5, 8, 10]
    :param x: 2
    :return: 1
    """
    # binary_search_for_file_index
    i = bisect_left(arr, x)
    return i


def upper_bound(arr, x):
    """
    the index of the first number larger than x
    :param arr: [0, 2, 5, 8, 10]
    :param x: 2
    :return: 2
    """
    # binary_search_for_file_index
    i = bisect_right(arr, x)
    return i


def edit_distance(str1, str2, rate=True):
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
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
            # return matrix
    if rate:
        return matrix[len(str1)][len(str2)] / max([len(str1), len(str2)])
    return matrix[len(str1)][len(str2)]


def colored_text_html(text, color=None, method='background'):
    if color is None:
        return text
    if method == 'font':
        return u"<text style=color:{}>{}</text>".format(color, text)
    elif method == 'background':
        return u"<text style=background-color:{}>{}</text>".format(color, text)
    elif method == 'border':
        return u'<text style="border-style:solid;border-color:{}">{}</text>'.format(color, text)


def highlight_positions(text, positions, color='yellow', p=True, as_html=False):
    """
    将 text 中的 positions处 进行高亮显示

    如果想要把高亮的数据收集起来，进行处理之后再打印，参考 highlight_keyword

    :param text: 可能包含关键字的文本
    :param positions: 要高亮的位置。
      如果是 list 就用 color 染色；
      如果是 dict，{color: positions, ...} 覆盖 color 参数
    :param color:
    :param p: 是否直接打印出来
    :param as_html: True 直接渲染好 html 显示颜色。False 打印出 html 源码
    :type as_html: bool
    :return: html 字符串::

        <text style=color:black>10/0.4kV<text style=color:red>智能箱变</text></text>
    """
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print('no IPython')

    if not isinstance(positions, dict):
        positions = {color: positions}
    position_to_color = {}
    for c, ps in positions.items():
        for pos in ps:
            position_to_color[pos] = c
    html = u''
    for i, char in enumerate(text):
        if i in position_to_color:
            html += colored_text_html(char, position_to_color[i])
        else:
            html += char
    to_print = colored_text_html(html)
    if p:
        if as_html:
            print(to_print)
        else:
            display(HTML(to_print))
    return to_print


def display_predictions(model, samples, show_number=None, do_predict=False):
    if isinstance(show_number, (list, tuple)):
        show_number = show_number
    elif isinstance(show_number, int):
        show_number = (0, show_number)
    else:
        show_number = (0, len(samples))

    def _tokenize(_text):
        token_list = model.tokenizer.tokenize(_text)
        return [item.replace('##', '') for item in token_list]

    for sample_idx, sample in enumerate(samples):
        if sample_idx < show_number[0]:
            continue
        if sample_idx >= show_number[1]:
            break
        if do_predict:
            sample['predict_text'] = model.predict([sample['original_text']])[0]
        if sample['correct_text'].lower() == sample['predict_text'].lower() == sample['original_text'].lower():
            continue
        
        o_tokens = _tokenize(samples[sample_idx]['original_text'].lower())
        c_tokens = _tokenize(samples[sample_idx]['correct_text'].lower())        
        p_tokens = _tokenize(samples[sample_idx]['predict_text'].lower())

        # wrong_ids = sample['wrong_ids']
        wrong_ids = [i for i in range(len(o_tokens)) if o_tokens[i] != c_tokens[i]]
        
        print('sample index:', sample_idx, '|', sample['id'])
        highlight_positions(o_tokens, [])
        highlight_positions(c_tokens, wrong_ids, 'green')

        delta = [i for i in range(o_tokens.__len__()) 
                if o_tokens[i] != p_tokens[i]]
        highlight_positions(p_tokens, wrong_ids + delta, 'red')


if __name__ == '__main__':
    a = [0, 2, 5, 8, 10]
    for x in [0, 1, 2, 3]:
        print(lower_bound(a, x))  # 0, 1, 1, 2
        print(upper_bound(a, x))  # 1, 1, 2, 2
