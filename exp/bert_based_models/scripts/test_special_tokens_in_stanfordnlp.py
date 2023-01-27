import os
from urllib.parse import quote
from stanfordcorenlp import StanfordCoreNLP
from bbcm.utils.text_utils import clean_text


SPECIAL_CHARS = """
、。·ˉˇ¨〃々—～‖…‘’“”〔〕〈〉《》「」『』〖〗【】±＋－×÷∧∨∑∏∪∩∈√⊥∥∠⌒⊙∫∮≡≌≈∽∝≠≮≯≤≥∞∶
∵∴∷♂♀°′″℃＄¤￠￡‰§№☆*〇○*◎**回□*△▽⊿▲▼▁▂▃▄▆**▉▊▋▌▍▎▏※→←↑↓↖↗↘↙**ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ①②③④⑤⑥⑦⑧⑨⑩
⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩
ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ！＂＃￥％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠
ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝
ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽ
まみむめもゃやゅゆょよらりるれろゎわゐゑをんァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノ
ハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψ
ω︵︶︹︺︿﹀︽︾﹁﹂﹃﹄︻︼︷︸АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüêɑńňǹɡ
ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ︱︳︴﹏﹋﹌─━│┃┄┅┆┇┈┉┊┋
┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋♁㊣㈱曱甴囍∟┅﹊﹍╭╮╰╯_^︵^﹕﹗/\'<>`,·。~～()-√$@*&#卐℡
ぁ〝〞ミ灬№＊ㄨ≮≯﹢﹣／∝≌∽≦≧≒﹤﹥じぷ┗┛￥￡§я-―‥…‰′″℅℉№℡∕∝∣═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╱╲╳▔▕〆〒〡〢〣〤〥〦〧〨〩
㎎㎏㎜㎝㎞㎡㏄㏎㏑㏒㏕ǹ
"""


# 加载模型
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
stanford_model = StanfordCoreNLP(r'/home/chendian/download/stanford-corenlp-4.2.2/', lang='zh', quiet=True)


texts = ["㋉7日", "10月7日", "100㎎", '100mg']
for sent_idx, text in enumerate(texts):
    text = clean_text(text)
    text_len = len(texts[sent_idx])
    r_dict = stanford_model._request('ner', text.replace("%", "％"))
    last_offset = r_dict['sentences'][-1]['tokens'][-1]['characterOffsetEnd']
    # if text_len != last_offset:
    print('->', texts[sent_idx])
    print(text_len, '!=', last_offset)
    print(r_dict['sentences'][-1]['tokens'])


from glob import glob
from tqdm import tqdm
from bbcm.utils.text_utils import clean_text

dir_path = '/data/chendian/clean_pretrain_data'
file_list = sorted(glob(dir_path + '/*.txt'))
bad_cases = []
for f_idx, file_path in tqdm(enumerate(file_list)):
    if f_idx < 980:
        continue
    texts = [clean_text(text) for text in open(file_path, 'r')]
    for sent_idx, text in enumerate(texts):
        text_len = len(texts[sent_idx])
        r_dict = stanford_model._request('ner', text.replace("%", "％"))
        last_offset = r_dict['sentences'][-1]['tokens'][-1]['characterOffsetEnd']
        if text_len != last_offset + 1:
            print('->', texts[sent_idx])
            print(f_idx, file_path, sent_idx)
            bad_cases.append((f_idx, sent_idx, f'not-matched: {text_len} and {last_offset+1}'))
            print(text_len, '!=', last_offset + 1)

