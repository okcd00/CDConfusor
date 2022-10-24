# coding: utf-8
# ==========================================================================
#   Copyright (C) 2020 All rights reserved.
#
#   filename : custom_encodings.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-08-05
#   desc     : text in processing must be unicode, mainly for json. (my own implemention)
# ==========================================================================
import os
from six import PY2, PY3


try:
    # if you have ujson, it will be faster
    # but the calling method is different.
    import ujson as json
    JSON_MODULE = 'ujson'
except ImportError:
    import json
    JSON_MODULE = 'json'


    class JsonBytesEncoder(json.JSONEncoder):
        # json.dumps
        def default(self, obj):
            # if isinstance(obj, np.ndarray):
            #     return obj.tolist()  # for further support.
            if isinstance(obj, bytes):
                return convert_to_unicode(obj)
                # return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)


def json_dumps(obj_, encrypt=False):
    if JSON_MODULE == 'json':
        _json_str = json.dumps(
            obj_, cls=JsonBytesEncoder)
    elif JSON_MODULE == 'ujson':
        if int(json.__version__[0]) < 2:
            # standard ujson-1.35 for python2.7
            _json_str = json.dumps(obj_)
        else:  # standard ujson-3.0.0 for python3.6
            _json_str = json.dumps(
                obj_, reject_bytes=False)
    else:
        _json_str = json.dumps(obj_)
    if encrypt:
        return zlib_compress(_json_str)
    return _json_str


def json_dump(obj_, path=None, mode='w', stream=None, encrypt=False):
    # the same as json.dump(zlib_encrypt(obj_), open(path, 'w'))
    # use 'w', not 'wb' in python3 for
    # TypeError: a bytes-like object is required, not 'str'
    if encrypt:  # the zlib.compress transfers data into bytes
        mode = 'wb'
    if stream is not None:
        # stream contains path and mode
        stream.write(json_dumps(obj_, encrypt))
    else:
        with open(path, mode) as f:
            f.write(json_dumps(obj_, encrypt))


def json_loads(str_, decrypt=False):
    if decrypt:
        str_ = zlib_decompress(str_)
    # all kinds of json have the same loads()
    data = json.loads(str_)
    return data


def json_load(path, mode='r', decrypt=False):
    # the same as json.load(open(path, mode))
    if decrypt:  # the zlib.compress transfers data into bytes
        mode = 'rb'
    with open(path, mode) as f:
        obj_ = json_loads(f.read(), decrypt)
    return obj_


def zlib_compress(str_):
    """ return an encrypted string """
    import zlib
    # to unicode (py2-unicode or py3-str)
    j_str = convert_to_unicode(str_)
    # zlib only allow bytes-like inputs
    return zlib.compress(convert_to_bytes(j_str))


def zlib_decompress(str_):
    """ return a json_str in unicode """
    import zlib
    b_str = zlib.decompress(str_)
    return convert_to_unicode(b_str)


def path_join(*args):
    parts = [convert_to_unicode(each) for each in args]
    return os.path.join(*parts)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, (int, float)):
        text = '{}'.format(text)
    if PY3:
        if isinstance(text, str):  # py3-str is unicode
            return text
        elif isinstance(text, bytes):  # py3-bytes is py2-str
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif PY2:
        if isinstance(text, str):  # py2-str is py3-bytes
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):  # py2-unicode is py3-str
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_bytes(text):
    if PY2 and isinstance(text, str):
        return text
    elif PY3 and isinstance(text, bytes):
        return text
    u_text = convert_to_unicode(text)
    return u_text.encode('utf-8')


def recursive_encoding_unification(cur_node):
    from collections import OrderedDict
    reu = recursive_encoding_unification

    if isinstance(cur_node, (list, tuple)):
        return type(cur_node)(
            [reu(item) for item in cur_node])
    elif isinstance(cur_node, (dict, OrderedDict)):
        return type(cur_node)(
            [(reu(k), reu(v)) for (k, v) in cur_node.items()])
    elif isinstance(cur_node, (int, float)):
        return cur_node
    elif cur_node is None:
        return None
    else:  # str, bytes, unicode
        # only convert leaf-nodes
        return convert_to_unicode(cur_node)


def json_unicode(json_dict):
    return recursive_encoding_unification(json_dict)
