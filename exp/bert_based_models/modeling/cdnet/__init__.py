"""
@Time   :   2021-07-28 17:34:56
@File   :   __init__.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""

from .modeling_cdnet import BertForCsc as CDNET
from .modeling_cdmac import CDMacBertForCsc as CDMAC
from .modeling_cdpmb import CDPinyinMaskedBert as CDPMB
from .modeling_cdsmb import SoftMaskedBertModel as CDSMB


CSC_MODEL_CLASSES = {
    'cdmac': CDMAC,
    'cdnet': CDNET,
    'cdpmb': CDPMB,
    'cdsmb': CDSMB
}