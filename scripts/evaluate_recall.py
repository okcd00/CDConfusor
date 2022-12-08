import sys
sys.path.append("../")
from utils.recall_evaluator import Evaluator

"""
CONFIGURATION   
"""
# configure the confusor
conf_args = {
'method': 'dcc',
'cos_threshold': (0.1, 0.5),
'cand_pinyin_num': 10,
# 'mode': 'sort',
'debug': True,
'cand_fzimu_num': 100,  # two-stage
'cand_dcc_num': 50,  # dcc
'cand_zi_num': 50,  # beam
'keep_num': 500,  # beam
}

# configure the evaluation variable parameters and details.
log_details_two_stage = {
    'name': 'two_stage',
    'info': 'evaluate time cost and recall of stage retrieval.'
}
params_two_stage = dict(cand_fzimu_num=[5, 10, 15, 25, 40, 50, 60, 75, 100, 125, 150, 200, 400, 600, 800, 1000, 1200])

log_details_dcc = {
    'name': 'dcc',
    'info': 'evaluate time cost and recall of DCC retrieval.'
}
params_dcc = dict(cand_dcc_num=[5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60])

log_details_beam = {
    'name': 'beam_search',
    'info': 'evaluate time cost and recall of beam-search retrieval.'
}
params_beam = dict(cand_zi_num=[5, 10, 15, 25, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300], keep_num=[500])


# start evaluations.
conf_args['method'] = 'dcc'
eval = Evaluator(conf_args)
eval(log_details_dcc, **params_dcc)

conf_args['method'] = 'beam'
eval = Evaluator(conf_args)
eval(log_details_beam, **params_beam)

conf_args['method'] = 'two-stage'
eval = Evaluator(conf_args)
eval(log_details_two_stage, **params_two_stage)


