import sys
sys.path.append("../../../")

from bbcm.data.loaders.confusor import Confusor
from itertools import product
from tqdm import tqdm
import json

DATAPATH = '/data/pangchaoxu/tencent_embedding/test_data/recall_test_data_1000.json'
LOGPATH = '/home/pangchaoxu/confusionset/logs/pinyin_retrieval_recall/'


class Evaluator(object):
    def __init__(self, confusor_args, evaluate_num=200, topk=5, weights=None):
        """
        Evaluate the recall and time cost of different pinyin sequence retrieval methods.
        @param confusor_args: [Dict] Arguments of the evaluated confusor.
        @param evaluate_num: [int] The number of pinyin sequences to retain for the evaluation.
        @param topk: [int] NCG parameters. Compute the NCG of top-k scores
        @param weights: [List] Weights of top-down scores. The length should be topk.
        """
        self.confusor_args = confusor_args
        self.conf = Confusor(**confusor_args)
        self.evaluate_num = evaluate_num
        self.topk = topk
        self.weights = weights
        self.recall_test_data = self.load_test_data()


    def load_test_data(self):
        return json.load(open(DATAPATH, 'r'))

    def normalized_cumulative_gain(self, res_retrieval, res_fact, topk, weights):
        """
        Compute NCG.
        """
        def generate_score_len_dict(res):
            score2pyseqs = {}
            for pyseq, score in res:
                score2pyseqs.setdefault(score, [])
                score2pyseqs[score].append(pyseq)
            score2len = {score: len(pys) for score, pys in score2pyseqs.items()}
            return score2len
        sco2len_fact = generate_score_len_dict(res_fact)
        sco2len_retr = generate_score_len_dict(res_retrieval)
        if topk > len(sco2len_fact):
            print("warning: topk = {} is large than the number of scores in facts. Use topk = {} instead." \
                  .format(topk, len(sco2len_fact)))
            topk = len(sco2len_fact)
        weights = weights or [-i for i in range(-topk, 0)]
        if len(weights) != topk:
            raise ValueError("invalid weights, the length must be topk.")
        target_scores = [score for score, _ in sorted(sco2len_fact.items(), key=lambda x: x[0])[:topk]]
        cum_retr = cum_fact = 0
        for score, weight in zip(target_scores, weights):
            cum_retr += weight * sco2len_retr.get(score, 0)
            cum_fact += weight * sco2len_fact[score]
        return cum_retr / cum_fact

    def evaluate(self, evaluate_num, topk, weights, **var_params):
        """
        Evaluate the recall and time cost of the confusor.
        @param params: a dict of variable parameters.
        @return:{'param1_param2': {tok_len:{'time_cost':[tpoint], 'NCG': [NCG]}}}
        """
        res = {}
        param_name = list(var_params.keys())
        param_list = list(var_params.values())
        for param in product(*param_list):
            param_key = '_'.join([str(p) for p in param])
            param_dict = {param_name[i]: param[i] for i in range(len(param_name))}
            print("Method: {}. Params: {}.".format(self.confusor_args['method'], param_dict))
            res.setdefault(param_key, {})
            for word, targets in tqdm(self.recall_test_data):
                tok_len = len(word)
                res[param_key].setdefault(tok_len, {})
                res[param_key][tok_len].setdefault('time_cost', [])
                res[param_key][tok_len].setdefault('NCG', [])
                timer, pinyins = self.conf.pinyin_retrieval_recall_evaluator(word, evaluate_num=evaluate_num, **param_dict)
                tpoint = [tpoint for info, tpoint in timer if info == 'pinyin retrieval'][0]
                time_cost = tpoint - timer[0][1]
                res[param_key][tok_len]['time_cost'].append(time_cost)

                # recall = len(set(targets[:evaluate_num]) & set(pinyins)) / evaluate_num
                ncg = self.normalized_cumulative_gain(pinyins, targets[:evaluate_num], topk, weights)
                if ncg > 1:
                    print("error: {}, NCG > 1".format(word))
                res[param_key][tok_len]['NCG'].append(ncg)
        return res

    def generate_logs(self, res):
        """
        Process and generate logs.
        """
        average_log = {}  # {param: {tok_len: {'time_cost': avg_time_cost, 'NCG': avg_NCG}}}
        simple_log = {}  # {param:{'time_cost':total_time_cost, 'NCG': total_NCG}}
        for param, res_dict in res.items():
            average_log.setdefault(param, {})
            simple_log.setdefault(param, {})
            cache = {'time_cost':[], 'NCG': []}
            for tok_len, tok_dict in res_dict.items():
                average_log[param].setdefault(tok_len, {})

                tok_time_cost = tok_dict['time_cost']
                average_log[param][tok_len]['time_cost'] = sum(tok_time_cost)/len(tok_time_cost)
                cache['time_cost'].extend(tok_time_cost)

                tok_recall = tok_dict['NCG']
                average_log[param][tok_len]['NCG'] = sum(tok_recall) / len(tok_recall)
                cache['NCG'].extend(tok_recall)
            simple_log[param]['time_cost'] = sum(cache['time_cost']) / len(cache['time_cost'])
            simple_log[param]['NCG'] = sum(cache['NCG']) / len(cache['NCG'])
        return simple_log, average_log

    def __call__(self, log_details, evaluate_num=None, topk=5, weights=None, **var_params):
        evaluate_num = evaluate_num or self.evaluate_num
        weights = weights or self.weights
        res = self.evaluate(evaluate_num, topk, weights, **var_params)
        simple_log, average_log = self.generate_logs(res)
        final_log = dict(info=log_details['info'], confusor_args=self.confusor_args,
                         params=var_params, simple_log=simple_log, average_log=average_log)
        json.dump(final_log, open(LOGPATH + log_details['name'] + '.json', 'w'))


if __name__ == "__main__":
    conf_args = {
        'method': 'beam',
        'cos_threshold': (0.1, 0.5),
        'cand_pinyin_num': 10,
        'debug': True,
        'cand_fzimu_num': 100,  # two-stage
        'cand_dcc_num': 50,  # dcc
        'cand_zi_num': 50,  # beam
        'keep_num': 500,  # beam
    }
    log_details_beam = {
        'name': 'beam_search',
        'info': 'evaluate time cost and recall of beam-search retrieval.'
    }
    params_beam = dict(cand_zi_num=[50], keep_num=[500])
    eval = Evaluator(conf_args)
    eval(log_details_beam, **params_beam)
