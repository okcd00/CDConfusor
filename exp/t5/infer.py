# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./output/mengzi-t5-base-chinese-correction-test/', help='save dir')
    parser.add_argument('--test_path', type=str, default='../data/cn/sighan15/sighan15_test.tsv', help='')
    parser.add_argument('--max_len', type=int, default=128, help='')
    args = parser.parse_args()
    return args


def predict(example_sentences):
    args = parse_args()
    model_dir = args.save_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    results = []
    for s in example_sentences:
        model_inputs = tokenizer(s, max_length=args.max_len, truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**model_inputs, max_length=args.max_len)
        r = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(r)
    return results


if __name__ == '__main__':
    example_sentences = [
        "我跟我朋唷打算去法国玩儿。",
        "少先队员因该为老人让坐。",
        "我们是新时代的接斑人",
        "我咪路，你能给我指路吗？",
        "他带了黑色的包，也带了照像机",
        '因为爸爸在看录音机，所以我没得看',
        '不过在许多传统国家，女人向未得到平等',
        '妈妈说："别趴地上了，快起来，你还吃饭吗？"，我说："好。"就扒起来了。',
        '你说：“怎么办？”我怎么知道？',
        '我父母们常常说：“那时候吃的东西太少，每天只能吃一顿饭。”想一想，人们都快要饿死，谁提出化肥和农药的污染。',
        '这本新书《居里夫人传》将的很生动有趣',
        '֍我喜欢吃鸡，公鸡、母鸡、白切鸡、乌鸡、紫燕鸡……֍新的食谱',
        '注意：“跨类保护”不等于“全类保护”。',
        '12.——对比文件中未公开的数值和对比文件中已经公开的中间值具有新颖性；',
        '《著作权法》（2020修正）第23条：“自然人的作品，其发表权、本法第',
        '三步检验法（三步检验标准）（three-step test）：若要',
        '三步检验法“三步‘检验’标准”（three-step test）：若要',
        '泗阳交警机动中队查获一起小型面包车车超员的违法行为。',
        '妻子遭国民党联保“打地雷公”的酷刑，生活无依靠，沿村乞讨度日。',
        '风力预报5日白天起风力逐渐加大，预计5～7日高海拔山区区最大风力可达6～7级阵风8～9级。',
        '8月1日下午，芦淞区副区长、芦淞公安分局分局党委书记、局长樊伟；芦淞分局党委成员、副局长汤征等一行人到辖区隔离酒店进行督导检查。',
        '发布了头条文章：《广东公安八大专项行动｜古玩骗局“换装上线”，警方提醒请勿利令智》',
    ]
    r = predict(example_sentences)
    for i, o in zip(example_sentences, r):
        print(i, ' -> ', o)
