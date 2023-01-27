"""
@Time   :   2021-01-21 11:17:25
@File   :   bases.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import os
import copy
import shutil
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from bbcm.config import cfg
from bbcm.utils.logger import setup_logger
from bbcm.utils import get_abs_path, dump_json
from collections import OrderedDict


def args_parse(config_file=''):
    parser = argparse.ArgumentParser(
        description="bbcm")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--opts", help="Modify config options using the command-line key value", 
                        default=[], nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config_file = args.config_file or config_file

    if config_file != "":
        cfg.merge_from_file(get_abs_path('configs', config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    name = cfg.MODEL.NAME
    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger(name, get_abs_path(output_dir), 0)
    logger.info(args)

    if config_file != '':
        logger.info("Loaded configuration file {}".format(config_file))
        with open(get_abs_path('configs', config_file), 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    # save config.yml
    shutil.copyfile(
        get_abs_path('configs', config_file),
        get_abs_path(output_dir, 'config.yml'))
    logger.info("Running with config:\n{}".format(cfg))
    return cfg


def to_transformers_form(ckpt_callback, model):
    # 模型转为transformers可加载 from shibing624/pycorrector
    if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
        ckpt_path = ckpt_callback.best_model_path
    elif cfg.MODEL.WEIGHTS and os.path.exists(cfg.MODEL.WEIGHTS):
        ckpt_path = cfg.MODEL.WEIGHTS
    else:
        ckpt_path = ''
    
    if ckpt_path and os.path.exists(ckpt_path):
        from transformers import BertTokenizer, BertForMaskedLM
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        
        # 先保存原始 transformer bert model files
        cfg = args_parse("csc/train_bert4csc.yml")
        tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
        tokenizer.save_pretrained(cfg.OUTPUT_DIR)
        bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        bert.save_pretrained(cfg.OUTPUT_DIR)

        # load trained model
        state_dict = torch.load(ckpt_path)['state_dict']
        new_state_dict = OrderedDict()
        if cfg.MODEL.NAME in ['macbert4csc']:
            for k, v in state_dict.items():
                if k.startswith('bert.'):
                    new_state_dict[k[5:]] = v
        else:
            new_state_dict = state_dict
        # 再保存finetune训练后的模型文件，替换原始的pytorch_model.bin
        torch.save(new_state_dict, os.path.join(cfg.OUTPUT_DIR, 'pytorch_model.bin'))


def train(config, model, loaders, ckpt_callback=None):
    """
    训练
    Args:
        config: 配置
        model: 模型
        loaders: 各个数据的loader，包含train，valid，test
        ckpt_callback: 按需保存模型的callback，如为空则默认每个epoch保存一次模型。
    Returns:
        None
    """
    train_loader, valid_loader, test_loader = loaders

    """
    Lightning supports two backends. 
    DataParallel and DistributedDataParallel. 
    Both can be used for single-node multi-GPU training. 
    For multi-node training you must use DistributedDataParallel.

    DATAPARALLEL (DP)
    Splits a batch across multiple GPUs on the same node. 
    Cannot be used for multi-node training.

    DISTRIBUTEDDATAPARALLEL (DDP)
    Trains a copy of the model on each GPU and only syncs gradients. 
    If used with DistributedSampler, each GPU trains on a subset of the full dataset.

    DISTRIBUTEDDATAPARALLEL-2 (DDP2)
    Works like DDP, except each node trains a single copy of the model using ALL GPUs on that node. 
    Very useful when dealing with negative samples, etc…
    """
    from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
    trainer = pl.Trainer(max_epochs=config.SOLVER.MAX_EPOCHS,
                         # limit_train_batches=0.1,  小数据训练
                         # limit_val_batches=0.2,  小数据验证
                         gpus=None if config.MODEL.DEVICE == 'cpu' else config.MODEL.GPU_IDS,
                         accumulate_grad_batches=config.SOLVER.ACCUMULATE_GRAD_BATCHES,
                         distributed_backend="ddp" if len(config.MODEL.GPU_IDS) > 1 else None,
                         plugins=[DDPPlugin(find_unused_parameters=False)],
                         callbacks=[ckpt_callback])
                         
    # 满足以下条件才进行训练
    # 1. 配置文件中要求进行训练
    # 2. train_loader不为空
    # 3. train_loader中有数据
    if 'train' in config.MODE and train_loader and len(train_loader) > 0:
        if valid_loader and len(valid_loader) > 0:
            trainer.fit(model, train_loader, valid_loader)
        else:
            trainer.fit(model, train_loader)
        
        # save/load the trainer (single GPU)
        # trainer.save_checkpoint("example.ckpt")
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
        
        # 如果需要存成 transformers 可以读取的格式
        # to_transformers_form(ckpt_callback, model)

    # 是否进行测试的逻辑同训练
    if 'test' in config.MODE and test_loader and len(test_loader) > 0:
        ckpt_path = None
        if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
            ckpt_path = ckpt_callback.best_model_path  # from callback
        elif len(config.MODEL.WEIGHTS) > 0:  # load during pure-inference
            ckpt_path = get_abs_path(config.OUTPUT_DIR, config.MODEL.WEIGHTS)
        if trainer.global_rank == 0:
            print("Load", ckpt_path, "as ckpt_path.")
        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])  
        trainer.test(model, test_loader)
    
    return trainer.global_rank


def dynamic_train(config, model, loaders, ckpt_callback=None, fixed=False):
    """
    训练
    Args:
        config: 配置
        model: 模型
        loaders: 各个数据的loader，包含train，valid，test
        ckpt_callback: 按需保存模型的callback，如为空则默认每个epoch保存一次模型。
    Returns:
        None
    """
    logs = {}
    train_loader, valid_loader, test_loader = loaders

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    from bbcm.data.loaders.collator import DataCollatorForCsc
    _collate_fn = DataCollatorForCsc(tokenizer=tokenizer)

    from bbcm.data.build import get_train_loader

    if fixed:  # always the first epoch.
        train_loader = get_train_loader(
            cfg=config, ep=1, _collate_fn=_collate_fn)

    for epoch in range(config.SOLVER.MAX_EPOCHS):
        trainer = pl.Trainer(max_epochs=1,  # train one single epoch
                             gpus=None if config.MODEL.DEVICE == 'cpu' else config.MODEL.GPU_IDS,
                             accumulate_grad_batches=config.SOLVER.ACCUMULATE_GRAD_BATCHES,
                             enable_progress_bar=True,
                             callbacks=[ckpt_callback])
        if not fixed:
            train_loader = get_train_loader(
                cfg=config, ep=epoch + 1,
                _collate_fn=_collate_fn)
        if 'train' in config.MODE and train_loader and len(train_loader) > 0:
            if valid_loader and len(valid_loader) > 0:
                trainer.fit(model, train_loader, valid_loader)
            else:
                trainer.fit(model, train_loader)

        logs[epoch] = []
        # for ep in range(epoch, -1, -1):
        for ep in range(1, config.SOLVER.MAX_EPOCHS + 1):  # test for all
            print(f"\n=====Test on {ep}-th epoch=====\n")
            t_loader = get_train_loader(  # test on train set
                cfg=config, ep=ep, _collate_fn=_collate_fn)
            res = trainer.test(model, t_loader)
            logs[epoch].append(copy.deepcopy(res))

        if 'test' in config.MODE and test_loader and len(test_loader) > 0:
            if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
                ckpt_path = ckpt_callback.best_model_path
            elif len(config.MODEL.WEIGHTS) > 0:
                # for the case of testing without training
                ckpt_path = get_abs_path(config.OUTPUT_DIR, config.MODEL.WEIGHTS)
            else:
                ckpt_path = None
            print(ckpt_path)
            # if (ckpt_path is not None) and os.path.exists(ckpt_path):
            #     model.load_state_dict(torch.load(ckpt_path)['state_dict'])
            res = trainer.test(model, test_loader)
            logs[epoch].append(copy.deepcopy(res))

    from pprint import pprint
    pprint(logs)
    dump_json(logs, get_abs_path(config.OUTPUT_DIR, 'dynamic_training.log'))
