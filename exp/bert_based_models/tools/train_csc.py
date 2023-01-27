"""
@Time   :   2021-01-21 11:47:09
@File   :   train_csc.py
@Author :   okcd00, Abtion
@Email  :   okcd00@qq.com, abtion{at}outlook.com
"""
import os
import sys
import torch
print("PyTorch Verstion:", torch.__version__)
import torch.multiprocessing as mp
# mp.set_start_method('spawn')

import shutil
import tensorboard
sys.path.append('..')

from transformers import BertTokenizer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from tools.bases import args_parse, train, dynamic_train
from bbcm.utils import get_abs_path

# data
from bbcm.data.build import make_loaders, get_dynamic_loader
from bbcm.data.loaders import get_csc_loader, DataCollatorForCsc, DynamicDataCollatorForCsc
from bbcm.data.processors.csc import preproc, preproc_cd

# modeling
from bbcm.modeling.csc import SoftMaskedBertModel
from bbcm.modeling.csc.modeling_bert4csc import BertForCsc
from bbcm.modeling.cdnet import CSC_MODEL_CLASSES


def set_random_seeds(seed):
    import torch
    import random 
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Set {seed} as the Random Seed.")


class PretCheckpoint(ModelCheckpoint):
    """save checkpoint after each training epoch without validation.
    if ``last_k == -1``, all models are saved. and no monitor needed in this condition.
    otherwise, please log ``global_step`` in the training_step. e.g. self.log('global_step', self.global_step)

    :param last_k: the latest k models will be saved.
    :param save_weights_only: if ``True``, only the model's weights will be saved,
    else the full model is saved.
    """
    def __init__(self, monitor=None, dirpath=None,  # Optional[Union[str, Path]] 
                 filename=None,  # Optional[str]
                 save_top_k=3,  # Optional[int]
                 save_weights_only=False,
                 save_last=False,
                 verbose=False):

        self.filename = filename
        if save_top_k == -1:
            super().__init__(mode='max',
                             save_top_k=-1, verbose=verbose,
                             dirpath=dirpath, filename=filename,
                             save_last=save_last, save_weights_only=save_weights_only)
        else:
            super().__init__(monitor='global_step', mode='max', 
                             save_top_k=save_top_k, verbose=verbose,
                             dirpath=dirpath, filename=filename,
                             save_last=save_last, save_weights_only=save_weights_only)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """Save a checkpoint at the end of the training epoch."""
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        print("[Saving] Now saving checkpoint...")
        self.save_checkpoint(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        """Save a checkpoint at the end of the validation stage."""
        """We don't have a valid loader in pre-training"""
        pass


def build_model():
    set_random_seeds(42)
    # cfg = args_parse("csc/train_bert4csc.yml")
    cfg = args_parse()
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    # from transformers import PreTrainedTokenizerFast
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=json_cfg_path,
    #                                     bos_token='<s>', eos_token='</s>', unk_token='<unk>',
    #                                     pad_token='<pad>', mask_token='<mask>')
    if cfg.MODEL.NAME in ["bert4csc", "macbert4csc"]:
        model = BertForCsc(cfg, tokenizer)
    else:  # SoftMaskedBert
        model_class = CSC_MODEL_CLASSES.get(
            cfg.MODEL.NAME.lower(), SoftMaskedBertModel)
        model = model_class(cfg, tokenizer)

    if len(cfg.MODEL.WEIGHTS) > 0:
        ckpt_path = get_abs_path(
            cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS)
        if os.path.exists(cfg.OUTPUT_DIR):
            # auto-find a ckpt file with the lowest loss.
            if not os.path.exists(ckpt_path):
                from glob import glob
                ckpt_path = glob(f"{cfg.OUTPUT_DIR}/*.ckpt")[-1]
            if os.path.exists(ckpt_path):
                print("WEIGHT {} exists.".format(ckpt_path))
        model = model.load_from_checkpoint(
            ckpt_path, cfg=cfg, tokenizer=tokenizer, 
            map_location='cpu', strict=False)
        print("Loaded from {}".format(ckpt_path))

    return cfg, model, tokenizer


def main():
    cfg, model, tokenizer = build_model()

    loaders = make_loaders(cfg, get_csc_loader,
                           _collate_fn=DataCollatorForCsc(
                               cfg=cfg,
                               tokenizer=tokenizer, 
                               need_pinyin=model.predict_pinyins))
    # loaders = make_dynamic_loaders(cfg, get_csc_loader, _collate_fn=None)
    ckpt_callback = ModelCheckpoint(
        monitor='valid_loss',  # 'valid_loss',
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch:02d}_{valid_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    # the training from which rank is done
    global_rank = train(cfg, model, loaders, ckpt_callback)
    # print(f"Rank-[{global_rank}] Done training phase.")

    # only RANK0 take the inference phase
    if global_rank == 0:
        from tools.infer import Inference
        m = Inference(ckpt_path=cfg.OUTPUT_DIR)

        print("Now testing on SIGHAN15")
        from bbcm.utils.evaluations import eval_sighan2015_by_model
        acc, precision, recall, f1 = eval_sighan2015_by_model(
            correct_fn=m.predict_with_error_detail,
            # os.path.join(cfg.MODEL.PROJECT_ROOT, "datasets/csc_aug/sighan15_test_ct_covered.txt"))
            sighan_path=str(cfg.DATASETS.TEST))
    print(f"Acc: {acc:.4f}")
    print(f"{precision:.4f}/{recall:.4f}/{f1:.4f}")


def main_last(save_per_n_steps=0):
    cfg, model, tokenizer = build_model()

    loaders = make_loaders(cfg, get_csc_loader, 
                           pin_memory=False,
                           _collate_fn=DataCollatorForCsc(
                               cfg=cfg,
                               tokenizer=tokenizer, 
                               need_pinyin=model.predict_pinyins))
    # loaders = make_dynamic_loaders(
    #     cfg, get_csc_loader, _collate_fn=None)

    if save_per_n_steps > 0:
        ckpt_callback = PretCheckpoint(
            monitor='global_step',  
            dirpath=get_abs_path(cfg.OUTPUT_DIR),
            save_top_k=-1,
        )
    else:
        default_namespace = '{epoch:02d}_{train_loss_epoch:.4f}_{train_det_f1_epoch:.4f}_{train_cor_f1_epoch:.4f}'
        ckpt_callback = ModelCheckpoint(
            monitor='train_loss_epoch',        
            dirpath=get_abs_path(cfg.OUTPUT_DIR),
            filename=cfg.NAMESPACE if cfg.NAMESPACE else default_namespace,
            save_top_k=3,
            mode='min'
        )

    # the training from which rank is done
    global_rank = train(cfg, model, loaders, ckpt_callback)
    # print(f"Rank-[{global_rank}] Done training phase.")

    # only RANK0 take the inference phase
    if global_rank == 0 and False:  # skip this for pre-training.
        from tools.infer import Inference
        from bbcm.utils.evaluations import eval_sighan2015_by_model
        m = Inference(ckpt_path=cfg.OUTPUT_DIR)

        if os.path.exists(str(cfg.DATASETS.TRAIN)) and False:  # skip eval on train.
            print("Now validating on SIGHAN15-train")
            acc, precision, recall, f1 = eval_sighan2015_by_model(
                correct_fn=m.predict_with_error_detail,
                sighan_path=str(cfg.DATASETS.TRAIN))
            print(f"Acc: {acc:.4f}")
            print(f"{precision:.4f}/{recall:.4f}/{f1:.4f}")

        if os.path.exists(str(cfg.DATASETS.TEST)) and False:  # sktip eval on test
            print("Now validating on SIGHAN15-test")
            acc, precision, recall, f1 = eval_sighan2015_by_model(
                correct_fn=m.predict_with_error_detail,
                sighan_path=str(cfg.DATASETS.TEST))
            print(f"Acc: {acc:.4f}")
            print(f"{precision:.4f}/{recall:.4f}/{f1:.4f}")


def dynamic_main_fixed():
    cfg, model, tokenizer = build_model()

    loaders = make_loaders(
        cfg, get_csc_loader, 
        _collate_fn=DataCollatorForCsc(tokenizer=tokenizer))

    ckpt_callback = ModelCheckpoint(
        monitor=None,
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch:02d}_{val_loss:.5f}',
        save_top_k=-1,
    )
    fixed = "fix" in str(cfg.OUTPUT_DIR)
    dynamic_train(cfg, model, loaders, ckpt_callback, fixed=fixed)


def dynamic_main():
    cfg, model, tokenizer = build_model()

    # the collate_function
    col_fn_train = DynamicDataCollatorForCsc(
        tokenizer=tokenizer, augmentation=True, raw_first_epoch=True)
    col_fn_test = DataCollatorForCsc(tokenizer=tokenizer)

    train_loader = get_csc_loader(
        cfg.DATASETS.TRAIN,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=False,
        _collate_fn=col_fn_train,
        pure_text_dataset=False,  # auto-generating
    )
    
    # No validation
    # valid_loader = None
    valid_loader = get_csc_loader(
        get_abs_path(cfg.DATASETS.VALID),
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=False,
        _collate_fn=col_fn_test,
        pure_text_dataset=False,
    )  
    
    test_loader = get_csc_loader(
        get_abs_path(cfg.DATASETS.TEST),
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=False,
        _collate_fn=col_fn_test,
        pure_text_dataset=False,
    )

    # loaders = make_dynamic_loaders(cfg, get_csc_loader, _collate_fn=None)
    ckpt_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch:02d}_{val_loss:.5f}',
        save_top_k=3,
        mode='min'
    )
    loaders = (train_loader, valid_loader, test_loader)
    train(cfg, model, loaders, ckpt_callback)
    return "Finished"


def dynamic_train_on_texts():
    cfg, model, tokenizer = build_model()

    # the collate_function
    col_fn_train = DynamicDataCollatorForCsc(tokenizer=tokenizer, augmentation=True)
    col_fn_test = DataCollatorForCsc(tokenizer=tokenizer)

    train_loader = get_csc_loader(
        cfg.DATASETS.TRAIN,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=False,
        _collate_fn=col_fn_train,
        pure_text_dataset=True,  # auto-generating
    )
    
    # No validation
    # valid_loader = None
    valid_loader = get_csc_loader(
        get_abs_path(cfg.DATASETS.VALID),
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=False,
        _collate_fn=col_fn_test,
        pure_text_dataset=False,
    )  
    
    test_loader = get_csc_loader(
        get_abs_path(cfg.DATASETS.TEST),
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=False,
        _collate_fn=col_fn_test,
        pure_text_dataset=False,
    )

    ckpt_callback = ModelCheckpoint(
        monitor='trn_loss',
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch:02d}_{val_loss:.5f}',
        save_weights_only=False,
        save_last=True,
        mode='min'
    )

    """
    # loaders = make_dynamic_loaders(cfg, get_csc_loader, _collate_fn=None)
    ckpt_callback = PretCheckpoint(
        monitor='global_step',  
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        save_top_k=-1,
    )
    """

    loaders = (train_loader, valid_loader, test_loader)
    train(cfg, model, loaders, ckpt_callback)
    return "Finished"


if __name__ == '__main__':
    # if not os.path.exists(get_abs_path(cfg.DATASETS.TRAIN)):
    # 如果不存在训练文件则先处理数据
    # preproc()
    # preproc_cd()

    # main()
    main_last()
    # dynamic_main()
    # dynamic_train_on_texts()
