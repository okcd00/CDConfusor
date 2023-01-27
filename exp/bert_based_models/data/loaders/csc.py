"""
@Time   :   2021-01-21 14:58:30
@File   :   csc.py
@Author :   Abtion, okcd00
@Email  :   abtion{at}outlook.com
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from bbcm.data.datasets.csc import CscDataset, PureTextDataset


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_items = next(self.loader)
        except StopIteration:
            self.next_items = None
            return

        with torch.cuda.stream(self.stream):
            self.next_items = list(map(
                lambda x: x.cuda(non_blocking=True), 
                self.next_items
            ))
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_items = self.next_items
        self.preload()
        return next_items


def get_csc_loader(fp, _collate_fn, pure_text_dataset=False, **kwargs):
    load_start_time = time.time()
    if pure_text_dataset:
        dataset = PureTextDataset(fp)
        kwargs['shuffle'] = False
    else:
        dataset = CscDataset(fp)
    load_time_cost = time.time() - load_start_time
    print(f"Loaded {len(dataset)} samples from {fp}. Cost {load_time_cost:.3f} seconds.")
    loader = DataLoader(
        dataset, collate_fn=_collate_fn, 
        prefetch_factor=1, **kwargs)
    return loader
