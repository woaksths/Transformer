import os
import argparse
import torch
import torchtext
import torch.nn as nn

import models
from models import make_model
from models import SourceField, TargetField


parser = argparse.ArgumentParser(description='Transformer Tutorial')
parser.add_argument('--train_path', help='Path to train data')
parser.add_argument('--dev_path', help='Path to dev data')
parser.add_argument('--expt_dir', default='./experiment', help='Path to experiment directory' )
parser.add_argument('--load_checkpoint', help='The name of checkpoint to load')
parser.add_argument('--resume', default=False, help='Indicates if training has to be resumed from the latest checkpoint')
opt = parser.parse_args()


if opt.load_checkpoint is not None:
    pass
else:
    src = SourceField()
    tgt = TargetField()
    max_len = 50
    
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    
    train = torchtext.data.TabularDataset(
        path = opt.dev_path, format='tsv',
        fields = [('src',src),('tgt',tgt)],
        filter_pred = len_filter
    )
    dev = torchtext.data.TabularDataset(
        path = opt.dev_path, format='tsv',
        fields = [('src',src),('tgt',tgt)],
        filter_pred = len_filter
    )
    
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    
    input_vocab = src.vocab
    output_vocab = tgt.vocab
    
    