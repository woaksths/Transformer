import os
import argparse
import torch
import torchtext
import torch.nn as nn
import logging

import models
from models import make_model
from models import SourceField, TargetField
from loss import NLLLoss, CrossEntropyLoss
from trainer import SupervisedTrainer
from optim import NoamOpt, get_std_opt, Optimizer


parser = argparse.ArgumentParser(description='Transformer Tutorial')
parser.add_argument('--train_path', help='Path to train data')
parser.add_argument('--dev_path', help='Path to dev data')
parser.add_argument('--layer', default=6, type=int, help='hp > num of layer')
parser.add_argument('--d_model', default=512, type=int, help='hp > model dimension')
parser.add_argument('--d_ff', default=2048, type=int, help='hp > positionwise feedforward dimension')
parser.add_argument('--head', default=8, type=int, help='num of head')
parser.add_argument('--expt_dir', default='./experiment', help='Path to experiment directory' )
parser.add_argument('--load_checkpoint', help='The name of checkpoint to load')
parser.add_argument('--resume', default=False, help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',default='info', help='Logging level.')
opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    pass
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 100
    
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    
    train = torchtext.data.TabularDataset(
        path = opt.train_path, format='tsv',
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

    #Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = CrossEntropyLoss(weight, pad, reduction='sum')
    
    if torch.cuda.is_available():
        loss.cuda()
    
    transformer = None
    optimizer = None
    
    #Initialize model
    if not opt.resume:
        transformer = make_model(len(input_vocab), len(output_vocab),
                                 N=opt.layer, d_model=opt.d_model, d_ff=opt.d_ff, h=opt.head, dropout=0.1)

        if torch.cuda.is_available():
            transformer.cuda()

    #Initialize optimizer
    if optimizer is None:
        optimizer = Optimizer(torch.optim.Adam(transformer.parameters(),0.0001), max_grad_norm=5) 
        #optimizer = get_std_opt(transformer) # Noam optimizer 
    
    #Train
    t = SupervisedTrainer(loss=loss, batch_size=128,
                          checkpoint_every=300,
                          print_every=300, expt_dir=opt.expt_dir,
                          input_vocab=input_vocab, output_vocab=output_vocab)
    
    transformer = t.train(transformer, train,
                         num_epochs = 100, dev_data=dev,
                         optimizer=optimizer, resume=opt.resume)