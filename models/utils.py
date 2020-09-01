import torch
import torch.nn as nn
import copy 
import numpy as np


def clones(module, N):
    '''
    Args: moduel, N
        - **module**: kind of nn.module class 
        - **N**: copy num
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
