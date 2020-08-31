import torch
import torch.nn as nn
import copy 

def clones(module, N):
    '''
    Args: moduel, N
        - **module**: kind of nn.module class 
        - **N**: copy num
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

