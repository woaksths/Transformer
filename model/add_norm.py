import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        '''
        Args:
            **features** (int): hidden dim size 
        '''
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std= x.std(-1, keepdim=True)
        return self.a2*(x-mean) / (std+self.eps) + self.b2


class ResidualNet(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualNet, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        '''
        Args:
            **sublayer** : three type of sublayer (self attn, src attn, ffn)
        '''
        return x + self.dropout(sublayer(self.norm(x)))
