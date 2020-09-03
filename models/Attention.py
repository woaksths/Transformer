import torch
import torch.nn as nn
import math
from .utils import clones


def attention(query, key, value, mask=None, dropout=None):
    '''
    Applies a scaled dot product attention
    
    Args:  query, key, value
         - **query, key, value** (batch_size, head_size, seq_len, hidden_dim) 
    Returns: output, p_attn
         - **output** (batch_size, head_size, seq_len, hidden_dim)
         - **p_attn** (batch_size, head_size, seq_len, seq_len)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0          
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, query, key, value, mask=None):
        '''
        Applies a multiheaded attention 
        
        Args: query, key, value
             - **query, key, value** (batch_size, seq_len, embedding_dim)
             - **mask** (batch_size, seq_len)
        Returns: output
            - **output** (batch_size, seq_len, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        seq_len = query.size(1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) \
                             for l,x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h*self.d_k)
        output = self.linears[-1](x) 
        return output
