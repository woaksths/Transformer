import torch
import torch.nn as nn
from .utils import clones
from .AddNorm import ResidualNet, LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.resnet = clones(ResidualNet(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x =  self.resnet[0](x, lambda x: self.self_attn(x,x,x, mask))
        return self.resnet[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, encoder_layer ,N):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, N)
        self.norm = LayerNorm(encoder_layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

