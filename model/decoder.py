
import torch
import torch.nn as nn
from utils import clones
from add_norm import LayerNorm, ResidualNet


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.resnet = clones(ResidualNet(size, dropout), 3) 

    def forward(self, x, enc_outputs, src_mask, tgt_mask):
        x = self.resnet[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.resnet[1](x, lambda x: self.src_attn(x, enc_outputs, enc_outputs, src_mask))
        return self.resnet[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, enc_outputs, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_outputs, src_mask, tgt_mask)
        return self.norm(x)

