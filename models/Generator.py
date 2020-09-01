import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

