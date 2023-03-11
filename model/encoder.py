 #import torch
from torch import Tensor
import torch.nn as nn
import torch
import math
import sys

sys.path.insert(0, '../')

import config
DEVICE = config.Config.DEVICE

from model.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 dropout, 
                 pf_dim):
        super().__init__()
                
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]

        #src_mask =  [batch_size, 1, 1, src_len]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
        return src