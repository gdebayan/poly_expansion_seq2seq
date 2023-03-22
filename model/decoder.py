 #import torch
from torch import Tensor
import torch.nn as nn
import torch
import math
import sys

sys.path.insert(0, '../')

from model.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 dropout,
                 pf_dim):
        super().__init__()        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout)
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid-dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]            
        return trg, attention