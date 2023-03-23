 #import torch
from torch import Tensor
import torch.nn as nn
import torch
import math
import sys

sys.path.insert(0, '../')

from model.encoder_layer import EncoderLayer
from model.s4.s4_layer import S4Layer

class EncoderS4Attention(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 dropout, 
                 pf_dim):
        super().__init__()
                
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout),
                                     S4Layer(d_input=hid_dim, d_s4_state=64, d_output=hid_dim, d_model=hid_dim, dropout=0),
                                     S4Layer(d_input=hid_dim, d_s4_state=64, d_output=hid_dim, d_model=hid_dim, dropout=0),
                                     EncoderLayer(hid_dim, n_heads, pf_dim, dropout)])
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, src, src_mask, src_lens):
        
        #src = [batch size, src len, hid dim]

        #src_mask =  [batch_size, 1, 1, src_len]
        
        for layer in self.layers:
            if isinstance(layer, EncoderLayer):
                src = layer(src, src_mask)
            else:
                src = layer(x=src, hidden_state=None, input_lengths=src_lens)
                        
        #src = [batch size, src len, hid dim]
        return src