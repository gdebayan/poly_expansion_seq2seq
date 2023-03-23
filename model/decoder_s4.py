 #import torch
from torch import Tensor
import torch.nn as nn
import torch
import math
import sys

sys.path.insert(0, '../')

from model.decoder_layer import DecoderLayer
from model.s4.s4_layer import S4Layer

class DecoderS4Attention(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 dropout,
                 pf_dim,
                 s4_conv_eval):
        super().__init__()        
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout),
                                     S4Layer(d_input=hid_dim, d_s4_state=64, d_output=hid_dim, d_model=hid_dim, dropout=0, conv_validation=s4_conv_eval),
                                     S4Layer(d_input=hid_dim, d_s4_state=64, d_output=hid_dim, d_model=hid_dim, dropout=0, conv_validation=s4_conv_eval),
                                     DecoderLayer(hid_dim, n_heads, pf_dim, dropout)])

        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, trg, enc_src, trg_mask, src_mask, tgt_lens):
        
        #trg = [batch size, trg len, hid-dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                        
        for layer in self.layers:
            if isinstance(layer, DecoderLayer):
                trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            else:
                trg = layer(x=trg, hidden_state=None, input_lengths=tgt_lens)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]            
        return trg, attention