 #import torch
from torch import Tensor
import torch.nn as nn
import torch
import math
import sys

sys.path.insert(0, '../')

from model.encoder import Encoder
from model.encoder_s4 import EncoderS4Attention
from model.decoder import Decoder
from model.token_embed import TokenEmbedding
from model.pos_encoding import PositionalEncoding

from data_utils.polynomial_vocab import PolynomialVocab


class S4Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int=512,
                 dropout: float=0.1,
                 src_pad_idx=1,
                 tgt_pad_idx=1):
        super().__init__()

        # self.encoder = Encoder(hid_dim=emb_size, 
        #                        n_layers=num_encoder_layers,
        #                        n_heads=nhead,
        #                        dropout=dropout,
        #                        pf_dim=dim_feedforward) 
        self.encoder = EncoderS4Attention(hid_dim=emb_size, 
                                            n_heads=nhead,
                                            dropout=dropout,
                                            pf_dim=dim_feedforward) 

        self.decoder = Decoder(hid_dim=emb_size, 
                               n_layers=num_decoder_layers, 
                               n_heads=nhead, 
                               dropout=dropout,
                               pf_dim=dim_feedforward)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_tgt_mask(self, tgt):
        
        #tgt = [batch size, tgt len]
        
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #tgt_pad_mask = [batch size, 1, 1, tgt len]
        
        tgt_len = tgt.shape[1]
        
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        
        #tgt_sub_mask = [tgt len, tgt len]
            
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        
        return tgt_mask

    def encode(self, src, src_mask, src_lens):
        #src = [batch size, src len]

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        #src_emb = [batch size, src len, dim]

        enc_src = self.encoder(src_emb, src_mask, src_lens)

        return enc_src

    def decode(self, tgt, enc_src, tgt_mask, src_mask):
        #tgt = [batch size, tgt len]

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        #tgt_emb = [batch size, src len, dim]

        output, attention = self.decoder(tgt_emb, enc_src, tgt_mask, src_mask)
        #output = [batch size, tgt len, output dim]
        #attention = [batch size, n heads, tgt len, src len]

        return output, attention

    def forward(self, src, tgt, src_lens, tgt_lens):
        #src = [batch size, src len]
        #tgt = [batch size, tgt len]

        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src = self.encode(src=src, src_mask=src_mask, src_lens=src_lens)

        output, attention = self.decode(tgt=tgt, enc_src=enc_src, 
                                         tgt_mask=tgt_mask, src_mask=src_mask)

        return self.generator(output)