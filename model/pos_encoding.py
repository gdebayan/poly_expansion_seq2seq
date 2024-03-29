 #import torch
from torch import Tensor
import torch.nn as nn
import torch
import math
import sys

sys.path.insert(0, '../')


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        """Module that adds positional encoding to the token embedding to introduce a notion of word order."""

        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):

        token_embedding = token_embedding.permute(1, 0, 2) # B, SEQ_LEN, D --> SEQ_LEN, B, D

        out =  self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

        out = out.permute(1, 0, 2) # SEQ_LEN, B, D ---> B, SEQ_LEN, D

        return out

