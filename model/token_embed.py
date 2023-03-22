
import sys
import torch
from torch import Tensor
import torch.nn as nn
import math

sys.path.insert(0, '../')


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        """Module to convert tensor of input indices into corresponding tensor of token embeddings."""
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)