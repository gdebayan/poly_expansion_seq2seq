import sys
from torch import Tensor, device
import torch.nn as nn
from torch.nn import Transformer
import torch

sys.path.insert(0, '../')

from model.token_embed import TokenEmbedding
from model.pos_encoding import PositionalEncoding
import config
DEVICE = config.Config.DEVICE

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int=512,
                 dropout: float=0.1,
                 batch_first: bool=True):
        super(Seq2SeqTransformer, self).__init__()
        print('batch first', batch_first)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout, device=DEVICE,
                                       batch_first=batch_first)
                                       
        self.generator = nn.Linear(emb_size, tgt_vocab_size, device=DEVICE)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # print("src", src.shape)
        # print("trg", trg.shape)
        # print("src_mask", src_mask.shape)
        # print("tgt_mask", tgt_mask.shape)
        # print("src_padding_mask", src_padding_mask.shape)
        # print("tgt_padding_mask", tgt_padding_mask.shape)
        # print("memory_key_padding_mask", memory_key_padding_mask.shape)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        # print("src_emb", src_emb.shape)
        # print("tgt_emb", tgt_emb.shape)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # print("outs", outs.shape)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)