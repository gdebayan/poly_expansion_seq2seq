import torch
from typing import List
from data_utils.polynomial_vocab import PolynomialVocab
import sys

sys.path.insert(0, '../')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskUtils:
        
    def generate_square_subsequent_mask(tgt_len):
        """Generates the Square Subsequent Mask for Masked-Self Attention"""
        mask = (torch.triu(torch.ones((tgt_len, tgt_len), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(src, tgt):
        """Creates Source/Target Padding Masks, and Target Mask for Masked-Self Attention"""
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = MaskUtils.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

        src_padding_mask = (src == PolynomialVocab.PAD_INDEX).transpose(0, 1)
        tgt_padding_mask = (tgt == PolynomialVocab.PAD_INDEX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask