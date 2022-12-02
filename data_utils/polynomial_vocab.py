import json
import re
import logging
from typing import List, Tuple

import sys
sys.path.insert(0, '../')

import config

logging.basicConfig(level = config.Config.LOGGING_LEVEL)

class PolynomialVocab:

    UNK_INDEX = 0
    PAD_INDEX = 1
    SOS_INDEX = 2
    EOS_INDEX = 3
    
    def __init__(self, pre_compute_vocab=True) -> None:
        """Utility class to process the Polynomial Dataset.
        
        The tokens used to represent the polynomial dataset are:
            sin, cos, tan, numeric digit, alphabet, (, ), +, -, **, *
        """
        self.regexp_pattern = r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*\*|\*"
 
        # {Token: Index} mapping which will be used for creating Input/Output Tensors for model training
        if pre_compute_vocab:
            # Load the pre-computed token-index mapping
            with open('data_utils/vocab.json', "r") as f:
              self.token_to_index = json.loads(f.read())
        else:
            # Else, the token-index mapping will be built on-the-fly as we process each ip-op pair
            self.token_to_index = {'<unk>': PolynomialVocab.UNK_INDEX,
                                   '<pad>': PolynomialVocab.PAD_INDEX,
                                   '<sos>': PolynomialVocab.SOS_INDEX,
                                   '<eos>': PolynomialVocab.EOS_INDEX}

        # {Token: token_count} mapping. Used to store token-statistics of a dataset
        self.token_to_count = {}

        # Token Count. Used to indicate Vocabulary Size for the dataset
        self.token_count = len(self.token_to_index)
        
    @property
    def index_to_token(self) -> dict:
        """Helper funtion to retun {index: token} mapping"""
        return {v: k for k, v in self.token_to_index.items()}

    def sentence_to_tokens(self, sentence: str) -> List[str]:
        """Tokenizes an input string to Tokens.

        Example:
            I/P: (cos(x)**2 + 4*x) * 3*x
            O/P: ['(', 'cos', '(', 'x', ')', '**', '2', '+', '4', '*', 'x', ')', '*', '3', '*', 'x']
        """
        return re.findall(self.regexp_pattern, sentence.lower().strip())

    def add_token(self, token: str) -> None:
        """Adds token to Dataset Vocabulary."""

        if self.token_to_index.get(token, None) is None:
            self.token_to_index[token] = self.token_count
            self.token_count += 1

        self.token_to_count[token] = self.token_to_count.get(token, 0) + 1

    def add_sentence(self, sentence: str) -> None:
        """Adds Tokens from a Sentence to the Dataset Vocabulary."""
        tokens = self.sentence_to_tokens(sentence=sentence)
        for token in tokens:
            self.add_token(token=token)

    @staticmethod
    def process_ip_op_pairs(ip_op_pairs: List[List[str]], pre_compute_vocab: bool = True) -> Tuple['PolynomialVocab', 'PolynomialVocab']:
        """
        Process the entire dataset for the Ip / Op pairs seperately.
        """
        src_vocab = PolynomialVocab(pre_compute_vocab=pre_compute_vocab)
        tgt_vocab = PolynomialVocab(pre_compute_vocab=pre_compute_vocab)
        for ip, op in ip_op_pairs:
            src_vocab.add_sentence(ip)
            tgt_vocab.add_sentence(op)
        return src_vocab, tgt_vocab
        