import imp
import sys
from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List

sys.path.insert(0, '../')

from data_utils.dataset_utils import DatasetUtils
from data_utils.polynomial_vocab import PolynomialVocab
import config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VocabDataset(torch.utils.data.Dataset):
    """
    Dataset Class for the Polynomial Dataset
    """

    def __init__(self, pairs: List[List[str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ind):        
        ip = self.pairs[ind][0]
        op = self.pairs[ind][1]
        return ip, op

class DataClass:
    """
    Utilities class to return Dataset(s), Dataloader(s) for
    Train, Valdation, Test
    """

    # Pass skip_load_if_exist=True/ Config.RECOMPUTE_DATASE=False, as we don't want to generate a new dataset every run
    # This is so that we can test our models with a standardized test data (which does not change every run)
    skip_load_if_exist = not config.Config.RECOMPUTE_DATASET
    train_pairs, \
    val_pairs, \
    test_pairs = DatasetUtils.train_val_test_split(train_ratio=config.Config.TRAIN_SPLIT_RATIO,
                                                   val_ratio=config.Config.VAL_SPLIT_RATIO,
                                                   test_ratio=config.Config.TEST_SPLIT_RATIO,
                                                   skip_load_if_exist=skip_load_if_exist)

    train_dataset = VocabDataset(pairs=train_pairs)
    val_dataset   = VocabDataset(pairs=val_pairs)
    test_dataset  = VocabDataset(pairs=test_pairs) 

    # Vocabulary for Source (Input Polynomial), and Target (Output Polynomial)
    # If called with pre_compute_vocab=False, different token-index vocab will be constructed for src/tgt
    # If pre_compute_vocab=True, the same token-index will be constructed for both the src/tgt
    src_vocab_cls, tgt_vocab_cls  = PolynomialVocab.process_ip_op_pairs(ip_op_pairs=train_pairs, pre_compute_vocab=True)

    @staticmethod
    def token_transform_src(sentence: str) -> List[str]:
        """Tokenizes an input Sentence (str) into Vocabulary Tokens (List[str]) for Source (Input Polynomial)"""
        return DataClass.src_vocab_cls.sentence_to_tokens(sentence=sentence)

    @staticmethod
    def vocab_transform_src(tokens: List[str]) -> List[int]:
        """Converts Tokenized Sentence into Numeric Form for Source Text (Input Polynomial)"""
        return [DataClass.src_vocab_cls.token_to_index[token] for token in tokens]

    @staticmethod
    def token_transform_tgt(sentence: str) -> List[str]:
        """Tokenizes an input Sentence (str) into Vocabulary Tokens (List[str]) for Target (Expanded Output Polynomial)"""
        return DataClass.tgt_vocab_cls.sentence_to_tokens(sentence=sentence)

    @staticmethod
    def vocab_transform_tgt(tokens) -> List[int]:
        """Converts Tokenized Sentence into Numeric Form for Target Text (Expanded Output Polynomial)"""
        return [DataClass.tgt_vocab_cls.token_to_index[token] for token in tokens]

    @staticmethod
    def token_transform_map():
        """Returns mapping for {src/tgt: token_transform function}"""
        return {'src': DataClass.token_transform_src, 'tgt': DataClass.token_transform_tgt}

    @staticmethod
    def vocab_transform_map():
        """Returns mapping for {src/tgt: vocab_transform function}"""
        return {'src': DataClass.vocab_transform_src, 'tgt': DataClass.vocab_transform_tgt}
    
    @staticmethod
    def sequential_transforms(*transforms):
        """helper method to club together sequential operations"""
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    @staticmethod
    def tensor_transform(token_ids: List[int]):
        """method to add SOS/EOS and create tensor for input sequence indices"""
        return torch.cat((torch.tensor([PolynomialVocab.SOS_INDEX]), 
                          torch.tensor(token_ids), 
                          torch.tensor([PolynomialVocab.EOS_INDEX])))

    @staticmethod
    def text_transform_map():
        """src (Input Polynomial) and tgt (Target Polynomial) text transforms
             to convert raw strings into tensors indices."""
        text_transform_map = {}
        for ln in ['src', 'tgt']:
            text_transform_map[ln] = DataClass.sequential_transforms(DataClass.token_transform_map()[ln], #Tokenization
                                                                     DataClass.vocab_transform_map()[ln], #Numericalization
                                                                     DataClass.tensor_transform) # Add SOS/EOS and create tensor
        return text_transform_map

    @staticmethod
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(DataClass.text_transform_map()['src'](src_sample))
            tgt_batch.append(DataClass.text_transform_map()['tgt'](tgt_sample))

        src_batch = pad_sequence(src_batch, padding_value=PolynomialVocab.PAD_INDEX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PolynomialVocab.PAD_INDEX)
        return src_batch, tgt_batch

    @staticmethod
    def train_dataloader():
        return DataLoader(dataset=DataClass.train_dataset, 
                          batch_size=config.Config.BATCH_SIZE, 
                          collate_fn=DataClass.collate_fn,
                          shuffle=True)

    @staticmethod
    def val_dataloader():
        return DataLoader(dataset=DataClass.val_dataset, 
                          batch_size=config.Config.BATCH_SIZE, 
                          collate_fn=DataClass.collate_fn,
                          shuffle=False)

    @staticmethod
    def test_dataloader():
        return DataLoader(dataset=DataClass.test_dataset, 
                          batch_size=1, 
                          collate_fn=DataClass.collate_fn,
                          shuffle=False)
