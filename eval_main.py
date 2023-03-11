from operator import mod
from eval import Evaluation
import torch
from trainer import Trainer
from config import Config
from data_utils.dataset import DataClass
from data_utils.polynomial_vocab import PolynomialVocab
from model.transformer import Seq2SeqTransformer

import logging
logging.basicConfig(level = Config.LOGGING_LEVEL)

src_vocab, tgt_vocb = DataClass.src_vocab_cls, DataClass.tgt_vocab_cls

DEVICE = Config.DEVICE

SRC_VOCAB_SIZE = src_vocab.token_count
TGT_VOCAB_SIZE = tgt_vocb.token_count

SRC_PAD_INDEX = PolynomialVocab.PAD_INDEX
TGT_PAD_INDEX = PolynomialVocab.PAD_INDEX

dropout = 0.1
model = Seq2SeqTransformer(Config.NUM_ENCODER_LAYERS, Config.NUM_DECODER_LAYERS, Config.EMB_SIZE, 
                          Config.NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, Config.FFN_HID_DIM, dropout,
                          SRC_PAD_INDEX, TGT_PAD_INDEX)

model = model.to(DEVICE)

MODEL_PATH = 'checkpoints/model_epoch_20.pth'
model = Evaluation.load_checkpoint(model=model, load_path=MODEL_PATH, device=DEVICE)

test_acc = Evaluation.test_acc(model=model)

logging.info(f"Test Accuracy {test_acc}")
logging.info(f"#Trainable Parameters, {Evaluation.num_trainable_parms(model=model)}")
