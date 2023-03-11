import torch
from trainer import Trainer
from config import Config
from data_utils.dataset import DataClass
from data_utils.polynomial_vocab import PolynomialVocab

from model.transformer import Seq2SeqTransformer

src_vocab, tgt_vocb = DataClass.src_vocab_cls, DataClass.tgt_vocab_cls


SRC_VOCAB_SIZE = src_vocab.token_count
TGT_VOCAB_SIZE = tgt_vocb.token_count

SRC_PAD_INDEX = PolynomialVocab.PAD_INDEX
TGT_PAD_INDEX = PolynomialVocab.PAD_INDEX

dropout = 0.1
model = Seq2SeqTransformer(Config.NUM_ENCODER_LAYERS, Config.NUM_DECODER_LAYERS, Config.EMB_SIZE, 
                          Config.NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, Config.FFN_HID_DIM, dropout,
                          SRC_PAD_INDEX, TGT_PAD_INDEX)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=DataClass.src_vocab_cls.PAD_INDEX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, threshold=0.001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08, verbose=True)


trainer_cls = Trainer(model=model,
                      train_loader=DataClass.train_dataloader(),
                      val_loader=DataClass.val_dataloader(),
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      num_last_models_save=Config.NUM_LATEST_CHECKPOINT_SAVE,
                      init_params=True)

# trainer_cls.fit(num_epochs=Config.NUM_EPOCHS, pre_train_path='/home/debayan/h3/state-spaces/machine_translation/poly_expansion_seq2seq/checkpoints/model_epoch_11.pth')
trainer_cls.fit(num_epochs=Config.NUM_EPOCHS)