import torch
from trainer import Trainer
from config import Config
from data_utils.dataset import DataClass
from data_utils.polynomial_vocab import PolynomialVocab

from model.transformer_s4_enc_dec import S4EncDec

DEVICE = Config.DEVICE

src_vocab, tgt_vocb = DataClass.src_vocab_cls, DataClass.tgt_vocab_cls


SRC_VOCAB_SIZE = src_vocab.token_count
TGT_VOCAB_SIZE = tgt_vocb.token_count

SRC_PAD_INDEX = PolynomialVocab.PAD_INDEX
TGT_PAD_INDEX = PolynomialVocab.PAD_INDEX

dropout = 0.1
model = S4EncDec(Config.NUM_ENCODER_LAYERS, Config.NUM_DECODER_LAYERS, Config.EMB_SIZE, 
                          Config.NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, Config.FFN_HID_DIM, dropout,
                          SRC_PAD_INDEX, TGT_PAD_INDEX)
model = model.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=DataClass.src_vocab_cls.PAD_INDEX)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, threshold=0.001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08, verbose=True)


def setup_optimizer(model, lr, weight_decay, patience):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically 
    require a smaller learning rate (typically 0.001), with no weight decay. 
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01) 
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())
    
    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(
        params, 
        lr=lr, 
        weight_decay=weight_decay,
    )

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in set(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    
    # Print optimizer info 
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

optimizer, scheduler = setup_optimizer(
    model, lr=0.0001, weight_decay=0.01, patience=10
)

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