from torch import Tensor
import torch
import torch.nn as nn
import os
import config
from time import time

DEVICE = config.Config.DEVICE

import logging
logging.basicConfig(level = config.Config.LOGGING_LEVEL)

class Trainer:

    def __init__(self, model: nn.Module, 
                       train_loader, 
                       val_loader, 
                       loss_fn, 
                       optimizer,
                       scheduler=None,
                       num_last_models_save: int=4,
                       init_params: bool=True):
        """`
        Module to encapsulate the Training Process

        Inputs:
            model: The model to be trained
            train_loader: The Train dataset Dataloader
            val_loader:   The Val dataset Dataloader
            loss_fn:      The Loss Function (Usually the Decoder Softmax CE Loss)
            optimizer:    The Optimizer
            num_last_models_save: Number of Latest Model Checkpoints to retain
                                  Older checkpoints will be deleted
            init_params: Initialize model parameters with Xavier Initialization
        """
        self.model = model
        if init_params:
            self.init_params() # Initialize Model Parameters

        self.model = self.model.to(DEVICE)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler

        self.num_last_models_save = num_last_models_save 

        self.train_losses = []
        self.val_losses   = []

        self.checkpoint_paths = []
        self.best_val_checkpoint = None

        self.checkpoint_base_path = config.Config.CHECKPOINT_FOLDER
        os.makedirs(self.checkpoint_base_path, exist_ok=True)

    def init_params(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def train_epoch(model, optimizer, train_loader, loss_fn):
        model.train()
        losses = 0
        
        for src, tgt, src_lens, tgt_lens in train_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_lens = src_lens.to(DEVICE)
            tgt_lens = tgt_lens.to(DEVICE)

            tgt_input = tgt[:, :-1] # remove EOS from input
            tgt_lens = tgt_lens - 1

            # logits = model(src, tgt_input)
            logits = model(src, tgt_input, src_lens, tgt_lens)

            optimizer.zero_grad()

            tgt_out = tgt[:, 1:] # Remove SOS from tgt

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        loss_avg = losses / len(train_loader)
        return loss_avg

    @staticmethod
    def val_epoch(model, val_loader, loss_fn):
        model.eval()
        losses = 0

        for src, tgt, src_lens, tgt_lens in val_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_lens = src_lens.to(DEVICE)
            tgt_lens = tgt_lens.to(DEVICE)

            tgt_input = tgt[:, :-1] # remove EOS from input
            tgt_lens = tgt_lens - 1

            # logits = model(src, tgt_input)
            logits = model(src, tgt_input, src_lens, tgt_lens)

            tgt_out = tgt[:, 1:] # Remove SOS from tgt
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        loss_avg = losses / len(val_loader)
        return loss_avg

    def delete_old_checkpoints(self):
        while len(self.checkpoint_paths) > self.num_last_models_save:
            pop_path = self.checkpoint_paths.pop(0)
            os.remove(pop_path)
            logging.info(f"Deleted Checkpoint {pop_path}")

    def save_checkpoint(self, model_state: dict, 
                        opt_state: dict,
                        epoch_num: int, 
                        val_loss: float, 
                        train_loss: float, 
                        train_loss_list: list = [],
                        val_loss_list: list = [],
                        custom_name: str=None):
        save_path = self.checkpoint_base_path + f'model_epoch_{epoch_num}.pth'
        if custom_name:
            save_path = self.checkpoint_base_path + f"{custom_name}.pth"

        torch.save({
                'epoch': epoch_num,
                'model_state_dict': model_state,
                'optimizer_state_dict': opt_state,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'train_loss_list': train_loss_list,
                'val_loss_list':val_loss_list
                },  save_path)
        logging.info(f"Saved Checkpoint: {save_path}")
        if not custom_name:
            # We append to the general checkpoint path, only when custom_name is None
            # custom_name is a valid input, only for saving the best "val checkpoint", which should not be added
            # to the general checkpoint_paths list
            self.checkpoint_paths.append(save_path)
        self.delete_old_checkpoints()
        return save_path

    def load_checkpoint(self, load_path):
        assert os.path.exists(load_path), logging.error(f"Checkpoint {load_path} does not exist")
  
        checkpoint = torch.load(load_path, map_location=torch.device(DEVICE))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_best_val_checkpoint(self):

        assert self.checkpoint_paths is not None, logging.error(f"No Checkpoint Paths Saved. Unable to save best Val checkpoint")

        curr_best_val_ckpt = None
        if self.best_val_checkpoint:
            curr_best_val_ckpt = torch.load(self.best_val_checkpoint)
    
        latest_ckpt = torch.load(self.checkpoint_paths[-1])

        if curr_best_val_ckpt is None or latest_ckpt['val_loss'] <= curr_best_val_ckpt['val_loss']:
            val_best_ckpt_path = self.save_checkpoint(model_state=latest_ckpt['model_state_dict'],
                                                      opt_state=latest_ckpt['optimizer_state_dict'],
                                                      epoch_num=latest_ckpt['epoch'], 
                                                      val_loss=latest_ckpt['val_loss'],
                                                      train_loss=latest_ckpt['train_loss'],
                                                      custom_name='val_best')
            self.best_val_checkpoint = val_best_ckpt_path
            
    def fit(self, num_epochs, pre_train_path=None):
        if pre_train_path:
          self.load_checkpoint(pre_train_path)
          logging.info(f"Loaded weights from {pre_train_path}")

        for epoch in range(1, num_epochs+1):
            
            start_time = time()
            train_loss = self.train_epoch(self.model, self.optimizer, self.train_loader, self.loss_fn)
            end_time = time()
            val_loss = self.val_epoch(self.model, self.val_loader, self.loss_fn)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.save_checkpoint(model_state=self.model.state_dict(),
                                 opt_state=self.optimizer.state_dict(),
                                 epoch_num=epoch, 
                                 val_loss=val_loss, 
                                 train_loss=train_loss,
                                 train_loss_list=self.train_losses,
                                 val_loss_list=self.val_losses)
            self.save_best_val_checkpoint()

            if self.scheduler:
                self.scheduler.step(val_loss)
            
            logging.info((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    

        