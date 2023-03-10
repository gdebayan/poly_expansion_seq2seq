from torch import Tensor
import torch
import torch.nn as nn
import os
from model.mask import MaskUtils
import config
from tqdm import tqdm
from data_utils.dataset import DataClass 

DEVICE = 'cpu'

import logging
logging.basicConfig(level = config.Config.LOGGING_LEVEL)



class Evaluation:

    @staticmethod
    def load_checkpoint(model, load_path, device):
        """Utility Function to load Model Weights"""
        assert os.path.exists(load_path), logging.error(f"Checkpoint {load_path} does not exist")
  
        checkpoint = torch.load(load_path, map_location=torch.device(device))
        logging.info(f"Checkpoint Val Loss {checkpoint['val_loss']}")
        model.load_state_dict(checkpoint['model_state_dict'])        
        return model

    @staticmethod
    def num_trainable_parms(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # function to generate output sequence using greedy algorithm 
    @staticmethod
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = model.encode(src, src_mask)

        # (Batch Size, Seq Len) or (Seq Len, Batch Size)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len-1):

            memory = memory.to(DEVICE)
            
            seq_len_current = ys.size(1) if config.Config.BATCH_FIRST else ys.size(0)

            tgt_mask = (MaskUtils.generate_square_subsequent_mask(seq_len_current)
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)

            out = out if config.Config.BATCH_FIRST else out.transpose(0, 1)

            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            dim_append = 1 if config.Config.BATCH_FIRST else 0
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=dim_append)
                            
            if next_word == DataClass.src_vocab_cls.EOS_INDEX:
                break
        return ys


    # actual function to translate input polynomial into target expanded polynomial
    @staticmethod
    def expand_polynomial(model: torch.nn.Module, src_sentence: str):
        model.eval()
        src = DataClass.text_transform_map()['src'](src_sentence)
        src = src.view(1, -1) if config.Config.BATCH_FIRST else src.view(-1, 1)

        num_tokens = src.shape[1] if config.Config.BATCH_FIRST else src.shape[0]
        
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = Evaluation.greedy_decode(
            model,  src, src_mask, max_len=config.Config.MAX_LEN, start_symbol=DataClass.src_vocab_cls.SOS_INDEX).flatten()
        tgt_tokens = tgt_tokens[1:-1]
        op = [DataClass.tgt_vocab_cls.index_to_token[tok.item()] for tok in tgt_tokens]
        return ''.join(op)

    @staticmethod
    def test_acc(model):
        num_correct = 0
        tot = 0
        num_samples = DataClass.test_dataset.__len__()
        batch_bar = tqdm(total=num_samples, dynamic_ncols=True, leave=False, position=0, desc='Test Acc')

        for src, tgt in DataClass.test_dataset:
            op = Evaluation.expand_polynomial(model, src)
            if op == tgt:
                num_correct += 1

            tot += 1
            
            acc = num_correct/tot

            batch_bar.set_postfix(
                acc="{:.04f}%".format(acc),
                num_correct=f"{num_correct}",
                total_samples=f"{tot}")
            batch_bar.update() # Update tqdm bar  

        batch_bar.close() 
        return num_correct/tot