import os
import logging

class Config:

    # Dataset Paths
    DATASET_PATH = 'dataset_files/dataset.txt'
    TRAIN_DATASET_PATH = 'dataset_files/train.txt'
    VAL_DATASET_PATH = 'dataset_files/val.txt'
    TEST_DATASET_PATH = 'dataset_files/test.txt'
    
    RECOMPUTE_DATASET = False 
    """
        If RECOMPUTE_DATASET=False
            If the train.txt/ val.txt / test.txt files already exist, 
            skip generating the train/test/val dataset 

        If RECOMPUTE_DATASET=True
            Always Generate a new train.txt/ val.txt/ test.txt

        Note:
            1) Always re-train the model if RECOMPUTE_DATASET=True
               This is because the Train/Test/Val data is shuffled every
               time a new dataset is computed
            2) If your using a model trained prior, there could be overlap between
               the Train dataset the model was trained on, and new eval dataset generated
    """

    # Train/ Test/ Val split
    TRAIN_SPLIT_RATIO = 0.93
    VAL_SPLIT_RATIO = 0.01
    TEST_SPLIT_RATIO = 0.06
    assert TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO + TEST_SPLIT_RATIO == 1
    

    LOGGING_LEVEL = logging.INFO

    # Model Training Parameters
    BATCH_SIZE = 2048
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 256
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    MAX_LEN = 30
    NUM_EPOCHS = 30

    # Checkpoint Path
    CHECKPOINT_FOLDER = 'checkpoints/'
    NUM_LATEST_CHECKPOINT_SAVE = 4  # Number of Latest Checkpoints to Retain