import logging
from typing import List, Tuple
import random
import sys
import os

sys.path.insert(0, '../')
import config

logging.basicConfig(level = config.Config.LOGGING_LEVEL)

class DatasetUtils:

    @staticmethod
    def read_raw_dataset(dataset_path: str) -> List[str]:
        """Reads the raw dataset.txt file, and return each line as a list
        """
        with open(dataset_path) as f:
            data = [line.strip() for line in f]
        return data

    @staticmethod
    def write_raw_dataset(data_list: List[str], save_path: str) -> None:
        """Writes a List of Strings to a new text file."""
        with open(save_path, 'w') as f:
            for line in data_list:
                f.write(f"{line}\n")

    @staticmethod
    def read_dataset_ip_op_pairs(dataset_path: str) -> List[List[str]]:
        """Reads a dataset file (eg. train/test/val .txt files) and returns a 
            list of (input_polynomial, expanded_polynomial).
        """
        with open(dataset_path) as fi:
            ip_op_pairs = [line.strip().split("=") for line in fi]
        return ip_op_pairs

    @staticmethod
    def train_val_test_split(train_ratio: float, 
                             val_ratio: float, 
                             test_ratio: float, 
                             skip_load_if_exist=True) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """
        Splits the input dataset.txt, into train.txt, val.txt, test.txt based on 'train_ratio'.
        The 'Val' and 'Test' samples have the same split-ratio.

        Further, processes each sample and returns the (ip_polynomial, op_polynomial) pairs 
        """
        assert os.path.exists(config.Config.DATASET_PATH), logging.error(f"{config.Config.DATASET_PATH} does not exist")
        assert train_ratio + val_ratio + test_ratio == 1, logging.error(f"Expected train+val+test ratio to sum to one")

        if skip_load_if_exist is True \
             and os.path.exists(config.Config.TRAIN_DATASET_PATH) \
             and os.path.exists(config.Config.VAL_DATASET_PATH) \
             and os.path.exists(config.Config.TEST_DATASET_PATH):

            logging.info(f"Train/Test/Val files present. Skipping processing again as skip_load_if_exist=True")
            return DatasetUtils.read_dataset_ip_op_pairs(dataset_path=config.Config.TRAIN_DATASET_PATH), \
                   DatasetUtils.read_dataset_ip_op_pairs(dataset_path=config.Config.VAL_DATASET_PATH), \
                   DatasetUtils.read_dataset_ip_op_pairs(dataset_path=config.Config.TEST_DATASET_PATH)

        full_datset = DatasetUtils.read_raw_dataset(dataset_path=config.Config.DATASET_PATH)
        random.shuffle(full_datset)
        split = int(train_ratio * len(full_datset))
        train_data, val_test_data = full_datset[:split], full_datset[split:]

        split = int(val_ratio * len(full_datset))
        val_data, test_data = val_test_data[0:split], val_test_data[split:]

        logging.info(f"#Train Samples: {len(train_data)}, #Val Samples: {len(val_data)}, #Test Samples: {len(test_data)}")

        # Save dataset for future reference
        DatasetUtils.write_raw_dataset(data_list=train_data, save_path=config.Config.TRAIN_DATASET_PATH)
        DatasetUtils.write_raw_dataset(data_list=val_data, save_path=config.Config.VAL_DATASET_PATH)
        DatasetUtils.write_raw_dataset(data_list=test_data, save_path=config.Config.TEST_DATASET_PATH)

        return DatasetUtils.read_dataset_ip_op_pairs(dataset_path=config.Config.TRAIN_DATASET_PATH), \
               DatasetUtils.read_dataset_ip_op_pairs(dataset_path=config.Config.VAL_DATASET_PATH), \
               DatasetUtils.read_dataset_ip_op_pairs(dataset_path=config.Config.TEST_DATASET_PATH)