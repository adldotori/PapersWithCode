import os
import os.path as osp
import re
import pickle
import argparse
import random
from tqdm import tqdm

from core.module import Trainer

def get_args():
    parser = argparse.ArgumentParser(description='Seq2Seq')

    # Training Configuration

    # Test Configuration

    # Model Configuration
    
    # Path Configuration

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()