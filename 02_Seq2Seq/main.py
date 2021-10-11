import os
import os.path as osp
import re
import pickle
import argparse
import random
from tqdm import tqdm

from core.module import Trainer
from core.dataloader import *

def get_args():
    parser = argparse.ArgumentParser(description='Seq2Seq')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--ck_path', type=str, default='checkpoint')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--vocab_path', type=str, default='vocab')
    parser.add_argument('--beam_size', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()
