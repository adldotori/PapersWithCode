import os
import os.path as osp
import pandas as pd
import re
import pickle
import argparse
import random
from tqdm import tqdm

from core.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description='Executer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--ck_path', type=str, default='checkpoint/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pad_idx', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()