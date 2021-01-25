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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'])
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--step', type=int, default=1000000)
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--kor_token_path', type=str, default='kor_token.pkl')
    parser.add_argument('--eng_token_path', type=str, default='eng_token.pkl')
    parser.add_argument('--kor_vocab_path', type=str, default='kor.pkl')
    parser.add_argument('--eng_vocab_path', type=str, default='eng.pkl')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.infer()