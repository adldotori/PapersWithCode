import os
import os.path as osp
import re
import pickle
import argparse
import random
from tqdm import tqdm

from core.module import Chat

def get_args():
    parser = argparse.ArgumentParser(description='TextCNN')
    parser.add_argument('--name', type=str, default='base', help='name of checkpoint file')
    parser.add_argument('--mode', type=str, choices=['rand','static','non-static','multichannel'], default='rand')

    # Model Configuration
    parser.add_argument('--embedding_dim', type=int, default=300, help='dimension of word vectors')
    parser.add_argument('--n_filters', type=int, default=100, help='number of feature maps')
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5], help='filter window size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate of final layer')
    parser.add_argument('--l2_constraint', type=int, default=3, help='reference value of weight vectors to rescale')

    # Path Configuration
    parser.add_argument('--ck_path', type=str, default='checkpoint/', help='checkpoint path')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    Chat(args).talk()