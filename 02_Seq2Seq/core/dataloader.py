import os
import os.path as osp
import random
import pickle
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import *


class Vocabulary(object):
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add(self.PAD_TOKEN)
        self.add(self.SOS_TOKEN)
        self.add(self.EOS_TOKEN)
        self.add(self.UNK_TOKEN)

    def add(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class FrEnCorpus(Dataset):
    """
    <FrEnCorpus dataset>
    Can build vocabulary file with FrEnCorpus datset
    Call FrEnCorpus Data in order 
    """
    SRC_FILE_NAME = 'giga-fren.release2.fixed.fr'
    TRG_FILE_NAME = 'giga-fren.release2.fixed.en'
    TEST_SIZE = 1000

    def __init__(self, args, num_src_words, num_trg_words, train=True):
        super().__init__()
        self.args = args

        if train:
            print('>>> Vocab building...')
            self.src_vocab = self._build_vocab(self.SRC_FILE_NAME, num_src_words)
            self.trg_vocab = self._build_vocab(self.TRG_FILE_NAME, num_trg_words)
            print(f'>>> Vocab Length : {len(self.src_vocab)} / {len(self.trg_vocab)}')

        print('>>> Dataset loading...')
        self.src, self.trg = self._read_data(self.SRC_FILE_NAME, self.TRG_FILE_NAME, train)

    def collate_fn(self, data):
        """
        add padding for text of various lengths
        Args:
            [(text(tensor), label(tensor)), ...]
        Returns:
            tensor, tensor : text, label
        """
        src_text, trg_text = zip(*data)
        src_text = pad_sequence(src_text, batch_first=True, padding_value=self.src_vocab('<pad>'))
        trg_text = pad_sequence(trg_text, batch_first=True, padding_value=self.trg_vocab('<pad>'))
        return src_text, trg_text

    def _build_vocab(self, file_name, num_words, train_size=None, test_size=1000):
        """
        call or build vocabulary file
        this method must be overridden
        Args:
            file_name(str): 
            num_words(int):
            max_size(int):
        Returns:
            Vocabulary
        """
        
        if osp.isfile(osp.join(self.args.vocab_path, file_name + '.vocab')): # load vocab file if exists vocab file
            with open(osp.join(self.args.vocab_path, file_name + '.vocab'), 'rb') as f:
                return pickle.load(f)

        if osp.isfile(osp.join(self.args.path, file_name)):
            with open(osp.join(self.args.path, file_name), 'r') as f:

                texts = f.readlines()[:-self.TEST_SIZE]

                vocab_dict = {}
                for i, text in enumerate(texts):
                    text = preprocess(text)
                    for i in text:
                        if i in vocab_dict.keys():
                            vocab_dict[i] += 1
                        else:
                            vocab_dict[i] = 1

            vocab = Vocabulary()
            for i, (key, value) in enumerate(sorted(vocab_dict.items(), key=lambda item: -item[1])):
                if i == num_words:
                    break
                vocab.add(key)

            with open(osp.join(self.args.ck_path, file_name+'.vocab'), 'wb') as f:
                pickle.dump(vocab, f)

            return vocab

        else:
            print('>>> There aren\'t data files.')
            exit()

    def _read_data(self, src_file_name, trg_file_name, train):
        """
        return src data and trg dataa
        Args:
            src_file_name(str), trg_file_name(str), train(bool)
        Returns:
            [str], [str]
        """
        if osp.isfile(osp.join(self.args.path, src_file_name)) and osp.isfile(osp.join(self.args.path, trg_file_name)): # only exist data file, not vocab file
            with open(osp.join(self.args.path, src_file_name), 'r') as f1, open(osp.join(self.args.path, trg_file_name), 'r') as f2:
                src = f1.readlines()
                trg = f2.readlines()
                if train:
                    src = src[:-self.TEST_SIZE]
                    trg = trg[:-self.TEST_SIZE]
                else:
                    src = src[-self.TEST_SIZE:]
                    trg = trg[-self.TEST_SIZE:]

            return src, trg

        else:
            print('>>> There aren\'t data files.')
            exit()

    def __getitem__(self, index):
        src, trg = self.src[index], self.trg[index]
        src, trg = preprocess(src), preprocess(trg)
        src, trg = [self.src_vocab(i) for i in src], [self.trg_vocab(i) for i in trg]

        return torch.tensor(src), torch.tensor(trg)

    def __len__(self):
        return len(self.src)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-p', '--path', type=str, default='../data')
    parser.add_argument('--vocab_path', type=str, default='../vocab')
    args = parser.parse_args()

    train_dataset = FrEnCorpus(args, 160000, 80000, True)
    test_dataset = FrEnCorpus(args, 160000, 80000, False)
    
    train_data_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.batch_size,
                            collate_fn=train_dataset.collate_fn)
    test_data_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size,
                            collate_fn=test_dataset.collate_fn)

    for i in train_data_loader:
        print(i)
        break 