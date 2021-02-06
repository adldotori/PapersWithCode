import os
import os.path as osp
import random
import pickle
import argparse
import numpy as np
from random import randint

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

    def build_vocab(self, tokens):
        for i in tokens:
            self._add(i)


class FrEnCorpus(Dataset):
    """
    <FrEnCorpus dataset>
    Can build vocabulary file with FrEnCorpus datset
    Call FrEnCorpus Data in order 
    """
    def __init__(self, args, num_src_words, num_trg_words):
        super().__init__()
        self.args = args

        # make directory if not exist data path
        if not osp.isdir(args.path): 
            os.makedirs(args.path, exist_ok=True)

        FR_FILE_NAME = 'giga-fren.release2.fixed.fr'
        EN_FILE_NAME = 'giga-fren.release2.fixed.en'

        print('>>> Vocab building...')
        fr = self._build_vocab(FR_FILE_NAME, num_src_words)
        en = self._build_vocab(EN_FILE_NAME, num_trg_words)
        print(f'>>> Vocab Length : {len(fr)} / {len(en)}')

        print('>>> Dataset loading...')
        self.dataset = self._call_data()

    def collate_fn(self, data):
        """
        add padding for text of various lengths
        Args:
            [(text(tensor), label(tensor)), ...]
        Returns:
            tensor, tensor : text, label
        """
        text, label = zip(*data)
        text = pad_sequence(text, batch_first=True, padding_value=self.vocab('<pad>'))
        label = torch.stack(label, 0)
        return text, label

    def _build_vocab(self, file_name, num_words, max_size=1e9):
        """
        call or build vocabulary file
        this method must be overridden
        Args:
            None
        Returns:
            None
        """
        
        # if osp.isfile(osp.join(self.args.ck_path, file_name)): # if exist vocab file
        #     with open(osp.join(self.args.ck_path, file_name), 'rb') as f:
        #         return pickle.load(f)
 
        if osp.isfile(osp.join(self.args.path, file_name)): # only exist data file, not vocab file
            vocab = Vocabulary()
            with open(osp.join(self.args.path, file_name), 'r') as f:
                text = True
                cnt = 0
                vocab_dict = {}
                while text and cnt < max_size:
                    text = preprocess(f.readline())
                    for i in text:
                        if i in vocab_dict.keys():
                            vocab_dict[i] += 1
                        else:
                            vocab_dict[i] = 1
                    cnt += 1

            for i, (key, value) in enumerate(sorted(vocab_dict.items(), key=lambda item: -item[1])):
                if i == num_words:
                    break
                vocab.add(key)

            with open(osp.join(self.args.ck_path, file_name+'.vocab'), 'wb') as f:
                pickle.dump(vocab, f)

            return vocab

        else:
            print('>>> There aren\'t data files.')

    def _call_data(self):
        """
        return cleaned and tokenized data
        this method must be overridden
        Args:
            None
        Returns:
            [(list, int)]
        """
        return data

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index][0]), torch.tensor(float(self.dataset[index][1]))

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-p', '--path', type=str, default='../data')
    parser.add_argument('--ck_path', type=str, default='../checkpoint')
    args = parser.parse_args()

    dataset = FrEnCorpus(args, 160000, 80000)
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)

    for i in data_loader:
        print(i)
        break 