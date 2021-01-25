import os
import os.path as osp
import random
import re
import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from gensim.models import KeyedVectors
from gensim.models import Word2Vec 


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self._add('<pad>')
        self._add('<sos>')
        self._add('<eos>')
        self._add('<unk>')

    def _add(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(zself.word2idx)

    def build_vocab(self, tokens):
        for i in tokens:
            self._add(i)

class CustomDataset(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.args = args
        self.vocab = Vocabulary()
        self._build_vocab()
        print('>>> Dataset loading...')
        self.dataset = self._call_data()
        print('>>> Word2Vec loading...')
        # self.model = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)   

    def collate_fn(self, data):
        """
        add padding for text of various lengths
        Args:
            [(text(tensor), label(tensor)), ...]
        Returns:
            text(tensor), label(tensor)
        """
        text, label = zip(*data)
        text = pad_sequence(text, batch_first=True, padding_value=self.vocab('<pad>'))
        return text, label

    def _build_vocab(self):
        """
        call or build vocabulary file
        this method must be overridden
        Args:
            None
        Returns:
            None
        """
        pass

    def _call_data(self):
        """
        return cleaned and tokenized data
        this method must be overridden
        Args:
            None
        Returns:
            [(list, int)]
        """
        pass

    def _clean_str(self, string):
        """
        clean string from the input sentence to normalize it
        Args:
            string(str)
        Returns:
            (str)
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return self._tokenize(string.strip())

    def _tokenize(self, string):
        """
        Divide string to token
        Args:
            string(str)
        Returns:
            [str]
        """
        return string.split(' ')

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index][0]), torch.tensor(self.dataset[index][1])

    def __len__(self):
        return len(self.dataset)


class MR(CustomDataset):
    def _build_vocab(self):
        if osp.isfile(osp.join(self.args.path, 'mr.p')):
            with open(osp.join(self.args.path, 'mr.p'), 'rb') as f:
                self.vocab = pickle.load(f)

        else:
            all_tokens = []
            with open(osp.join(self.args.path, 'rt-polarity.pos'),'rb') as f:
                pos = f.readlines()
                for i in pos:
                    tokens = self._clean_str(i.decode('latin1'))
                    all_tokens += tokens

            with open(osp.join(self.args.path, 'rt-polarity.neg'),'rb') as f:
                neg = f.readlines()
                for i in neg:
                    tokens = self._clean_str(i.decode('latin1'))
                    all_tokens += tokens
            self.vocab.build_vocab(all_tokens)
            with open(osp.join(self.args.path, 'mr.p'), 'wb') as f:
                pickle.dump(self.vocab, f)

    def _call_data(self):
        data = []
        with open(osp.join(self.args.path, 'rt-polarity.pos'),'rb') as f:
            pos = f.readlines()
            for i in pos:
                tokens = self._clean_str(i.decode('latin1'))
                data.append(([self.vocab(i) for i in tokens], 1))

        with open(osp.join(self.args.path, 'rt-polarity.neg'),'rb') as f:
            neg = f.readlines()
            for i in neg:
                tokens = self._clean_str(i.decode('latin1'))
                data.append(([self.vocab(i) for i in tokens], 0))

        return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-p', '--path', type=str, default='../data')
    args = parser.parse_args()

    dataset = MR(args)
    data_loader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)

    for i in data_loader:
        print(i)
        break 