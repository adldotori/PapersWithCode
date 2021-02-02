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

from gensim.models import KeyedVectors

from .utils import *


class Vocabulary(object):
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self._add(self.PAD_TOKEN)
        self._add(self.SOS_TOKEN)
        self._add(self.EOS_TOKEN)
        self._add(self.UNK_TOKEN)

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
        super().__init__()

        self.args = args
        
        # make directory if not exist data path
        if not osp.isdir(args.path): 
            os.makedirs(args.path, exist_ok=True)

        print('>>> Vocab building...')
        self.vocab = Vocabulary()
        self._build_vocab()
        print('>>> Dataset loading...')
        self.dataset = self._call_data()

        print('>>> Word2Vec loading...')
        word2vec = KeyedVectors.load_word2vec_format(osp.join(args.path, 'GoogleNews-vectors-negative300.bin.gz'), binary=True, limit=500000)   
        print('>>> Word2Vec loaded')
        self.pretrained_embeddings = self._build_embeddings(word2vec)

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

    def _build_embeddings(self, model):
        """
        adapt pretrained vector to embedding vector
        Args:
            model(Word2VecKeyedVectors)
        Returns:
            (tensor)
        """
        new_embeddings = []
        for (word, idx) in self.vocab.word2idx.items():
            try:
                new_embeddings.append(model[word])
            except KeyError:
                value = randint(0, len(model.index2word))
                a = np.var(model[model.index2word[value]])
                new_embeddings.append(np.random.uniform(-a, a, 300))

        return torch.tensor(new_embeddings)

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

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index][0]), torch.tensor(float(self.dataset[index][1]))

    def __len__(self):
        return len(self.dataset)


class MR(CustomDataset):
    """
    <MR dataset>
    Can build vocabulary file with MR datset
    Call MR Data in order 
    """
    def _build_vocab(self):
        if osp.isfile(osp.join(self.args.ck_path, 'mr.p')):
            with open(osp.join(self.args.ck_path, 'mr.p'), 'rb') as f:
                self.vocab = pickle.load(f)

        else:
            all_tokens = []
            with open(osp.join(self.args.path, 'rt-polarity.pos'),'rb') as f:
                pos = f.readlines()
                for i in pos:
                    tokens = preprocess(i.decode('latin1'))
                    all_tokens += tokens

            with open(osp.join(self.args.path, 'rt-polarity.neg'),'rb') as f:
                neg = f.readlines()
                for i in neg:
                    tokens = preprocess(i.decode('latin1'))
                    all_tokens += tokens
            self.vocab.build_vocab(all_tokens)
            with open(osp.join(self.args.ck_path, 'mr.p'), 'wb') as f:
                pickle.dump(self.vocab, f)

    def _call_data(self):
        data = []
        with open(osp.join(self.args.path, 'rt-polarity.pos'),'rb') as f:
            pos = f.readlines()
            for i in pos:
                tokens = preprocess(i.decode('latin1'))
                data.append(([self.vocab(i) for i in tokens], 1))

        with open(osp.join(self.args.path, 'rt-polarity.neg'),'rb') as f:
            neg = f.readlines()
            for i in neg:
                tokens = preprocess(i.decode('latin1'))
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