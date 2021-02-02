import os
import os.path as osp
import re
import pickle
import argparse
import random
from tqdm import tqdm
import numpy as np
import functools

import torch
import torch.nn as nn
import torch.optim as optim

from .model import *
from .dataloader import *
from .utils import *

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer():
    def __init__(self, args):
        self.args = args

        # data
        dataset = MR(args)
        self.collate_fn = dataset.collate_fn
        train_size = int(1 / args.cv_num * len(dataset))
        self.dataset_list = torch.utils.data.random_split(dataset, 
                                            [train_size for i in range(args.cv_num - 1)] +\
                                            [len(dataset) - (args.cv_num - 1) * train_size])

        # arguments, loss
        self.vocab_size = dataset.vocab.idx
        self.pad_idx = dataset.vocab(dataset.vocab.PAD_TOKEN)
        self.embeddings = dataset.pretrained_embeddings
        self.criterion = nn.BCEWithLogitsLoss().to(device)

        # make directory if not exist data path
        if not osp.isdir(args.ck_path): 
            os.makedirs(args.ck_path, exist_ok=True)

    def train(self):        
        best_valid_loss = 1e9
        all_valid_loss, all_valid_acc = 0, 0

        for i in range(self.args.cv_num):
            model = TextCNN(self.vocab_size, self.pad_idx, self.args).to(device)

            # model variations
            if self.args.mode == 'static':
                model.embedding.weight.data.copy_(self.embeddings)
                model.embedding.weight.requires_grad = False
            elif self.args.mode == 'non-static':
                model.embedding.weight.data.copy_(self.embeddings)
            elif self.args.mode == 'multichannel':
                model.embedding.weight.data.copy_(self.embeddings)
                model.embedding2.weight.data.copy_(self.embeddings)
                model.embedding2.weight.requires_grad = False

            optimizer = optim.Adam(model.parameters())
            model.train()

            # generate train dataset
            print(f'>>> {i+1}th dataset is testset')
            dataset = self.dataset_list.copy()
            del dataset[i]
            dataset = functools.reduce(lambda x, y: x + y, dataset)
        
            data_loader = DataLoader(dataset=dataset, 
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      collate_fn=self.collate_fn)

            for epoch in range(self.args.epochs):

                pbar = tqdm(data_loader)

                for batch in pbar:
                    text, label = batch
                    text = text.to(device)
                    label = label.to(device)

                    predictions = model(text)
                    loss = self.criterion(predictions, label)
                    acc = self._binary_accuracy(predictions, label)                    
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_description((f"loss : {loss.item():.4f}, acc : {acc.item():.4f}"))

            valid_loss, valid_acc = self.evaluate(model, i)
            all_valid_loss += valid_loss.item()
            all_valid_acc += valid_acc.item()
            print(f'valid loss : {valid_loss.item():.3f}, valid acc : {valid_acc.item():.3f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 
                            osp.join(self.args.ck_path, f'{self.args.name}_{self.args.mode}_best.pt'))

            if not self.args.cv:
                return
        
        print(f'\nFinal loss : {all_valid_loss / self.args.cv_num:.3f}'+
                 f'\nFinal acc : {all_valid_acc / self.args.cv_num:.3f}')

    def evaluate(self, model, cnt):
        """
        get loss, accuracy about test dataset
        Args:
            model(CNN) : trained model
            cnt(int) : test dataset's number in dataset
        Returns:
            loss(float), acc(float)
        """
        loss, acc = 0, 0

        model.eval()

        data_loader = DataLoader(dataset=self.dataset_list[cnt], 
                                 batch_size=self.args.batch_size,
                                 shuffle=True,
                                 collate_fn=self.collate_fn)

        with torch.no_grad():
            for batch in data_loader:
                text, label = batch
                text = text.to(device)
                label = label.to(device)
                predictions = model(text)
                
                loss += self.criterion(predictions, label)
                
                acc += self._binary_accuracy(predictions, label)

        loss /= len(data_loader)
        acc /= len(data_loader)

        return loss, acc

    def _binary_accuracy(self, preds, y):
        """
        get accuracy when given the correct answer and prediction
        Args:
            preds(tensor) : prediction value
            y(tensor) : real data's value
        Returns:
            acc(float)
        """
        rounded_preds = torch.round(preds)
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc
            
    def _sample_data(self, loader):
        while True:
            for batch in loader:
                yield batch


class Chat():
    def __init__(self, args):
        # vocabulary load
        with open(osp.join(args.ck_path, 'mr.p'), 'rb') as f:
            self.vocab = pickle.load(f)
        self.vocab_size = self.vocab.idx
        self.pad_idx = self.vocab(self.vocab.PAD_TOKEN)

        self.min_length = args.filter_sizes[-1]

        # model load
        self.model = TextCNN(self.vocab_size, self.pad_idx, args).to(device)
        self.model.load_state_dict(torch.load(osp.join(args.ck_path, f'{args.name}_{args.mode}_best.pt')))
        self.model.eval()
    
    def talk(self):
        """
        Determines whether a given sentence is positive or negative
        """ 
        while True:
            input_str = preprocess(input('>>> ')) # clean and tokenize sentence
            input_str = [self.vocab(i) for i in input_str] # change vocab to idx with vocabulary
            if len(input_str) < self.min_length: # if a give sentence is too short, add padding
                input_str += [self.pad_idx for i in range(self.min_length - len(input_str))]

            input_str = torch.tensor(input_str).to(device).unsqueeze(0)
            predictions = self.model(input_str)
            print(f'{predictions.item():.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--mode', type=str, choices=['rand','static','non-static','multichannel'], default='rand')
    parser.add_argument('--ck_path', type=str, default='../checkpoint')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cv_num', type=int, default=10)
    parser.add_argument('--l2_constraint', type=int, default=3)
    parser.add_argument("--cv", type=bool)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

    chat = Chat(args)