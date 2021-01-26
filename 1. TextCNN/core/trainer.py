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
        self.criterion = nn.BCEWithLogitsLoss().to(device)

        # make directory if not exist data path
        if not osp.isdir(args.ck_path): 
            os.makedirs(args.ck_path, exist_ok=True)

    def train(self):        
        best_valid_loss = 1e9
        all_valid_loss, all_valid_acc = 0, 0

        for i in range(self.args.cv_num):
            model = CNN(self.vocab_size, self.pad_idx, self.args).to(device)
            optimizer = optim.Adadelta(model.parameters())
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

                    predictions = model(text).squeeze(1)
                    loss = self.criterion(predictions, label)
                    acc = self._binary_accuracy(predictions, label)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        if torch.norm(model.fc.weight) > self.args.l2_constraint:
                            model.fc.weight *= self.args.l2_constraint / torch.norm(model.fc.weight)

                    pbar.set_description(
                        (
                            f"loss : {loss.item():.4f}, acc : {acc.item():.4f}"
                        )
                    )

            valid_loss, valid_acc = self.evaluate(model, i)
            all_valid_loss += valid_loss.item()
            all_valid_acc += valid_acc.item()
            print(f'valid loss : {valid_loss.item():.3f}, valid acc : {valid_acc.item():.3f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 
                            osp.join(self.args.ck_path, f'{self.args.name}_best.pt'))

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
                predictions = model(text).squeeze(1)
                
                loss += self.criterion(predictions, label)
                
                acc += self._binary_accuracy(predictions, label)

        loss /= len(data_loader)
        acc /= len(data_loader)

        return loss, acc

    def _binary_accuracy(self, preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc
            
    def _sample_data(self, loader):
        while True:
            for batch in loader:
                yield batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--ck_path', type=str, default='../checkpoint')
    parser.add_argument('--epochs', type=int, default=5)
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