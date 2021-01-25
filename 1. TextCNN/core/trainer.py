import os
import os.path as osp
import pandas as pd
import re
import pickle
import argparse
import random
from tqdm import tqdm
import numpy as np

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
        train_size = int(0.9 * len(dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        self.train_loader = DataLoader(dataset=train_dataset, 
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=dataset.collate_fn)
        self.test_loader = DataLoader(dataset=test_dataset, 
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      collate_fn=dataset.collate_fn)
        # self.data_loader = self._sample_data(self.data_loader_)

        # model, optimizer, loss
        self.model = CNN1d(dataset.vocab.idx, args).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.BCEWithLogitsLoss().to(device)

        # make directory if not exist data path
        if not osp.isdir(args.ck_path): 
            os.makedirs(args.ck_path, exist_ok=True)

    def train(self):        
        best_valid_loss = 1e9

        self.model.train()
        
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.train_loader)

            for i, batch in enumerate(pbar):
                text, label = batch
                text = text.to(device)
                label = label.to(device)

                predictions = self.model(text).squeeze(1)
                loss = self.criterion(predictions, label)
                
                acc = self._binary_accuracy(predictions, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(
                    (
                        f"loss : {loss.item():.4f}, acc : {acc.item():.4f}"
                    )
                )

            valid_loss, valid_acc = self.evaluate()
            print(f'valid loss : {valid_loss.item():.3f}, valid acc : {valid_acc.item():.3f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 
                            osp.join(self.args.ck_path, f'{self.args.name}_best.pt'))

    def evaluate(self):
        loss, acc = 0, 0

        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                text, label = batch
                text = text.to(device)
                label = label.to(device)
                predictions = self.model(text).squeeze(1)
                
                loss += self.criterion(predictions, label)
                
                acc += self._binary_accuracy(predictions, label)

        loss /= len(self.test_loader)
        acc /= len(self.test_loader)

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
    parser.add_argument('--pad_idx', type=int, default=0)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()