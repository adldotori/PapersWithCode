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

from nltk.translate.bleu_score import sentence_bleu

from model import *
from dataloader import *
from utils import *

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer():
    NUM_SRC_WORDS = 160000
    NUM_TRG_WORDS = 80000

    def __init__(self, args):
        self.args = args

        # data
        train_dataset = FrEnCorpus(args, self.NUM_SRC_WORDS, self.NUM_TRG_WORDS, True)
        test_dataset = FrEnCorpus(args, self.NUM_SRC_WORDS, self.NUM_TRG_WORDS, False)
        self.vocab = train_dataset.trg_vocab

        self.train_data_loader = DataLoader(dataset=train_dataset, 
                                batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
        self.test_data_loader = DataLoader(dataset=test_dataset, 
                                batch_size=1,
                                collate_fn=test_dataset.collate_fn)

        # model
        self.model = Seq2Seq(self.NUM_SRC_WORDS, self.NUM_TRG_WORDS).to(device)

        # optimizer
        self.optim = optim.Adam(self.model.parameters())
        
        # loss
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        """
        training seq2seq model

        Args:
            None
        Returns:
            None
        """
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.train_data_loader)
            for batch in pbar:
                src, trg = batch
                src, trg = src.to(device), trg.to(device)

                # forward
                res = self.model(torch.flip(src, [1]), trg[:,:-1]) # [batch_size, seq_len, num_trg_words]
                res = res.permute(0, 2, 1) # [batch_size, num_trg_words, seq_len]
                loss = self.criterion(res, trg[:,1:])   
                
                # backward
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                        
                pbar.set_description((f"loss : {loss.item():.4f}"))

            # evaluate
            valid_score = self.evaluate()
            print(f'valid score : {valid_score:.3f}')

    def evaluate(self):
        """
        get loss, BLEU score about test dataset
        Args:
            None
        Returns:
            loss(float), BLEU score(float)
        """
        bleu_score = 0
        self.model.eval()

        with torch.no_grad():
            for batch in self.test_data_loader:
                src, trg = batch
                src = src.to(device) # [1, in_seq_len]
                trg = trg.to(device) # [1, out_seq_len]
                output = torch.tensor(((self.vocab(self.vocab.SOS_TOKEN),),)).to(device) # [1,1]
                
                while True:
                    predictions = self.model(src, output)[0] # [seq_len, num_trg_words]
                    index = torch.argmax(predictions[-1])
                    output = torch.cat((output, torch.tensor(((index,),)).to(device)), -1) # [1, x]

                    if index == self.vocab(self.vocab.EOS_TOKEN):
                        break

                bleu_score += sentence_bleu([trg[0][1:-1].tolist()], output[0][1:-1].tolist())
        self.model.train()
        bleu_score /= len(self.test_data_loader)

        return bleu_score


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
    parser.add_argument('--ck_path', type=str, default='../checkpoint')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--vocab_path', type=str, default='../vocab')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

    chat = Chat(args)