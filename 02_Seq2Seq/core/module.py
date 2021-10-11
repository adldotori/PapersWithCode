import os
import os.path as osp
import re
import pickle
import argparse
import random
from tqdm import tqdm
import numpy as np
import functools
import heapq

import torch
import torch.nn as nn
import torch.optim as optim

from nltk.translate.bleu_score import sentence_bleu

from .model import *
from .dataloader import *
from .utils import *

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


NUM_SRC_WORDS = 160000
NUM_TRG_WORDS = 80000

class Trainer():
    def __init__(self, args):
        self.args = args

        # data
        train_dataset = FrEnCorpus(args, NUM_SRC_WORDS, NUM_TRG_WORDS, True)
        test_dataset = FrEnCorpus(args, NUM_SRC_WORDS, NUM_TRG_WORDS, False)
        self.vocab = train_dataset.trg_vocab

        self.train_data_loader = DataLoader(dataset=train_dataset, 
                                batch_size=args.batch_size,
                                collate_fn=train_dataset.collate_fn)
        self.test_data_loader = DataLoader(dataset=test_dataset, 
                                batch_size=1,
                                collate_fn=test_dataset.collate_fn)

        # model
        self.model = Seq2Seq(NUM_SRC_WORDS, NUM_TRG_WORDS).to(device)

        # optimizer
        self.optim = optim.Adam(self.model.parameters())
        
        # loss
        self.criterion = nn.CrossEntropyLoss()

        # make directory if not exist data path
        if not osp.isdir(args.ck_path): 
            os.makedirs(args.ck_path, exist_ok=True)

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
                src, src_len, trg = batch
                src, trg = src.to(device), trg.to(device)

                # forward
                res = self.model(torch.flip(src, [1]), src_len, trg[:,:-1]) # [batch_size, seq_len, num_trg_words]
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

            torch.save(self.model.state_dict(), 
                        osp.join(self.args.ck_path, f'{self.args.name}_best.pt'))

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
                
                heap = [torch.tensor(((self.vocab(self.vocab.SOS_TOKEN),),)).to(device)] # [1,1]

                while True:
                    # remain the best elements
                    if len(heap) > self.args.beam_size:
                        tmp = heap.copy()
                        heap = []
                        for i in self.args.beam_size:
                            heapq.heappush(heap, heapq.heappop(tmp))

                    for element in heap:
                        predictions = self.model(src, element)[0] # [seq_len, num_trg_words]
                        values, indices = torch.topk(predictions[-1], self.args.beam_size)

                        # add to heap
                        for value, index in zip(values, indices):
                            heapq.heappush(heap, \
                                (-value, torch.cat((element, torch.tensor(((index,),)).to(device)), -1))) # maximum heap / [1,x]
                        
                    if index == self.vocab(self.vocab.EOS_TOKEN):
                        break

                output = heapq.heappop(heap)
                bleu_score += sentence_bleu([trg[0][1:-1].tolist()], output[0][1:-1].tolist())

        self.model.train()
        bleu_score /= len(self.test_data_loader)

        return bleu_score * 100


class Tranlator():
    def __init__(self, args):
        self.args = args

        # vocabulary load
        train_dataset = FrEnCorpus(args, NUM_SRC_WORDS, NUM_TRG_WORDS, True)
        self.vocab = train_dataset.trg_vocab

        # model load
        self.model = Seq2Seq(NUM_SRC_WORDS, NUM_TRG_WORDS).to(device)
        self.model.load_state_dict(torch.load(osp.join(args.ck_path, f'{args.name}_best.pt')))
        self.model.eval()
    
    def translate(self):
        """
        Determines whether a given sentence is positive or negative
        """ 
        while True:
            input_str = preprocess(input('>>> ')) # clean and tokenize sentence
            input_str = [self.vocab.SOS_TOKEN] + input_str + [self.vocab.EOS_TOKEN] 
            src = [self.vocab(i) for i in input_str] # change vocab to idx with vocabulary

            heap = [torch.tensor(((self.vocab(self.vocab.SOS_TOKEN),),)).to(device)] # [1,1]

            while True:
                # remain the best elements
                if len(heap) > self.args.beam_size:
                    tmp = heap.copy()
                    heap = []
                    for i in self.args.beam_size:
                        heapq.heappush(heap, heapq.heappop(tmp))

                for element in heap:
                    predictions = self.model(src, element)[0] # [seq_len, num_trg_words]
                    values, indices = torch.topk(predictions[-1], self.args.beam_size)

                    # add to heap
                    for value, index in zip(values, indices):
                        heapq.heappush(heap, \
                            (-value, torch.cat((element, torch.tensor(((index,),)).to(device)), -1))) # maximum heap / [1,x]
                    
                if index == self.vocab(self.vocab.EOS_TOKEN):
                    break

            output = heapq.heappop(heap)
            
            seq_trans = ''
            for word in output[1:-1]: # except sos, eos token
                seq_trans += self.vocab.idx2word[word]

            print(f'>>> {seq_trans}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--ck_path', type=str, default='../checkpoint')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--vocab_path', type=str, default='../vocab')
    parser.add_argument('--beam_size', type=int, default=1)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

    translator = Tranlator(args)
    translator.translate()
