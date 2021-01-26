import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, pad_idx, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(vocab_size, args.embedding_dim, padding_idx = pad_idx)
        self.embedding2 = nn.Embedding(vocab_size, args.embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = args.n_filters, 
                                              kernel_size = (fs, args.embedding_dim)) 
                                    for fs in args.filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(args.filter_sizes) * args.n_filters, args.output_dim)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text) # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) # [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # [batch size, n_filters, sent len - filter_sizes[n] + 1]

        if self.args.mode == 'multichannel':
            embedded2 = self.embedding2(text) # [batch size, sent len, emb dim]
            embedded2 = embedded2.unsqueeze(1) # [batch size, 1, sent len, emb dim]
            
            conved2 = [F.relu(conv(embedded2)).squeeze(3) for conv in self.convs]
                
            conved = [conved[i] + conved2[i] for i in range(len(conved))]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1)) # [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Model Builder')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN(vocab_size=1000, pad_idx=0, args=args).to(device)
    sample = torch.randint(20, (3, 5)).to(device)
    res = model(sample)

    print(res.shape)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')