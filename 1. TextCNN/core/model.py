import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1d(nn.Module):
    def __init__(self, vocab_size, args):
        super().__init__()
        
        self.args = args
        self.embedding = nn.Embedding(vocab_size, 
                                      self.args.embedding_dim,
                                      padding_idx = self.args.pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = self.args.embedding_dim, 
                                              out_channels = self.args.n_filters, 
                                              kernel_size = fs)
                                    for fs in self.args.filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(self.args.filter_sizes) * self.args.n_filters, self.args.output_dim)
        
        self.dropout = nn.Dropout(self.args.dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)# [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1) # [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs] # [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1)) # [batch size, n_filters * len(filter_sizes)]  
        return self.fc(cat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Model Builder')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pad_idx', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN1d(1000, args)
    sample = torch.randint(20, (3, 5)).to(device)
    res = model(sample)

    print(res.shape)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')