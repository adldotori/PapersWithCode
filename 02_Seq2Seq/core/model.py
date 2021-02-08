import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, cell_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, cell_dim, num_layers)
       
    def forward(self, text): # [batch_size, in_seq_len]
        emb = self.embedding(text) # [batch_size, in_seq_len, emb_dim]
        emb = emb.permute(1, 0, 2) # [in_seq_len, batch_size, emb_dim]
        output, (h_n, c_n) = self.lstm(emb) # [in_seq_len, batch_size, cell_dim]
        # h_n: [num_layers, batch_size, cell_dim]
        # c_n: [num_layers, batch_size, cell_dim]
        return h_n, c_n

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, cell_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, cell_dim, num_layers)
        self.fc = nn.Linear(cell_dim, output_dim)

    def forward(self, text, h, c): # [batch_size, out_seq_len], [num_layers, batch_size, cell_dim], [num_layers, batch_size, cell_dim]
        emb = self.embedding(text) # [batch_size, out_seq_len, emb_dim]
        emb = emb.permute(1, 0, 2) # [out_seq_len, batch_size, emb_dim]
        output, (h_n, c_n) = self.lstm(emb, (h, c)) # [batch_size, out_seq_len, cell_dim]
        output = output.permute(1, 0, 2) # [out_seq_len, batch_size, cell_dim]
        output = self.fc(output) # [out_seq_len, batch_size, output_dim]
        return output

class Seq2Seq(nn.Module):
    EMB_DIM = 1000
    CELL_DIM = 1000
    NUM_LAYERS = 4
    def __init__(self, num_src_words, num_trg_words):
        super().__init__()
        self.encoder = Encoder(num_src_words, self.EMB_DIM, self.CELL_DIM, self.NUM_LAYERS)
        self.decoder = Decoder(num_src_words, num_trg_words, self.EMB_DIM, self.CELL_DIM, self.NUM_LAYERS)
        
    def forward(self, source, target): # [batch_size, in_seq_len] / [batch_size, out_seq_len]
        h_n, c_n = self.encoder(source) # [num_layers, batch_size, cell_dim] / [num_layers, batch_size, cell_dim]
        output = self.decoder(target, h_n, c_n) # [out_seq_len, batch_size, output_dim]]
        return output

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Seq2Seq(160000, 80000).to(device)
    sample_src = torch.randint(20, (3, 5)).to(device)
    sample_trg = torch.randint(20, (3, 6)).to(device)
    res = model(sample_src, sample_trg)

    print(f'{sample_src.shape}, {sample_trg.shape} => {res.shape}')
    print('[batch_size, in_seq_len], [batch_size, out_seq_len] => [batch_size, out_seq_len, output_dim]\n')
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters\n')

    print(model)

    encoder = 0
    decoder = 0
    for name, parameter in model.named_parameters():
        if 'encoder.lstm.weight' in name:
            encoder += parameter.numel()
        if 'decoder.lstm.weight' in name:
            decoder += parameter.numel()
    print(f'encoder parameter : {encoder/1e6}M\ndecoder parameter : {decoder/1e6}M\n')