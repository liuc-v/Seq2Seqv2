# # 测试dataloader
# import pickle
# from load_file import load_file
# from MyDataset import MyDataset
# from torchtext import vocab
#
# src, trg = load_file("./data/multi30k_train.txt")
# with open('./vocab/TRG_VOCAB.pkl', 'rb') as f:
#     trg_vocab = pickle.load(f)
#
# with open('./vocab/SRC_VOCAB.pkl', 'rb') as f:
#     src_vocab = pickle.load(f)
#
# mydata = MyDataset(src, trg, src_vocab, trg_vocab)
# print([trg_vocab.get_itos()[a] for a in mydata[28999][1]])

# import logging
#
# logging.info('11111111')
# import torch
# from torch import nn
#
# criterion = nn.CrossEntropyLoss()
# print(criterion(torch.Tensor([[100.0, 90000, 4]]), torch.tensor([1])))
# import numpy as np
# a = np.array([1,2,3,4,5,6])
# print(a.reshape(1, 6).argmax(1))
# print(a.reshape(6, 1).argmax(0))

import torch
import config
from main import get_dataloader
from tokens2index import tokens2index, index2tokens

from translate import translate
model = torch.load('LSTM21.model')
model = model.to(config.device)
a, b, c = get_dataloader()
for src, trg in c:
    for i in range(len(src[0])):
        ans = translate(src[:, i], model)
        prediction = index2tokens(ans, 'en')
        trg = index2tokens(trg[:, i], 'en')


import torch.nn as nn
import torch
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):


        input = input.unsqueeze(0)


        embedded = self.dropout(self.embedding(input))


        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))



        prediction = self.fc_out(output.squeeze(0))


        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):


        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim


        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)


        input = trg[0, :]

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1

        return outputs
