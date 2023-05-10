import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stop = 5
epoch_num = 100
embed_dim = 256
hidden_dim = 512
n_layers = 1
encoder_dropout = 0.5
decoder_dropout = 0.5
teacher_forcing_ratio = 0.5
batch_size = 1
lr = 0.001
unk = '<unk>'
pad = '<pad>'
bos = '<bos>'
eos = '<eos>'
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
