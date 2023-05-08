import torch
from torch.utils.data import Dataset
from torchtext import vocab
import config
import numpy as np


class MyDataset(Dataset):
    def __init__(self, src_data, trg_data, src_vocab, trg_vocab):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __getitem__(self, index):
        src_sentence = self.src_data[index]
        trg_sentence = self.trg_data[index]

        src_index = [2] + [self.src_vocab[token] for token in src_sentence] + [3]
        trg_index = [2] + [self.trg_vocab[token] for token in trg_sentence] + [3]

        return src_index, trg_index

    def batch_data_process(self, batch_datas):
        src_index, trg_index = [], []
        src_len, trg_len = [], []
        for src, trg in batch_datas:
            src_index.append(src)
            trg_index.append(trg)
            src_len.append(len(src))
            trg_len.append(len(trg))

        max_src_len = max(src_len)
        max_trg_len = max(trg_len)

        for i in range(len(src_index)):
            src_index[i] = src_index[i] + [self.src_vocab[config.pad]] * (max_src_len - len(src_index[i]))

        for i in range(len(trg_index)):
            trg_index[i] = trg_index[i] + [self.trg_vocab[config.pad]] * (max_trg_len - len(trg_index[i]))

        src_index = torch.LongTensor(np.transpose(src_index)).to(config.device)
        trg_index = torch.LongTensor(np.transpose(trg_index)).to(config.device)
        return src_index, trg_index

    def __len__(self):
        assert len(self.src_data) == len(self.trg_data)
        return len(self.src_data)
