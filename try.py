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
import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
print(criterion(torch.Tensor([[100.0, 90000, 4]]), torch.tensor([1])))

