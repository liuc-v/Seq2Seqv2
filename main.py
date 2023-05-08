import pickle
import config
from torch.utils.data import DataLoader
import torch.nn as nn
from load_file import load_file
from MyDataset import MyDataset
import torch
from LSTM import Encoder, Decoder, Seq2Seq
from train import train, evaluate


def get_dataloader():
    with open('./vocab/SRC_VOCAB.pkl', 'rb') as f:
        src_vocab = pickle.load(f)

    with open('./vocab/TRG_VOCAB.pkl', 'rb') as f:
        trg_vocab = pickle.load(f)

    src, trg = load_file("./data/multi30k_train.txt")
    train_data = MyDataset(src, trg, src_vocab, trg_vocab)
    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, collate_fn=train_data.batch_data_process)

    src, trg = load_file("./data/multi30k_valid.txt")
    valid_data = MyDataset(src, trg, src_vocab, trg_vocab)
    valid_dataloader = DataLoader(valid_data, config.batch_size, shuffle=False, collate_fn=valid_data.batch_data_process)

    src, trg = load_file("./data/multi30k_test.txt")
    test_data = MyDataset(src, trg, src_vocab, trg_vocab)
    test_dataloader = DataLoader(test_data, config.batch_size, shuffle=False, collate_fn=test_data.batch_data_process)

    return train_dataloader, valid_dataloader, test_dataloader


def run():
    print("--------------LOAD DATA------------------")
    with open('./vocab/SRC_VOCAB.pkl', 'rb') as f:
        src_vocab = pickle.load(f)

    with open('./vocab/TRG_VOCAB.pkl', 'rb') as f:
        trg_vocab = pickle.load(f)

    enc = Encoder(len(src_vocab.get_itos()), config.embed_dim, config.hidden_dim, config.n_layers, config.encoder_dropout)
    dec = Decoder(len(trg_vocab.get_itos()), config.embed_dim, config.hidden_dim, config.n_layers, config.decoder_dropout)
    model = Seq2Seq(enc, dec, config.device).to(config.device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader()
    print("--------------FINISH LOAD DATA------------------")

    for i in range(1, config.epoch_num + 1):
        train_loss = train(model, train_dataloader, opt, criterion)
        torch.cuda.empty_cache()
        valid_loss = evaluate(model, valid_dataloader, criterion)
        torch.cuda.empty_cache()
        print("EPOCH:{}, TRAIN_LOSS{}, VALID_LOSS{}".format(i, train_loss, 1))
        torch.save(model, "LSTM{}.model".format(i))

        best_valid_loss = float('inf')
        early_stop = config.early_stop
        if valid_loss > best_valid_loss:
            early_stop = early_stop - 1
            if early_stop <= 0:
                print("-------------EARLY STOP!-----------------")
                break
        else:
            early_stop = config.early_stop


if __name__ == '__main__':
    run()
