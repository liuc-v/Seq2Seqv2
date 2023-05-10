import pickle
import torch
import config


def translate(sentence, model):   # 传入的sentence是index, 形状是n
    with open('./vocab/SRC_VOCAB.pkl', 'rb') as f:
        src_vocab = pickle.load(f)

    with open('./vocab/TRG_VOCAB.pkl', 'rb') as f:
        trg_vocab = pickle.load(f)

    en_hidden, en_cell = model.encoder(sentence.reshape(-1, 1))

    de_hidden, de_cell = en_hidden, en_cell  # 初始化

    results = [config.BOS_IDX]
    while True:
        de_input = torch.LongTensor([results[-1]]).to(config.device)
        prediction, de_hidden, de_cell = model.decoder(de_input, de_hidden, de_cell)
        index = int(prediction.argmax(dim=1))
        results.append(index)
        if index == config.EOS_IDX or len(results) > 50:
            break

    return results


# model = torch.load('LSTM21.model')
# model = model.to(config.device)
# print(translate(torch.LongTensor([2, 6, 7, 4, 3]).to(config.device), model))
