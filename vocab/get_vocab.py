import pickle
import config
from load_file import load_file
from torchtext.vocab import build_vocab_from_iterator

specials = [config.unk, config.pad, config.bos, config.eos]


def generate_vocab(sentences):
    vocab = build_vocab_from_iterator(sentences, min_freq=2, specials=specials, special_first=True)
    vocab.set_default_index(vocab[config.unk])

    return vocab


SRC = generate_vocab(load_file("../data/multi30k_train.txt")[0])
TRG = generate_vocab(load_file("../data/multi30k_train.txt")[1])

with open('SRC_VOCAB.pkl', 'wb') as f:
    pickle.dump(SRC, f)

with open('TRG_VOCAB.pkl', 'wb') as f:
    pickle.dump(TRG, f)
