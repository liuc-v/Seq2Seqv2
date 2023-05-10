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
