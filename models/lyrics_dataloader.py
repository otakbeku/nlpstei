import pandas as pd
import numpy as np

# Pytorch
from torch.utils.data import Dataset


# stanza
import stanza as st


TAG2CLASS = {
    '<PAD>': 0,
    '<UNK>': 404,
    'CC': 1,
    'CD': 2,
    'DT': 3,
    'EX': 4,
    'FW': 5,
    'IN': 6,
    'JJ': 7,
    'JJR': 8,
    'JJS': 9,
    'LS': 10,
    'MD': 11,
    'NN': 12,
    'NNS': 13,
    'NNP': 14,
    'NNPS': 15,
    'PDT': 16,
    'POS': 17,
    'PRP': 18,
    'PRP$': 19,
    'RB': 20,
    'RBR': 21,
    'RBS': 22,
    'RP': 23,
    'SYM': 24,
    'TO': 25,
    'UH': 26,
    'VB': 27,
    'VBD': 28,
    'VBG': 29,
    'VBN': 30,
    'VBP': 31,
    'VBZ': 32,
    'WDT': 33,
    'WP': 34,
    'WP$': 35,
    'WRB': 36, }
pos_tagger = st.Pipeline(lang='en')

class DataMapper(Dataset):
    def __init__(self, sentence_lyrics, wvmodel, sequence_len):
        self.sents = sentence_lyrics
        self.sequence_len = sequence_len
        self.model = wvmodel

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        doc = pos_tagger(self.sents[idx])
        xl = []
        yl = []
        seq = np.zeros(self.sequence_len, dtype=np.int64)
        yseq = np.zeros(self.sequence_len, dtype=object)
        for k in doc.sentences[0].words:
            if (self.model.wv.vocab.get(k.text) is None):
                xl.append(1000001)
                yl.append(TAG2CLASS.get('<UNK>'))
                continue
            xl.append(self.model.wv.vocab.get(k.text).index)
            yl.append(TAG2CLASS.get(k.xpos))
        seq[:len(xl)] = xl
        yseq[:len(yl)] = yl
        return seq, yseq
