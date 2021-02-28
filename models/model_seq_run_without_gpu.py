import pandas as pd

from time import time 
from gensim.models import KeyedVectors
from collections import namedtuple


# Pytorch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


# stanza
import stanza as st

import numpy as np



# Pretrained word2vec
import gensim.downloader as api
corpus = api.load('glove-wiki-gigaword-300', return_path=True)
pretrainedwvmodel = KeyedVectors.load_word2vec_format(corpus)
embedding_matrix = pretrainedwvmodel.wv.vectors
embedding_matrix = np.append(embedding_matrix, np.zeros((1,300)), axis=0) # Padding
embedding_matrix = np.append(embedding_matrix, np.zeros((1,300)), axis=0)
embedding_matrix = np.append(embedding_matrix, np.zeros((1,300)), axis=0) # Unknown word


TAG2CLASS = {
    '<PAD>': 0,
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
    'WRB': 36,
    '-RRB-': 37,
    '-LRB-':38,
        '<UNK>': 0,
    
}
pos_tagger = st.Pipeline(lang='en', use_gpu=False)

class DataMapper1(Dataset):
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
        yseq = np.zeros(self.sequence_len, dtype=np.int64)
        for k in doc.sentences[0].words:
            if (self.model.wv.vocab.get(k.text) is None):
                xl.append(400002)
                yl.append(TAG2CLASS.get('<UNK>'))
                continue
            xl.append(self.model.wv.vocab.get(k.text).index)
            yl.append(TAG2CLASS.get(k.xpos, 0))
        seq[:len(xl)] = xl[:self.sequence_len]
        yseq[:len(yl)] = yl[:self.sequence_len]
        return seq, yseq

data = pd.read_csv('Sentences_15klyrics_mls_20.csv')
train_data = data.sent[:8000].to_numpy()
val_random = np.random.choice(data[:8000].to_numpy().flatten(), 800)
val_data = np.append(val_random, data.sent[10001:10801].to_numpy())
test_data = data.sent[8000:10001].to_numpy()

training_set = DataMapper1(train_data, pretrainedwvmodel, 20)
val_set = DataMapper1(val_data, pretrainedwvmodel, 20)
test_set = DataMapper1(test_data, pretrainedwvmodel, 20)

loader_training = DataLoader(training_set, batch_size=16, persistent_workers=True, num_workers=3)
loader_val = DataLoader(training_set, batch_size=16, persistent_workers=True, num_workers=3)
loader_test = DataLoader(test_set, num_workers=5)

embedding_matrix = torch.FloatTensor(embedding_matrix)
train_on_gpu = torch.cuda.is_available()
lstm_dict = {
    # 'batch_size':8,
    'hidden_dim': embedding_matrix.shape[1],
    'lstm_layers':3,
    # 'input_size':embedding_matrix.shape[0],
    'padding_idx': 400001,
    'target_size': 20,
    'embedding_matrix': embedding_matrix
}
lstm_args = namedtuple('lstm_args', lstm_dict.keys())(**lstm_dict)

class Simple_Sequence_LSTM(nn.Module):
    def __init__(self, args):
        super(Simple_Sequence_LSTM, self).__init__()

        # Hyperparameters
        # self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        # self.input_size = args.input_size
        self.embedding_matrix = args.embedding_matrix
        self.target_size = args.target_size

        self.dropout = nn.Dropout(0.5)
        # self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.embedding = nn.Embedding.from_pretrained(
            self.embedding_matrix, padding_idx=args.padding_idx, freeze=True)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                            num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(self.hidden_dim*2, self.target_size)

    def forward(self, x):
        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        # Each sequence "x" is passed through an embedding layer
        out = self.embedding(x)
        # Feed LSTMs
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)
        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = torch.log_softmax(self.fc2(out), dim=1)

        return out

model = Simple_Sequence_LSTM(lstm_args)
def validation_metrics (model, valid_dl):
    loss_function = nn.MSELoss()
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y in valid_dl:
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        pred = torch.max(y_hat, 0)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total, correct/total

t_start = time()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0001)
loss_function = nn.MSELoss()
epochs = 20
validation_loss_min = np.inf
for i in range(epochs):
    model.train()
    sum_loss = 0.0
    total = 0
    for x, y in loader_training:
        x = torch.tensor(x).to(torch.long)
        y_pred = model(x)
        y = torch.tensor(y).to(torch.float)
        
        optimizer.zero_grad()
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()*y.shape[0]
        total += y.shape[0]
    val_loss, val_acc = validation_metrics(model, loader_val)
    print(f'Epoch: {i}\tTraining loss: {sum_loss}\tValidation loss: {val_loss}')
    if val_loss <= validation_loss_min:
        print('\t\tValidation loss-nya lebih kecil!')
        torch.save(model.state_dict(),"model_seq_lyrics_best_cpu.pth")
        validation_loss_min = val_loss
    if i % 5 == 1:
        print("train loss %.3f, val loss %.3f, val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))
elapsed_time = round((time() - t_start) / 60, 2)
print("\nElapsed time for training: " + str(elapsed_time) + " minutes")

t_start = time()
val_loss, val_acc = validation_metrics(model, loader_val)
print("val loss %.3f, val accuracy %.3f" % (val_loss, val_acc))


test_loss, test_acc = validation_metrics(model, loader_test)
print("test loss %.3f, test accuracy %.3f" % (test_loss, test_acc))

elapsed_time = round((time() - t_start) / 60, 2)
print("\nElapsed time for evaluation: " + str(elapsed_time) + " minutes")