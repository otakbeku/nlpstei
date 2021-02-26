# Pytorch
import torch.nn as nn
import torch.nn.functional as f
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split


class Simple_LSTM(nn.Module):
    def __init__(self, args):
        super(Simple_LSTM, self).__init__()

        # Hyperparameters
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        self.input_size = args.input_size
        self.embedding_matrix = args.embedding_matrix.cuda()

        self.dropout = nn.Dropout(0.5)
        # self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.embedding = nn.Embedding.from_pretrained(
            self.embedding_matrix, padding_idx=args.padding_idx, freeze=True)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                            num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(self.hidden_dim*2, 10)

    def forward(self, x):
        # Hidden and cell state definion
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).cuda()
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).cuda()

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
        out = torch.softmax(self.fc2(out))

        return out

class Simple_Sequence_LSTM(nn.Module):
    def __init__(self, args):
        super(Simple_Sequence_LSTM, self).__init__()

        # Hyperparameters
        # self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        # self.input_size = args.input_size
        self.embedding_matrix = args.embedding_matrix.cuda()
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
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).cuda()
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim)).cuda()

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
        out = torch.softmax(self.fc2(out))

        return out
