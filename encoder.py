
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.in_channels = config.embedding_size
        self.out_channels = config.hidden_size
        self.kernel_size = config.kernel_size
        self.cnn = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, padding=1)
        nn.init.xavier_normal_(self.cnn.weight)
        self.activation = nn.ReLU()
        self.config = config 
        self.dropout = nn.Dropout(config.drop_rate)

    def max_pooling(self, x):
        # x [B, H, N] -> [B, H]
        sen_embedding, _ = torch.max(x, dim = 2)
        return sen_embedding
    
    def forward(self, x):
        # x [B, N, E] -> [B, E, N]
        x = x.permute(0, 2, 1)
        # x [B, E, N] -> [B, H, N]
        x = self.cnn(x)
        #x [B, H, N] -> [B, H]
        x = self.max_pooling(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x 

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.input_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size)
        nn.init.xavier_normal_(self.rnn.weight)
        self.dropout = nn.Dropout(config.drop_rate)
    
    def forward(self, x):
        # x [B, N, E] -> [N, B, E]
        x = x.permute(1, 0, 2)
        # output [N, B, H], hidden [1, B, H]
        output, hidden = self.rnn(x)
        # [B, H]
        x = self.dropout(torch.squeeze(hidden))
        return x