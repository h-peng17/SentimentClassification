
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.in_channels = config.embedding_size if config.use_att == 0 else config.embedding_size * 2
        self.out_channels = config.hidden_size
        self.kernel_size = config.kernel_size
        self.cnn = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, padding = int((self.kernel_size-1) / 2))
        nn.init.xavier_normal_(self.cnn.weight)
        self.activation = nn.ReLU()
        self.config = config 
        self.dropout = nn.Dropout(config.drop_rate)
        self.softmax = nn.Softmax(dim = 2)

    def max_pooling(self, x):
        # x [B, H, N] -> [B, H]
        sen_embedding, _ = torch.max(x, dim = 2)
        return sen_embedding
    
    def attention(self, x):
        # x [B, N, E]
        K = x.permute(0, 2, 1)
        # x [B, N, E]
        attention_x = torch.matmul(self.softmax(torch.matmul(x, K)/ x.size(1) ** -0.5), x)
        return attention_x
        
    def forward(self, x):
        x = self.dropout(x)
        if self.config.use_att != 0:
            attention_x = self.attention(x)
            x = torch.cat((x, attention_x), dim = 2)
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
        self.input_size = config.embedding_size if config.use_att == 0 else config.embedding_size * 2
        self.hidden_size = config.hidden_size
        self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size)
        self.dropout = nn.Dropout(config.drop_rate)
        self.softmax = nn.Softmax(dim = 2)
        self.config = config
    
    def attention(self, x):
        # x [B, N, E]
        K = x.permute(0, 2, 1)
        # x [B, N, E]
        attention_x = torch.matmul(self.softmax(torch.matmul(x, K)/ x.size(1) ** -0.5), x)
        return attention_x

    def forward(self, x):
        x = self.dropout(x)
        if self.config.use_att != 0:
            attention_x = self.attention(x)
            x = torch.cat((x, attention_x), dim = 2)
        # x [B, N, E] -> [N, B, E]
        x = x.permute(1, 0, 2)
        # output [N, B, H], hidden [1, B, H]
        output, hidden = self.rnn(x)
        # [B, H]
        x = torch.squeeze(hidden[0])
        x = self.dropout(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(config.drop_rate)
        self.linear = nn.Linear(config.embedding_size, config.hidden_size)
    
    def forward(self, x):
        # mean the x -> [B, E] 
        x = torch.mean(x, 1)
        x = self.dropout(x)
        return x

class ATT(nn.Module):
    def __init__(self, config):
        super(ATT, self).__init__()
        self.dropout = nn.Dropout(config.drop_rate)
        self.linear = nn.Linear(config.embedding_size, config.hidden_size)
        self.softmax = nn.Softmax(dim = 2)

    def self_att(self, x):
        # x [B, N, E]
        K = x.permute(0, 2, 1)
        # x [B, N, E]
        attention_x = torch.matmul(self.softmax(torch.matmul(x, K)/ x.size(1) ** -0.5), x)
        return attention_x
    
    def forward(self, x):
        # x (B, N, E)
        attention_x = self.self_att(x)
        x = torch.mean(attention_x, 1)
        x = self.dropout(x)
        return  x
