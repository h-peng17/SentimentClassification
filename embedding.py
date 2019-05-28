
import torch 
import torch.nn as nn
import numpy as np 


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(config.word_total, config.embedding_size, padding_idx=0)
        self.word = None

    def init_word_embedding(self):
        nn.init.xavier_normal_(self.word_embedding.weight.data)
        self.word_embedding.weight.data[0].fill_(0)

    def forward(self):
        embedding = self.word_embedding(self.word)
        return embedding