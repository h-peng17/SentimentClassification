
import torch 
import torch.nn as nn 

class Classifier(nn.Module):
    def __init__(self, config, weight_tabel = None):
        super(Classifier, self).__init__()
        self.mood_matrix = nn.Embedding(config.mood_total, config.hidden_size)
        self.bias = nn.Parameter(torch.randn(config.mood_total))
        nn.init.xavier_uniform_(self.mood_matrix.weight)
        nn.init.uniform_(self.bias)

        self.config = config
        self.dropout = nn.Dropout(config.drop_rate)
        self.label = None
        weight_tabel = torch.from_numpy(weight_tabel).to(torch.float32).cuda()
        self.loss = nn.CrossEntropyLoss()

    def logit(self, x):
        # x [B, H]
        return torch.matmul(x, torch.transpose(self.mood_matrix.weight, 0, 1)) + self.bias
    
    def forward(self, x):
        # get distribution on moods
        # x[B, H] -> [B, M]
        x = self.logit(x)
        _, output = torch.max(x, dim = 1)
        # x [B, M], label [B](0~M-1)
        loss = self.loss(x, self.label)
        return loss, output, x 
    
    def test(self, x):
        # get distribution on moods
        # x[B, H] -> [B, M]
        x = self.logit(x)
        _, output = torch.max(x, dim = 1)
        return output, x 
