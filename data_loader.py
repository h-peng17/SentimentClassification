
import json 
import numpy as np 
import random

class Data_loader():
    def __init__(self, mode, config):
        self.word = np.load("../data/{}_word.npy".format(mode))
        self.label = np.load("../data/{}_label.npy".format(mode))

        self.total = len(self.word)
        self.order = list(range(self.total))
        self.lef = 0
        self.rig = 0
        self.idx = 0
        
        _weight_tabel = np.zeros(shape = (config.mood_total), dtype = np.float32)
        for i in range(len(self.label)):
            _weight_tabel[self.label[i]] += 1.0
        print(_weight_tabel) 
        self.weight_tabel = 1 / (_weight_tabel ** 0.05)
        print(self.weight_tabel)
 
    def next(self, batch_size):
        if self.idx >= self.total:
            random.shuffle(self.order)
            self.idx = 0
        self.lef = self.idx
        self.rig = self.idx + batch_size
        self.idx = self.rig
        if self.rig >= self.total:
            self.rig = self.total

        batch_data = {}
        index = self.order[self.lef:self.rig]
        batch_data["word"] = np.array(self.word[index])
        batch_data["label"] = np.array(self.label[index])

        return batch_data

        

        