
import os 
import json
import parser
import numpy as np 
import sys


class Gendata():
    def __init__(self, MAX):
        self.word2id = {}
        self.mood2id = {}
        self.MAX = MAX
         
    def process_vec(self):
        f = open("../original_data/sinanews.train", encoding='utf8')
        self.word2id["BLK"] = 0
        self.word2id["UNK"] = 1
        while True:
            art = f.readline()
            if not art:
                break
            art = art.strip().split("\t")
            moods = art[1].split()[1:]
            for mood in moods:
                word = mood[0:2]
                if word not in self.mood2id:
                    self.mood2id[word] = len(self.mood2id)
            sen = art[2].split()
            for word in sen:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
    
    def process_trec_vec(self):
        f = open('../original_data/TREC_train.txt', encoding = 'utf8')
        self.word2id["BLK"] = 0
        self.word2id["UNK"] = 1
        while True:
            line = f.readline()
            if not line:
                break
            parts = line.strip().split(":")
            if parts[0] not in self.mood2id:
                self.mood2id[parts[0]] = len(self.mood2id)
            sen = parts[1].split()
            for word in sen:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)

                        
    def process_trec(self, mode):
        f = open('../original_data/TREC_{}.txt'.format(mode))
        total = 0
        while True:
            line = f.readline()
            if not line:
                break
            total += 1
          
        f.close()
        f = open("../original_data/TREC_{}.txt".format(mode))
        data_word = np.zeros(shape = [total, self.MAX], dtype = np.int32)
        data_label = np.zeros(shape = [total], dtype = np.int32)
        data_length = np.zeros(shape = [total], dtype = np.int32)
        for i in range(total):
            line = f.readline()
            parts = line.strip().split(":")
            sen = parts[1].split()
            for j, word in enumerate(sen):
                if j >= self.MAX:
                    break
                try:
                    data_word[i][j] = self.word2id[word]
                except:
                    data_word[i][j] = self.word2id["UNK"]
            data_length[i] = len(sen)
            data_label[i] = self.mood2id[parts[0]]
        
        f = open("../data/config.json", 'w')
        json.dump({"word_total":len(self.word2id), "mood_total": len(self.mood2id), "sen_len": self.MAX}, f)
        f.close()

        np.save("../data/{}_word.npy".format(mode), data_word)
        np.save("../data/{}_label.npy".format(mode), data_label)
        np.save("../data/{}_length.npy".format(mode), data_length)
                    
    
    def process_data(self, mode):
        f = open("../original_data/sinanews.{}".format(mode), encoding = 'utf8')
        total = 0
        while True:
            line = f.readline()
            if not line:
                break
            total += 1
        f.close()
        f = open("../original_data/sinanews.{}".format(mode), encoding = 'utf8')
        data_word = np.zeros(shape = [total, self.MAX], dtype = np.int32)
        data_label = np.zeros(shape = [total], dtype = np.int32)
        data_length = np.zeros(shape = [total], dtype = np.int32)
        for i in range(total):
            line = f.readline()
            art = line.strip().split('\t')
            moods = art[1].split()[1:]
            sen = art[2].split()
            for j, word in enumerate(sen):
                if j>= self.MAX:
                    break
                try:
                    data_word[i][j] = self.word2id[word]
                except:
                    data_word[i][j] = self.word2id["UNK"]
            
            data_length[i] = len(sen)
            mood_num = {}
            for mood in moods:
                mood_num[mood[0:2]] = int(mood[3])
            
            mood_key = ''
            num = 0
            for key in mood_num:
                if mood_num[key] > num:
                    num = mood_num[key]
                    mood_key = key 
            data_label[i] = self.mood2id[mood_key]
                    

        # save data 
        f = open("../data/config.json", 'w')
        json.dump({"word_total":len(self.word2id), "mood_total": len(self.mood2id), "sen_len": self.MAX}, f)
        f.close()

        np.save("../data/{}_word.npy".format(mode), data_word)
        np.save("../data/{}_label.npy".format(mode), data_label)
        np.save("../data/{}_length.npy".format(mode), data_length)

if not os.path.exists("../data"):
    os.mkdir("../data")

if sys.argv[3] == 'sina':
    print("generate sina data....")
    gen = Gendata(int(sys.argv[2]))
    gen.process_vec()
    gen.process_data(sys.argv[1])

elif sys.argv[3] == 'trec':
    print("generate trec data....")
    gen = Gendata(int(sys.argv[2]))
    gen.process_trec_vec()
    gen.process_trec(sys.argv[1])

