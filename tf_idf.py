

import json
import re
import jieba.analyse as analyse
tfidf = analyse.extract_tags


word2id = {}
word2id["BLK"] = 0
word2id["UNK"] = 1
f = open("../original_data/sinanews.train", encoding='utf8')
while True:
    art = f.readline()
    if not art:
        break
    art = art.strip().split("\t")
    sen = art[2].split()
    words = ''
    for word in sen:
        words += word
        
    keywords = tfidf(words, topK = 100)
    for word in keywords:
        if word not in word2id:
            word2id[word] = len(word2id)

print("Finish word2id...")

def process_data(mode):
    f = open("../original_data/sinanews.{}".format(mode), encoding = 'utf8')
    total = 0
    while True:
        line = f.readline()
        if not line:
            break
        total += 1
    f.close()
    f = open("../original_data/sinanews.{}".format(mode), encoding = 'utf8')
    data_word = np.zeros(shape = [total, 200], dtype = np.int32)
    for i in range(total):
        line = f.readline()
        # # # # # # pdb.set_trace()
        art = line.strip().split('\t')
        moods = art[1].split()[1:]
        sen = art[2].split()
        for j, word in enumerate(sen):
            if j >= 200:
                break
            try:
                data_word[i][j] = self.word2id[word]
            except:
                data_word[i][j] = self.word2id["UNK"]
        
    np.save("../data/{}_word.npy".format(mode), data_word)

process_data("train")
process_data("test")
    






