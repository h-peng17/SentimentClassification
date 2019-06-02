
import json 
import os 
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import sys 
from embedding import Embedding
from encoder import CNN, RNN, ATT, MLP
from classifier import Classifier
from data_loader import Data_loader 
from optparse import OptionParser
import pdb
import sklearn.metrics as metrics
import time 

class Model(nn.Module):
    def __init__(self, config, weight_tabel = None):
        super(Model, self).__init__()
        self.embedding = Embedding(config)
        if config.model_name == "CNN" or config.model_name == "CNN_ATT":
            self.encoder = CNN(config)
        elif config.model_name == 'RNN' or config.model_name == "RNN_ATT":
            self.encoder = RNN(config)
        elif config.model_name == 'MLP':
            self.encoder = MLP(config)
        elif config.model_name == 'ATT':
            self.encoder = ATT(config)
        self.classifier = Classifier(config, weight_tabel)

    def forward(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        loss, output, logit = self.classifier(sen_embedding)

        return loss, output, logit
    
    def test(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        output, logit = self.classifier.test(sen_embedding)

        return output, logit


class Config():
    def __init__(self, _config):
        self.word_total = _config["word_total"]
        self.sen_len = _config["sen_len"]
        self.mood_total = _config["mood_total"]
        self.embedding_size = 50
        self.hidden_size = 230
        self.kernel_size = 3
        self.drop_rate = 0.5 
        self.lr = 0.001
        self.optimizer = "SGD" 
        self.weight_decay = 0
        self.max_epoch = 20
        self.dev_step = 5
        self.batch_size = 20
        self.save_epoch = 1
        self.model_name = "CNN"
        self.use_att = 0
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
    
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    
    def set_lr(self, lr):
        self.lr = lr 
    
    def set_model_name(self, model_name):
        self.model_name = model_name
    
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
    
    def set_dev_step(self, dev_step):
        self.dev_step = dev_step
    
    def set_drop_rate(self, drop_rate):
        self.drop_rate = drop_rate
    
    def set_use_att(self, use_att):
        self.use_att = use_att



class Train():
    def __init__(self, train_data_loader, dev_data_loader, ckpt_dir, config):
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.ckpt_dir = ckpt_dir
        self.config = config
        self.correct = 0
        self.total = 0
    
    def init_train(self, model, optimizer):
        print("Initialize train model...") 
        print("optimizer:"+optimizer)
        print('lr:{}'.format(self.config.lr))
        self.train_model = model
        self.train_model.cuda()
        self.train_model.train()

        print("wash parameters...")
        parameters_to_optimize = filter(lambda x: x.requires_grad, self.train_model.parameters())
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(parameters_to_optimize, lr = self.config.lr, weight_decay = self.config.weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(parameters_to_optimize, lr = self.config.lr, weight_decay = self.config.weight_decay)
        elif optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(parameters_to_optimize, lr = self.config.lr, weight_decay = self.config.weight_decay)

        print("finish init")

    def to_var(self, x):
        return torch.from_numpy(x).to(torch.int64).cuda()

    def train_one_step(self, is_training):
        if is_training:
            batch = self.train_data_loader.next(self.config.batch_size)
            self.train_model.embedding.word = self.to_var(batch["word"])
            self.train_model.classifier.label = self.to_var(batch["label"])

            self.optimizer.zero_grad() # set grad = 0 at first 
            loss, output, logit = self.train_model()
            loss.backward()
            self.optimizer.step()    
        else:
            batch = self.dev_data_loader.next(self.config.batch_size)
            self.train_model.embedding.word = self.to_var(batch["word"])
            self.train_model.classifier.label = self.to_var(batch["label"])
            
            loss = -1 
            output, logit = self.train_model.test()

        output = np.array(((output.cpu()).detach()))
        # pdb.set_trace()

        self.correct += (batch["label"] == output).sum()
        self.total += len(batch["label"])

        return loss, logit
    
    def train(self):
        print("begin training....")
        begin_time = time.time()
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        train_order = self.train_data_loader.order
        dev_order = self.dev_data_loader.order

        for epoch in range(self.config.max_epoch):
            self.correct = 0
            self.total = 0 
            if epoch %  self.config.dev_step == 0:
                self.train_model.eval()
                for i in range(int(len(dev_order) / self.config.batch_size)):
                    _, logit = self.train_one_step(False)
                    sys.stdout.write('dev acc:{}\r'.format(round(self.correct / self.total, 6)))
                    sys.stdout.flush()
                print('dev acc:{}\r'.format(round(self.correct / self.total, 6)))
                
                self.train_model.train()
            else:
                for i in range(int(len(train_order) / self.config.batch_size)):
                    loss, _ = self.train_one_step(True)
                    sys.stdout.write("epoch:{} loss:{}, acc:{}\r".format(epoch, round(float(loss), 6), round(self.correct / self.total, 6)))
                    sys.stdout.flush()
                print("epoch:{} loss:{}, acc:{}\r".format(epoch, round(float(loss), 6), round(self.correct / self.total, 6)))
            if epoch % self.config.save_epoch == 0:
                path = os.path.join(self.ckpt_dir, self.config.model_name + '-' + str(epoch))
                torch.save(self.train_model.state_dict(), path)
                print('Have saved model to ' + path)
                print(" ")
        print("Training is done, using {}s".format(time.time() - begin_time))

class Test():
    def __init__(self, test_data_loader, ckpt_dir, config):
        self.test_data_loader = test_data_loader
        self.ckpt_dir = ckpt_dir
        self.config = config
        self.correct = 0
        self.total = 0
    
    def init_test(self, model):
        print("Initialize test model...") 
        print('lr:{}'.format(self.config.lr))
        self.test_model = model
        self.test_model.cuda()
        self.test_model.eval()

    def to_var(self, x):
        return torch.from_numpy(x).to(torch.int64).cuda()

    def test_one_step(self):
        batch = self.test_data_loader.next(self.config.batch_size)
        self.test_model.embedding.word = self.to_var(batch["word"])
        self.test_model.classifier.label = self.to_var(batch["label"])

        output, logit = self.test_model.test()
        output = np.array(((output.cpu()).detach()))
        self.correct += (batch["label"] == output).sum()
        self.total += len(batch["label"])

        return batch["label"], output
    
    def test(self):
        print("begin testing....")
        if not os.path.exists(self.ckpt_dir):
            exit("wrong!!")
        test_order = self.test_data_loader.order
        best_f1 = 0
        best_acc = 0
        for epoch in range(0, self.config.max_epoch):
            path = os.path.join(self.ckpt_dir, self.config.model_name + '-' + str(epoch))
            if not os.path.exists(path):
                continue
            print("Start testing epoch %d " % (epoch))
            self.test_model.load_state_dict(torch.load(path))
            self.correct = 0
            self.total = 0 
            self.result = []
            self.label = []
            for i in range(int(len(test_order) / self.config.batch_size)):
                label, output = self.test_one_step()
                sys.stdout.write("epoch:{}, acc:{}\r".format(epoch, round(self.correct / self.total, 6)))
                sys.stdout.flush()
                self.result.extend(output)
                self.label.extend(label.tolist())
            print("epoch:{}, acc:{}\r".format(epoch, round(self.correct / self.total, 6)))
            best_acc = best_acc if self.correct / self.total <= best_acc else self.correct / self.total
            f1 = metrics.f1_score(self.label, self.result, average='macro')
            print("F1: {}".format(f1))
            print(" ")
            if f1 > best_f1:
                best_f1 = f1 
        f = open("../res/{}".format(self.config.model_name), 'a+')
        data = "Acc: {}F1: {}, the paras are embedding_size: {}, hidden_size: {}, learn_rate: {}, droprate: {}, batch_size: {}\n".format(\
                   best_acc , best_f1, self.config.embedding_size, self.config.hidden_size, self.config.lr, self.config.drop_rate, self.config.batch_size)
        f.write(data)
        f.close()


# the main function
parser = OptionParser()
parser.add_option('--model_name', dest='model_name',default='CNN',help='model name')
parser.add_option('--gpu', dest='gpu',default=5,help='gpu id for running')
parser.add_option('--mode',dest='mode',default='train',help='to train or to test')
parser.add_option('--lr',dest='lr',default=0.001,help='learning rate')
parser.add_option('--hs',dest='hs',default=230,help='hidden size')
parser.add_option('--droprate',dest='droprate',default=0.7,help='keep rate')
parser.add_option('--weight_decay',dest='weight_decay',default=1e-5,help='keep rate')
parser.add_option('--max_epoch',dest='max_epoch',default=20,help='max epoch')
parser.add_option('--optimer',dest='optimer',default='SGD',help='optimizer for training')
parser.add_option('--dev_step',dest='dev_step',default=5,help='steps for dev')
parser.add_option('--batch_size',dest='batch_size',default=64,help='batch size')
(options, args) =parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)


# load config
f = open(os.path.join('../data', "config.json"),'r')
conf = json.load(f)
f.close()

config = Config(conf)
config.set_lr(float(options.lr))
config.set_dev_step(int(options.dev_step))
config.set_model_name(options.model_name)
config.set_max_epoch(int(options.max_epoch))
config.set_batch_size(int(options.batch_size))
config.set_weight_decay(float(options.weight_decay))
config.set_drop_rate(float(options.droprate))
config.set_hidden_size(int(options.hs))
config.set_use_att(0 if options.model_name[-3:] != 'ATT' else 1)

if not os.path.exists("../res"):
    os.mkdir("../res")

if options.mode == 'train':
    train_data_loader = Data_loader("train", config)
    dev_data_loader = Data_loader("test", config)
    ckpt_dir = '../' + options.model_name + '-'+ str(options.lr) + '-' + str(options.weight_decay) + '-' + str(options.droprate)
    print(ckpt_dir)

    train = Train(train_data_loader, dev_data_loader, ckpt_dir, config)
    train.init_train(Model(config, train_data_loader.weight_tabel), options.optimer)
    train.train()

else:
    ckpt_dir = '../' + options.model_name + '-'+ str(options.lr) + '-' + str(options.weight_decay) + '-' + str(options.droprate)
    print(ckpt_dir)

    test_data_loader = Data_loader("test", config)
    test = Test(test_data_loader, ckpt_dir, config)
    test.init_test(Model(config, test_data_loader.weight_tabel))
    test.test()

