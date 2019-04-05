import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bAbIDataset
from model import MemN2N
import json
import torch.utils.data as data


class WIdic:
    def __init__(self):
        self.word2index = {"<EOS>": 0, "START": 1}
        self.index2word = {0: "<EOS>", 1: "START"}
        self.totalwords = 2

    def addword(self, word):
        if self.word2index.get(word) is None:
            self.word2index[word] = self.totalwords
            self.index2word[self.totalwords] = word
            self.totalwords = self.totalwords + 1

    def addsent(self, sent):
        for word in sent.split():
            self.addword(word)

    def W2I(self, sent):
        outv = []
        for words in sent.split():
            outv.append(self.word2index[words])
        outv.append(0)
        return outv

    def I2W(self, inv):
        outs = ""
        for index in inv:
            outs += self.index2word[index]
        return outs

# learning_rate = 0.01
data_path = "../train_part.json"
# en_path = "default"
# de_path = "default"
# se_path = "default"
# train_mode = 1
# SOS = 1
# EOS = 0
# epoch = 20000
mydic = WIdic()
dataset = json.load(open(data_path, 'r', encoding='utf-8'))
train_num = len(dataset["train"])
test_num = len(dataset["test"])

max_sent_len = 0
max_story_len = 0
mean_story_len = 0
story_num = 0
total_story_len = 0
max_story_size = 0

for tri in dataset["train"]:
    for sent in dataset["train"][tri]["conversation"]:
        mydic.addsent(sent)
        max_sent_len = max(max_sent_len, len(sent))
    for kl in dataset["train"][tri]["knowledge"]:
        max_story_size = max(max_story_size, len(dataset["train"][tri]["knowledge"]))
        cur_len = len(kl[0]) + len(kl[1]) + len(kl[2]) + 2
        max_story_len = max(max_story_len, cur_len)
        story_num += 1
        total_story_len += cur_len
        tmp_sent = " ".join(kl)
        mydic.addsent(tmp_sent)
mean_story_len = total_story_len / story_num

for tei in dataset["test"]:
    for sent in dataset["test"][tei]["conversation"]:
        mydic.addsent(sent)
        max_sent_len = max(max_sent_len, len(sent))
    g_sent = ""
    for gi in dataset["test"][tei]["goal"]:
        g_sent = gi[0] + " " + gi[1] + " " + gi[2]
        mydic.addsent(g_sent)
    for kl in dataset["test"][tei]["knowledge"]:
        max_story_size = max(max_story_size, len(dataset["test"][tei]["knowledge"]))
        tmp_sent = " ".join(kl)
        mydic.addsent(tmp_sent)
print("finish building dict")

'''
Story: 32 * 20(max_story_size) * max_sent_len
Query: 32 * max_sent_len
Answer: 32 * max_sent_len

Data_story: train_num * max_story_size * max_sent_len
Data_answer: train_num * max_sent_len
Data_query:train_num * max_sent_len

'''


class licDataset(data.Dataset):
    def __init__(self, dataset_dir, memory_size=50, train=True):
        story = []
        query = []
        answer = []
        for i in range(train_num):
            story.append([])
            for j in range(max_story_size):
                story[i].append([])
                for k in range(max_sent_len):
                    story[i][j].append(0)

        for i in range(train_num):
            query.append([])
            for k in range(max_sent_len):
                query[i].append(0)

        for i in range(train_num):
            answer.append([])
            for k in range(max_sent_len):
                answer[i].append(0)

        train_dir = dataset["train"]
        for tri in train_dir:
            tmp_answer = mydic.W2I(train_dir[tri]["conversation"][2])
            for l in range(max_sent_len - len(tmp_answer)):
                tmp_answer.append(0)
            answer[int(tri)] = tmp_answer

            tmp_query = mydic.W2I(train_dir[tri]["conversation"][1])
            for l in range(max_sent_len - len(tmp_query)):
                tmp_query.append(0)
            query[int(tri)] = tmp_query

            cur_num = len(train_dir[tri]["knowledge"])
            for jth in range(cur_num):
                tmp_story = " ".join(train_dir[tri]["knowledge"][jth])
                tmp_story_ids = mydic.W2I(tmp_story)
                for l in range(max_sent_len - len(tmp_story_ids)):
                    tmp_story_ids.append(0)
                story[int(tri)][jth] =tmp_story_ids

        self.sentence_size = max_sent_len
        self.max_story_size = max_story_len
        self.mean_story_size = mean_story_len
        self.num_vocab = mydic.totalwords

        self.max_story_size = max_story_size

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(answer)
        # self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)


class Trainer():
    def __init__(self, config):
        # self.train_data = bAbIDataset(config.dataset_dir, config.task)
        self.train_data = licDataset(config.dataset_dir, config.task)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True)

        self.test_data = bAbIDataset(config.dataset_dir, config.task, train=False)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        settings = {
            "use_cuda": config.cuda,
            "num_vocab": mydic.totalwords,            # "num_vocab": self.train_data.num_vocab,
            "embedding_dim": 20,
            "sentence_size": max_sent_len,            # self.train_data.sentence_size,
            "max_hops": config.max_hops
        }

        print("Longest sentence length", self.train_data.sentence_size)
        print("Longest story length", self.train_data.max_story_size)
        print("Average story length", self.train_data.mean_story_size)
        print("Number of vocab", self.train_data.num_vocab)
        print("largest Number of story", max_story_size)

        self.mem_n2n = MemN2N(settings)
        self.ce_fn = nn.CrossEntropyLoss(size_average=False)
        self.mse_fn = nn.MSELoss()
        self.opt = torch.optim.SGD(self.mem_n2n.parameters(), lr=config.lr)
        print(self.mem_n2n)

        if config.cuda:
            self.ce_fn   = self.ce_fn.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        self.start_epoch = 0
        self.config = config

    def fit(self):
        config = self.config
        for epoch in range(self.start_epoch, config.max_epochs):
            loss = self._train_single_epoch(epoch)
            lr = self._decay_learning_rate(self.opt, epoch)

        #     if (epoch+1) % 10 == 0:
        #         train_acc = self.evaluate("train")
        #         test_acc = self.evaluate("test")
        #         print(epoch+1, loss, train_acc, test_acc)
        # print(train_acc, test_acc)

    def load(self, directory):
        pass

    def evaluate(self, _data="test"):
        correct = 0
        loader = self.train_loader if _data == "train" else self.test_loader
        for step, (story, query, answer) in enumerate(loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if self.config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            pred_prob = self.mem_n2n(story, query)[1]
            pred = pred_prob._data.max(1)[1] # max func return (max, argmax)
            correct += pred.eq(answer._data).cpu().sum()

        acc = correct / len(loader.dataset)
        return acc

    def _train_single_epoch(self, epoch):
        config = self.config
        num_steps_per_epoch = len(self.train_loader)
        for step, (story, query, answer) in enumerate(self.train_loader):
            # story = Variable(story)
            # query = Variable(query)
            # answer = Variable(answer)

            if config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            self.opt.zero_grad()
            tmp_output = self.mem_n2n(story, query)[0]
            tmp_softmax = self.mem_n2n(story, query)[1]
            # if (epoch+1) % 10 == 0:
            loss = torch.tensor(0.)
            for i in range(max_sent_len):
                loss += self.ce_fn(tmp_output[i], answer.transpose(0, 1)[i])
            # assert answer.shape == tmp_output.shape, "MSELoss requires the same shape"
            # loss = self.mse_fn(tmp_output, answer.float()) # loss is a scalar, no reduction needed
            loss.backward()

            tmp = torch.max(tmp_softmax, dim=2)[1]     # dim:sentence_size*batch_size*vocab->sentence_size*batch_size
            print("query: {}\noutput: {}\nground: {}\n"
                  .format(mydic.I2W(query[0].detach().numpy().tolist()),
                          mydic.I2W(tmp.transpose(0, 1)[0].detach().numpy().tolist()),
                          mydic.I2W(answer[0].detach().numpy().tolist())))
            print("loss:{}".format(loss.data.item()))

            self._gradient_noise_and_clip(self.mem_n2n.parameters(), noise_stddev=1e-3, max_clip=config.max_clip)
            self.opt.step()

        return loss.data.item()

    def _gradient_noise_and_clip(self, parameters,
                                 noise_stddev=1e-3, max_clip=40.0):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        nn.utils.clip_grad_norm(parameters, max_clip)

        for p in parameters:
            noise = torch.randn(p.size()) * noise_stddev
            if self.config.cuda:
                noise = noise.cuda()
            p.grad.data.add_(noise)

    def _decay_learning_rate(self, opt, epoch):
        decay_interval = self.config.decay_interval
        decay_ratio    = self.config.decay_ratio

        decay_count = max(0, epoch // decay_interval)
        lr = self.config.lr * (decay_ratio ** decay_count)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        return lr
