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
            outs = outs + self.index2word[index]
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
data = open(data_path, 'r', encoding='utf-8')
dataset = json.load(data)
data.close()
train_num = len(dataset["train"])
test_num = len(dataset["test"])
max_sent_len = 0
max_story_len = 0
mean_story_len = 0
story_num = 0
total_story_len = 0

for tri in dataset["train"]:
    for sent in dataset["train"][tri]["conversation"]:
        mydic.addsent(sent)
        max_sent_len = max(max_sent_len, len(sent))
    for kl in dataset["train"][tri]["knowledge"]:
        cur_len = len(kl[0]) + len(kl[1]) + len(kl[2]) + 2
        max_story_len = max(max_story_len, cur_len)
        story_num += 1
        total_story_len += cur_len
mean_story_len = total_story_len / story_num

for tei in dataset["test"]:
    for sent in dataset["test"][tei]["conversation"]:
        mydic.addsent(sent)
        max_sent_len = max(max_sent_len, len(sent))
    g_sent = ""
    for gi in dataset["test"][tei]["goal"]:
        g_sent = gi[0] + " " + gi[1] + " " + gi[2]
        mydic.addsent(g_sent)
print("finish building dict")


class Trainer():
    def __init__(self, config):
        self.train_data = bAbIDataset(config.dataset_dir, config.task)
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

        # print("Longest sentence length", self.train_data.sentence_size)
        print("Longest sentence length", max_sent_len)
        # print("Longest story length", self.train_data.max_story_size)
        print("Longest story length", max_story_len)
        # print("Average story length", self.train_data.mean_story_size)
        print("Average story length", mean_story_len)
        # print("Number of vocab", self.train_data.num_vocab)
        print("Number of vocab", mydic.totalwords)

        self.mem_n2n = MemN2N(settings)
        self.ce_fn = nn.CrossEntropyLoss(size_average=False)
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

            if (epoch+1) % 10 == 0:
                train_acc = self.evaluate("train")
                test_acc = self.evaluate("test")
                print(epoch+1, loss, train_acc, test_acc)
        print(train_acc, test_acc)

    def load(self, directory):
        pass

    def evaluate(self, data="test"):
        correct = 0
        loader = self.train_loader if data == "train" else self.test_loader
        for step, (story, query, answer) in enumerate(loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if self.config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            pred_prob = self.mem_n2n(story, query)[1]
            pred = pred_prob.data.max(1)[1] # max func return (max, argmax)
            correct += pred.eq(answer.data).cpu().sum()

        acc = correct / len(loader.dataset)
        return acc

    def _train_single_epoch(self, epoch):
        config = self.config
        num_steps_per_epoch = len(self.train_loader)
        for step, (story, query, answer) in enumerate(self.train_loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            self.opt.zero_grad()
            loss = self.ce_fn(self.mem_n2n(story, query)[0], answer)
            loss.backward()

            self._gradient_noise_and_clip(self.mem_n2n.parameters(),
                noise_stddev=1e-3, max_clip=config.max_clip)
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
