import os
import random
from itertools import chain
import numpy as np
import torch
import torch.utils.data as data
from data_utils import load_task, vectorize_data
from six.moves import range, reduce
from ..preprocess.preprocess.py import pre_process


class bAbIDataset(data.Dataset):
    def __init__(self, dataset_dir, task_id=1, memory_size=50, train=True):
        self.train = train
        self.task_id = task_id
        self.dataset_dir = dataset_dir

        train_data, test_data = load_task(self.dataset_dir, task_id)
        data = train_data + test_data

        self.vocab = set()
        for story, query, answer in data:
            self.vocab = self.vocab | set(list(chain.from_iterable(story))+query+answer)
        self.vocab = sorted(self.vocab)
        word_idx = dict((word, i+1) for i, word in enumerate(self.vocab))

        self.max_story_size = max([len(story) for story, _, _ in data])
        self.query_size = max([len(query) for _, query, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        # Add time words/indexes
        for i in range(self.memory_size):
            word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx) + 1 # +1 for nil word
        self.sentence_size = max(self.query_size, self.sentence_size) # for the position
        self.sentence_size += 1  # +1 for time words
        self.word_idx = word_idx

        self.mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))

        if train:
            story, query, answer = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)


class licDataset(data.Dataset):
    def __init__(self, dataset_dir, memory_size=50, train=True):
        train_data, test_data = pre_process()
        data = train_data + test_data
        self.vocab = set()
        for story, query, answer in data:
            self.vocab = self.vocab | set(list(chain.from_iterable(story))+query+answer)
        self.vocab = sorted(self.vocab)
        word_idx = dict((word, i+1) for i, word in enumerate(self.vocab))

        self.max_story_size = max([len(story) for story, _, _ in data])
        self.query_size = max([len(query) for _, query, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        # Add time words/indexes
        for i in range(self.memory_size):
            word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx) + 1 # +1 for nil word
        self.sentence_size = max(self.query_size, self.sentence_size) # for the position
        self.sentence_size += 1  # +1 for time words
        self.word_idx = word_idx

        self.mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))

        if train:
            story, query, answer = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    '''
        self.mydic = WIdic()
        dataset = json.load(open(dataset_dir, 'r', encoding='utf-8'))
        train_num = len(dataset["train"])
        test_num = len(dataset["test"])

        max_sent_len = 0
        max_story_len = 0
        mean_story_len = 0
        story_num = 0
        total_story_len = 0
        max_num_story = 0

        for tri in dataset["train"]:
            for sent in dataset["train"][tri]["conversation"]:
                self.mydict.addsent(sent)
                max_sent_len = max(max_sent_len, len(sent))
            for kl in dataset["train"][tri]["knowledge"]:
                max_num_story = max(max_num_story, len(dataset["train"][tri]["knowledge"]))
                cur_len = len(kl[0]) + len(kl[1]) + len(kl[2]) + 2
                max_story_len = max(max_story_len, cur_len)
                story_num += 1
                total_story_len += cur_len
                tmp_sent = " ".join(kl)
                self.mydict.addsent(tmp_sent)
        mean_story_len = total_story_len / story_num

        for tei in dataset["test"]:
            for sent in dataset["test"][tei]["history"]:
                self.mydict.addsent(sent)
                max_sent_len = max(max_sent_len, len(sent))
            g_sent = ""
            for gi in dataset["test"][tei]["goal"]:
                g_sent = gi[0] + " " + gi[1] + " " + gi[2]
                self.mydict.addsent(g_sent)
            for kl in dataset["test"][tei]["knowledge"]:
                max_num_story = max(max_num_story, len(dataset["test"][tei]["knowledge"]))
                tmp_sent = " ".join(kl)
                self.mydict.addsent(tmp_sent)
        self.mydict.refine()
        print("*" * 20 + "finish building dict" + "*" * 20)

        story = []
        query = []
        answer = []
        max_sent_length = min(max_sentence_len, max_sent_len)
        for i in range(train_num):
            story.append([])
            for j in range(max_num_story):
                story[i].append([])
                for k in range(max_sent_length):
                    story[i][j].append(0)

        for i in range(train_num):
            query.append([])
            for k in range(max_sent_length):
                query[i].append(0)

        for i in range(train_num):
            answer.append([])
            for k in range(max_sent_length):
                answer[i].append(0)

        train_dir = dataset["train"]
        for tri in train_dir:
            tmp_answer = self.mydict.W2I(train_dir[tri]["conversation"][2])
            for l in range(max_sent_length - len(tmp_answer)):
                tmp_answer.append(0)
            answer[int(tri)] = tmp_answer

            tmp_query = self.mydict.W2I(train_dir[tri]["conversation"][1])
            for l in range(max_sent_length - len(tmp_query)):
                tmp_query.append(0)
            query[int(tri)] = tmp_query

            cur_num = len(train_dir[tri]["knowledge"])
            for jth in range(cur_num):
                tmp_story = " ".join(train_dir[tri]["knowledge"][jth])
                tmp_story_ids = self.mydict.W2I(tmp_story)
                for l in range(max_sent_length - len(tmp_story_ids)):
                    tmp_story_ids.append(0)
                story[int(tri)][jth] = tmp_story_ids

        self.sentence_size = max_sent_length
        self.max_story_size = max_story_len         # useless, only for print
        self.mean_story_size = mean_story_len       # useless, only for print
        self.num_vocab = self.mydict.totalwords
        self.max_num_story = max_num_story

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(answer)
        # self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)
    '''