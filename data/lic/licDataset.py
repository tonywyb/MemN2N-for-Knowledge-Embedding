from itertools import chain
from collections import defaultdict
import numpy as np
import torch
import torch.utils.data as data
from utils.data_utils import load_task, vectorize_data
from preprocess.preprocess import pre_process, extract_knowledge
import codecs
import os


class licDataset(data.Dataset):
    def __init__(self, dataset_dir, memory_size=50, train=True):
        data_path = os.path.join(dataset_dir, "train+test.json")
        dict_path = os.path.join(dataset_dir, "lic_word")
        extract_knowledge(data_path)
        train_data, test_data = pre_process(data_path)
        data = train_data + test_data

        if not os.path.exists(dict_path):
            # self.vocab = set()
            self.vocab = list()
            for story, query, answer in data:
                # self.vocab = self.vocab | set(list(chain.from_iterable(story))+query+answer)
                self.vocab = self.vocab + (list(chain.from_iterable(story)) + query + answer)
            vocab_dict = dict()
            for v in set(self.vocab):
                vocab_dict[v] = self.vocab.count(v)
            vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
            self.vocab = list(vocab_dict.keys())
            vocab_len = len(vocab_dict) if len(vocab_dict) <= 9999 else 9999
            self.vocab = self.vocab[:vocab_len]
            # self.vocab = sorted(self.vocab)
            with codecs.open("data/lic/lic_word", "w", "utf-8") as f:
                f.write(" ".join(self.vocab))
        else:
            with codecs.open("data/lic/lic_word", "r", "utf-8") as f:
                self.vocab = f.read().strip().split(" ")

        # word_idx = defaultdict(int)
        word_idx = {}
        word_idx["<pad>"] = 0
        word_idx["<unk>"] = 1
        for i, word in enumerate(self.vocab, 2):
            word_idx[word] = i

        # <pad>:0, <unk>:1

        self.max_story_size = max([len(story) for story, _, _ in data])
        self.query_size = max([len(query) for _, query, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        # # Add time words/indexes
        # for i in range(self.memory_size):
        #     word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx)
        self.sentence_size = max(self.query_size, self.sentence_size)   # for the position
        # self.sentence_size += 1  # +1 for time words
        self.word_idx = word_idx
        self.idx_word = dict(zip(self.word_idx.values(), self.word_idx.keys()))
        self.idx_word[0] = "<pad>"
        self.idx_word[1] = "<unk>"

        self.mean_story_size = int(np.mean([len(s) for s, _, _ in data]))

        if train:
            story, query, answer = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        # answer = answer / np.expand_dims(answer.sum(axis=1), 1)
        answer.astype(np.float32)
        self.data_answer = torch.FloatTensor(answer)
        assert True, "dummy statement for debug"

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)


dataset_dir = "data/lic/"
tr_dataset = licDataset(dataset_dir, train=True)
te_dataset = licDataset(dataset_dir, train=False)
