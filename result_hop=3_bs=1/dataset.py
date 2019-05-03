from itertools import chain
import numpy as np
import torch
import torch.utils.data as data
from utils.data_utils import vectorize_data
from preprocess.preprocess import pre_process, extract_knowledge
import os
import pickle as pkl


class licDataset(data.Dataset):
    def __init__(self, dataset_dir, memory_size=50, mode="train"):
        data_path = os.path.join(dataset_dir, "train+test.json")
        dict_path = os.path.join(dataset_dir, "word_idx.pkl")
        extract_knowledge(data_path)
        train_data, test_data, val_data = pre_process(data_path, ground_filter=True)
        data = train_data + test_data + val_data

        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_id = 0
        self.pad_id = 1
        self.sos_id = 2
        self.eos_id = 3

        if not os.path.exists(dict_path):
            self.vocab = list()
            for story, query, answer in data:
                self.vocab = self.vocab + (list(chain.from_iterable(story)) + query + answer)
            vocab_dict = dict()
            for v in set(self.vocab):
                vocab_dict[v] = self.vocab.count(v)
            vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1], reverse=True))
            with open(os.path.join(dataset_dir, "word_cnt.pkl"), "wb") as f:
                pkl.dump(vocab_dict, f)
            self.vocab = list(vocab_dict.keys())
            token_lis = [(self.unk_token, self.unk_id), (self.pad_token, self.pad_id),
                         (self.sos_token, self.sos_id), (self.eos_token, self.eos_id)]
            vocab_len = len(vocab_dict) if len(vocab_dict) <= 10000 - len(token_lis) else 10000 - len(token_lis)
            self.word_idx = dict(token_lis)
            for i, word in enumerate(self.vocab[:vocab_len], 4):
                self.word_idx[word] = i
            vocab_len += len(token_lis)
            assert len(self.word_idx) == vocab_len
            self.idx_word = dict(zip(self.word_idx.values(), self.word_idx.keys()))
            with open(dict_path, "wb") as f:
                pkl.dump(self.word_idx, f)
        else:
            with open(dict_path, "rb") as f:
                self.word_idx = pkl.load(f)
                self.idx_word = dict(zip(self.word_idx.values(), self.word_idx.keys()))

        self.max_story_size = max([len(story) for story, _, _ in data])
        self.query_size = max([len(query) for _, query, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        self.num_vocab = len(self.word_idx)
        self.sentence_size = max(self.query_size, self.sentence_size)   # for the position
        self.sentence_size += 2     # +2 for <sos>, <eos>

        self.mean_story_size = int(np.mean([len(s) for s, _, _ in data]))

        if mode == "train":
            story, query, answer = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        elif mode == "test":
            story, query, answer = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)
        elif mode == "valid":
            story, query, answer = vectorize_data(val_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        answer.astype(np.float32)
        self.data_answer = torch.FloatTensor(answer)

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)


dataset_dir = "data/lic/"
tr_dataset = licDataset(dataset_dir, mode="train")
te_dataset = licDataset(dataset_dir, mode="test")
val_dataset = licDataset(dataset_dir, mode="valid")
