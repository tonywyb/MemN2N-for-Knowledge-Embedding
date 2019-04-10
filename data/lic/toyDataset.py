from itertools import chain
import numpy as np
import torch
import torch.utils.data as data
from utils.data_utils import load_task, vectorize_data
from six.moves import range
from preprocess.preprocess import pre_process


class licDataset(data.Dataset):
    def __init__(self, dataset_dir, memory_size=50, train=True):
        train_data, test_data = pre_process(dataset_dir)
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
        self.idx_word = dict(zip(self.word_idx.values(), self.word_idx.keys()))

        self.mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))

        if train:
            story, query, answer = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        # answer = answer / np.expand_dims(answer.sum(axis=1), 1)
        self.data_answer = torch.FloatTensor(answer)
        assert True, "dummy statement for debug"

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)


dataset_dir = "data/lic/train_part.json"
tr_dataset = licDataset(dataset_dir, train=True)
te_dataset = licDataset(dataset_dir, train=False)
