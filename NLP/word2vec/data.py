import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class Reader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, input_file_name, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.label2id = dict()
        self.id2label = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.input_file_name = input_file_name
        self.read_words(min_count)
        self.init_table_negatives()
        self.init_table_discards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in tqdm(open(self.input_file_name, encoding="utf8"),
                         total=len(open(self.input_file_name, encoding="utf8").readlines())):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.label2id[w] = wid
            self.id2label[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.label2id)))

    def init_table_discards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def init_table_negatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * Reader.NEGATIVE_TABLE_SIZE)
        for word_id, word_count in enumerate(count):
            self.negatives += [word_id] * int(word_count)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negatives(self, x, size):
        if x.shape == ():
            x_size = 1
        else:
            x_size = x.shape[0]

        response = self.negatives[self.negpos: self.negpos + x_size * size]
        self.negpos = (self.negpos + size) % len(self.negatives)

        if len(response) != x_size * size:
            out = np.concatenate(
                (response, np.random.choice(self.negatives, size=(x_size * size - response.shape[0],))))
            out = out.reshape(x_size, size)
            out[out == np.repeat(x.T, size, axis=0).reshape(x_size, size)] = np.random.randint(0,
                                                                                               len(self.id2label) - 1)
            return out

        response = response.reshape(x_size, size)
        response[response == np.repeat(x.T, size, axis=0).reshape(x_size, size)] = np.random.randint(0,
                                                                                                     len(self.id2label) - 1)
        return response


class Word2VecDataset(Dataset):
    def __init__(self, data, window_size):
        self.articles = None
        self.data = data
        self.window_size = window_size
        self.openfile()

    def openfile(self):
        input_file = open(self.data.input_file_name, encoding="utf8").read()
        self.articles = []
        for line in tqdm(input_file.splitlines()):
            splitted_line = line.split()
            if len(splitted_line) > 4:
                article = []
                for word in line.split():
                    if word in self.data.label2id:
                        article += [[self.data.label2id[word], self.data.discards[self.data.label2id[word]]]]
                self.articles += [article]

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        words = np.array(self.articles[idx])

        discard_vals = np.random.rand(words.shape[0])

        word_ids = words[discard_vals < words[:, 1]][:, 0]

        boundary = np.random.randint(2, self.window_size)

        word_pairs = np.array([[u, v] for i, u in enumerate(word_ids) for j, v in
                               enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v])
        if word_pairs.shape == (0,):
            return np.array([], dtype=np.int64).reshape(0, 7)
        word_negatives = self.data.get_negatives(word_pairs[:, 1], 5)
        out = np.concatenate((word_pairs, word_negatives), axis=1).astype(np.int32)
        return out


def collate_fn(batches):
    batches = np.concatenate(batches, axis=0)

    out = (torch.LongTensor(batches[:, 0]), torch.LongTensor(batches[:, 1]), torch.LongTensor(batches[:, 2:]))
    return out
