import os
import sys
from collections import defaultdict
from consts import global_consts as gc
import torch
# from mmsdk import mmdatasdk as md
from nltk.tokenize import word_tokenize


# if gc.SDK_PATH is None:
#     print("SDK path is not specified! Please specify first in constants/paths.py")
#     exit(0)
# else:
#     print("Added gc.SDK_PATH")
#     import os
#
#     print(os.getcwd())
#     sys.path.append(gc.SDK_PATH)
#
# DATASET = md.cmu_mosei
# train_split = DATASET.standard_folds.standard_train_fold
# dev_split = DATASET.standard_folds.standard_valid_fold
# test_split = DATASET.standard_folds.standard_test_fold

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.word_freq = {}
        self.idx2word = ['<unk>']

    def count_word(self, word):
        if word not in self.word_freq:
            self.word_freq[word] = 0
        else:
            self.word_freq[word] += 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        word_ids = self.tokenize(path)
        print("#unique words: %d" % len(self.dictionary.word2idx))
        total = len(word_ids)
        train = total//100 * 70
        valid = total//100 * 10
        self.train = word_ids[:train]
        self.valid = word_ids[train:train + valid]
        self.test = word_ids[train + valid:]

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        word_ids = []
        # Add words to the dictionary
        with open(os.path.join(path, 'mosei_lang.txt'), 'r', encoding="utf8") as f:
            num_tokens = 0
            for line in f:
                tokens = word_tokenize(line)
                words = [word.lower() for word in tokens if word.isalpha()] + ['<eos>']
                num_tokens += len(words)
                for word in words:
                    self.dictionary.count_word(word)

        with open(os.path.join(path, 'mosei_lang.txt'), 'r', encoding="utf8") as f:
            num_tokens = 0
            for line in f:
                tokens = word_tokenize(line)
                words = [word.lower() for word in tokens if word.isalpha()] + ['<eos>']
                num_tokens += len(words)
                for word in words:
                    if self.dictionary.word_freq[word] > 2:
                        word_ids.append(self.dictionary.add_word(word))
                    else:
                        word_ids.append(0)

        # torch.LongTensor(word_ids)
        # # Tokenize file content
        # with open(os.path.join(path, 'mosei_lang.txt'), 'r', encoding="utf8") as f:
        #     num_tokens = 0
        #     for line in f:
        #         tokens = word_tokenize(line)
        #         words = [word.lower() for word in tokens if word.isalpha()] + ['<eos>']
        #         for word in words:
        #             word_ids[num_tokens] = self.dictionary.word2idx[word]
        #             num_tokens += 1
        return torch.LongTensor(word_ids)

    def tokens_to_words(self, input):
        seq = []
        for words in input:
            for token in words:
                # print(token.shape)
                seq += [self.dictionary.idx2word[int(token.item())]]
        seq = " ".join(seq)
        return seq


