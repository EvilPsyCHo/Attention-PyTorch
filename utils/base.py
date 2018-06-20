# coding:utf8
# @Time    : 18-5-21 下午2:11
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import pickle

from utils.constant import *


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx = 0

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def update(self, x):
        if isinstance(x, str):
            self._add(x)
            self._convert()
        elif isinstance(x, list) and isinstance(x[0], str):
            for w in x:
                self._add(w)
            self._convert()
        elif isinstance(x, list) and isinstance(x[0], list):
            for s in x:
                for w in s:
                    self._add(w)
            self._convert()
        else:
            raise ValueError("input error")

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return "{}(size = {})".format(self.__class__.__name__, self.idx)

    def __repr__(self):
        return self.__str__()

    def save(self, path):
        with open(path, 'wb') as f:
            return pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)