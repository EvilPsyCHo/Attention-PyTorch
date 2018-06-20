# coding:utf8
# @Time    : 18-5-21 上午11:46
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np

from utils.constant import *
from utils.base import Dictionary


class Vocabulary(Dictionary):
    def __init__(self):
        super().__init__()
        self.word2idx = {PAD: PAD_IDX, UNK: UNK_IDX}
        self.idx = len(ALL_CONSTANT)

    def transform(self, x, max_length=None):

        if isinstance(x, str):
            return self.word2idx.get(x, UNK_IDX)
        elif isinstance(x, list) and isinstance(x[0], str):
            ri = [self.word2idx.get(w, UNK_IDX) for w in x]
            if max_length is not None:
                ri = ri[:max_length]
                ri = [PAD_IDX] * (max_length - len(ri)) + ri
            result = [ri]
            return result
        else:
            result = []
            for s in x:
                ri = [self.word2idx.get(w, UNK_IDX) for w in s]
                if max_length is not None:
                    ri = ri[:max_length]
                    ri = [PAD_IDX] * (max_length - len(ri)) + ri
                result.append(ri)
            return result


if __name__ == "__main__":
    s1 = [["我", "吃"], ["吃", "什么"]]
    s2 = [["我", "haha"], ["吃", "什么"]]
    d = Vocabulary()
    d.update(s1)
    d.update(s2)
    d.transform(s1, max_length=10)
    # d.save("/home/zhouzr/dict")
