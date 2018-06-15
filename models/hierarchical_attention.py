# coding:utf8
# @Time    : 18-6-14 上午11:28
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
import torch.nn.functional as F


class HAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lookup = nn.Embedding(self.args["vocab_size"],
                                   self.args["embed_dim"])
        self.gru = nn.GRU(self.args["embed_dim"],
                          self.args["hidden_size"],
                          dropout=self.args["dropout"],
                          batch_first=True,
                          bidirectional=self.args["bi"])
        self.bi = 2 if self.args["bi"] else 1
        self.att = nn.Linear(self.args["hidden_size"] * self.bi, self.args["att_size"])
        self.lookup_B = nn.Embedding(self.args["vocab_size"],
                                     self.args["att_size"])
        self.fc = nn.Linear(self.args["hidden_size"] * self.bi, self.args["class_num"])
        self.max_length = self.args["max_length"]

    def forward(self, x):
        x_embed = self.lookup(x)  # x_embed (batch, seq, embed)
        x_embed_b = self.lookup_B(x)
        out, _ = self.gru(x_embed)  # out (batch, seq, hidden_size * bi)
        att_q = self.att(out)  # att_q (batch, seq, att_size)
        att = []
        for t in range(self.max_length):
            print(att_q[:, t, :].size())
            att.append(torch.bmm(att_q[:, t, :], x_embed_b[:, t, :]))
        return att




if __name__ == "__main__":
    p = {
        "vocab_size": 1000,
        "class_num": 10,
        "bi": True,
        "hidden_size": 20,
        "embed_dim": 60,
        "att_size": 10,
        "dropout": 0.1,
        "max_length": 10,
    }
    import numpy as np
    x = np.random.randint(1, 1000, 100).reshape(-1, 10)
    x = torch.tensor(x)
    model = HAN(p)
    model(x).size()

    att = ScaledDotProductAttention(10)
