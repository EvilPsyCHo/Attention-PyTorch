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
        self.uw = nn.Parameter(torch.Tensor(self.args["att_size"], 1))
        self.uw.data.uniform_(-0.1, 0.1)
        self.fc = nn.Linear(self.args[""], self.args["class_num"])

    def forward(self, x):
        x_embed = self.lookup(x)  # x_embed (batch, seq, embed)
        out, _ = self.gru(x_embed)  # out (batch, seq, hidden_size * bi)
        att_q = self.att(x)  # att_q (batch, seq, att_size)
        torch.bmm





# gru = nn.GRU(60, 20, bidirectional=True, batch_first=True)
# x = torch.randn([8, 10, 60])
# output, h = gru(x)
