# coding:utf8
# @Time    : 18-6-15 下午2:17
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


if __name__ == "__main__":
    att = ScaledDotProductAttention(5)
    q = torch.randn(32, 10, 10)
    k = torch.randn(32, 10, 10)
    v = torch.randn(32, 10, 10)
    out, att = att(q, k, v)
    print("out size", out.size())
    print("att size", att.size())
