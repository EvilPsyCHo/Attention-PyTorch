# coding:utf8
# @Time    : 18-5-21 下午2:32
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


def read_corpus(path, precessing_func=None):
    s = []
    label = []
    with open(path, 'r') as f:
        data = f.readlines()
    for i in data:
        if precessing_func:
            i = precessing_func(i)
        label.append(i.split(" ")[0])
        s.append(i.strip("\n").split(" ")[1:])
    return s, label
