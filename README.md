# Attention-PyTorch

## Attention简介
Attention由  
$Attenion(Q,K,V) = \sum_{t=i}^{T}{similarity(Q,K_i)*V_i}$  

$其中:  
Q\in R^{n\times d_k}, K\in R^{n\times d_k}, V\in R^{n\times d_v}$

similarity的几种形式

1. 点乘

$similarity(Q,K_i)=softmax(\frac{QK_i^T}{\sqrt[]{d_k}})$

2. 余弦

$similarity(Q,K_i)=softmax(\frac{QK_i^T}{||Q|| * ||K||})$

3. MLP

$similarity(Q,K_i)=softmax(MLP(Q,K_i))$

![](./pic/attention_meta.jpg)

![](./pic/attention_detail.jpg)


## Reference

### Paper

1. [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. [Neural Machine Translation by Jointly Learning to Align and Translate]()

### github
1. [pytorch-attention](https://github.com/thomlake/pytorch-attention)
2. [seq2seq](https://github.com/keon/seq2seq)
3. [PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)

### Blog
1. [一文读懂「Attention is All You Need」| 附代码实现 ](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247486960&idx=1&sn=1b4b9d7ec7a9f40fa8a9df6b6f53bbfb&chksm=96e9d270a19e5b668875392da1d1aaa28ffd0af17d44f7ee81c2754c78cc35edf2e35be2c6a1&scene=21#wechat_redirect)
