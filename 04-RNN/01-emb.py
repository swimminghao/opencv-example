import torch
import torch.nn as nn
import jieba

# 语料
text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
# 分词+去重
words=jieba.lcut(text)
words_u = list(set(words))

print(len(words_u))
print(words_u)
# emb
embs=nn.Embedding(18,3)
print(embs)
# 获取词向量
print(embs(torch.tensor(0)))
