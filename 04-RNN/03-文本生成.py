import torch
import re
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# 1.加载数据
# 1.1 分词+去重
# 1.2 词表
def build_vocab():
    file_name = '/Users/mac/Desktop/AI 17期深度学习/02.code/04-RNN/data/jaychou_lyrics.txt'
    all_words = []
    u_words= []
    for line in open(file_name,'r'):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in u_words:
                u_words.append(word)
    # print(len(all_words))
    # print(len(u_words))
    word_to_idx={word:idx for idx,word in enumerate(u_words)}
    corpus_idx = []
    for words in all_words:
        temp =[]
        for word in words:
            temp.append(word_to_idx[word])
        temp.append(word_to_idx[' '])
        corpus_idx.extend(temp)
    return u_words,word_to_idx,len(u_words),corpus_idx

# 1.3 构建可迭代对象(相当于TensorDatadset)(重点)
class SongDataSet(torch.utils.data.Dataset):
    def __init__(self,corpus_idx,num_char):
        super(SongDataSet, self).__init__()
        self.corpus_idx = corpus_idx
        self.num_char = num_char
        self.wordcount = len(self.corpus_idx)
        self.number = self.wordcount//self.num_char

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start =min(max(0,idx),self.wordcount-self.num_char-10)
        x = self.corpus_idx[start:start+self.num_char]
        y = self.corpus_idx[start+1:start+1+self.num_char]
        return torch.tensor(x),torch.tensor(y)


# 2.构建模型
class TextGenerator(nn.Module):
    def __init__(self, word_count):
        super(TextGenerator, self).__init__()
        # 初始化词嵌入层: 词向量的维度为128
        self.ebd = nn.Embedding(word_count, 128)
        # 循环网络层: 词向量维度 128, 隐藏向量维度 128, 网络层数1
        self.rnn = nn.RNN(128,256,1)
        # 输出层: 特征向量维度128与隐藏状态维度相同,词表中词的个数
        self.out = nn.Linear(256, word_count)

    def forward(self, inputs, hidden):
        # 输出维度: (batch, seq_len, 128)
        embed = self.ebd(inputs)
        # 修改维度: (seq_len, batch, 128)
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)

        # 输入维度: (seq_len*batch, 128) 输出维度: (seq_len*batch, 5682)
        output = self.out(output.reshape((-1,output.shape[-1])))
        # 网络输出结果
        return output, hidden

    def init_hidden(self,bs):
        # 隐藏层的初始化:[seq_len, batch, 隐藏层向量维度]
        return torch.zeros(1, bs,256)



# 3.模型训练
# 模型训练
def train():
    # 构建词典
    index_to_word, word_to_index, word_count, corpus_idx = build_vocab()
    # 数据集
    lyrics = SongDataSet(corpus_idx, 32)
    # 初始化模型
    model = TextGenerator(word_count)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练轮数
    epoch = 2
    # 开始训练
    for epoch_idx in range(epoch):
        # 数据加载器
        lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=1)
        # 训练时间
        start = time.time()
        # 迭代次数
        iter_num = 0
        # 训练损失
        total_loss = 0.0
        # 遍历数据集
        for x, y in lyrics_dataloader:
            # 隐藏状态的初始化
            hidden = model.init_hidden(bs=1)
            # 模型计算
            output, hidden = model(x, hidden)
            # 计算损失
            # y:[batch,seq_len]->[seq_len,batch]->[seq_len*batch]
            y = torch.transpose(y, 0, 1).contiguous().view(-1)
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 迭代次数加1
            iter_num += 1
            total_loss += loss.item()
        # 打印训练信息
        print('epoch %3s loss: %.5f time %.2f' % \
                  (epoch_idx + 1,
                   total_loss / iter_num,
                   time.time() - start))
    # 模型存储
    torch.save(model.state_dict(), 'data/lyrics_model_%d.pth' % epoch)

# 4.模型预测,模型评估
def predict(start_word, sentence_length):
    # 构建词典
    index_to_word, word_to_index, word_count, _ = build_vocab()
    # 构建模型
    model = TextGenerator(word_count)
    # 加载参数
    model.load_state_dict(torch.load('data/lyrics_model_2.pth'))
    # 隐藏状态
    hidden = model.init_hidden(bs=1)
    # 将起始词转换为索引
    word_idx = word_to_index[start_word]
    # 产生的词的索引存放位置
    generate_sentence = [word_idx]
    # 遍历到句子长度，获取每一个词
    for _ in range(sentence_length):
        # 模型预测
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 获取预测结果
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)
    # 根据产生的索引获取对应的词，并进行打印
    for idx in generate_sentence:
        print(index_to_word[idx], end='')

if __name__ == '__main__':
    u_words, word_to_idx, count, corpus_idx= build_vocab()
    # print(u_words)
    # print(count)
    # print(word_to_idx)
    # print(corpus_idx)
    # dataset =SongDataSet(corpus_idx,10)
    # x,y = dataset.__getitem__(0)
    # print(x)
    # print(y)
    # train()
    predict('分手',50)