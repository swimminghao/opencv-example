import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary

# 1.数据集
def create_dataset():
    data = pd.read_csv('/home/xh/workspaces/pycharm/deep-learning/02-神经网络/data/手机价格预测.csv')
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    train_dataset=TensorDataset(torch.from_numpy(x_train.values),torch.from_numpy(y_train.values))
    valid_dataset=TensorDataset(torch.from_numpy(x_test.values),torch.from_numpy(y_test.values))
    return train_dataset,valid_dataset,x_train.shape,x_test.shape

# 2.模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(20,64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,4)

    def forward(self,x):
        x1 =self.layer1(x)
        x1_s=torch.relu(x1)
        x2 = self.layer2(x1_s)
        x2_s = torch.relu(x2)
        out =self.layer3(x2_s)
        return out


# 3.训练
def train():
    train_dataset, valid_dataset, train_shape, test_shape = create_dataset()
    model = Model()

    loss = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

    for epoch in range(100):
        dataloader =DataLoader(train_dataset,shuffle=True,batch_size=16)
        loss_total = 0
        iter = 0
        for x,y in dataloader:
            pre =model(x)
            loss_values = loss(pre,y)
            loss_total += loss_values.item()
            iter+=1
            opt.zero_grad()
            loss_values.backward()
            opt.step()

        print(loss_total/(iter+0.01))

    # 保存所有state_dict 参数
    torch.save(model.state_dict(),'/home/xh/workspaces/pycharm/deep-learning/02-神经网络/data/phone3.pth')



# 4.预测

def test():
    train_dataset, valid_dataset, train_shape, test_shape = create_dataset()
    model = Model()
    model.load_state_dict(torch.load('/home/xh/workspaces/pycharm/deep-learning/02-神经网络/data/phone3.pth'))

    dataloader =DataLoader(valid_dataset,batch_size=8,shuffle=False)

    correct = 0
    for x,y in dataloader:
        out = model(x)
        # print(out)
        # print(torch.softmax(out, dim=-1))
        # break
        y_pred =torch.argmax(out,dim=1)
        # correct 预测正确的个数
        correct += (y_pred==y).sum()
        # c_num+=correct

    print(correct/len(valid_dataset))

if __name__ == '__main__':
    train_dataset,valid_dataset,train_shape,test_shape=create_dataset()
    print(train_shape)
    print(test_shape)

    model = Model()
    summary(model,input_size=(20,),batch_size=4,device='cpu')
    train()
    test()