# 导入相关模块
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt

# 1.构建数据集
def create_dataset():
    x,y,coef =make_regression(n_samples=100,n_features=1,noise=10,bias=1.5,coef=True)
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x,y,coef

x,y,coef = create_dataset()
dataset=TensorDataset(x,y)
dataloader =DataLoader(dataset=dataset,batch_size=16,shuffle=True)

# 2.构建模型
model =nn.Linear(in_features=1,out_features=1)

# 3.损失函数和优化器设置
loss = nn.MSELoss()
opt = optim.SGD(params=model.parameters(),lr=0.0001)

# 4.模型训练
loss_list = []
total_loss = 0
num = 0
for epoch in range(10):
    for train_x,train_y in dataloader:
        y_pred =model(train_x.float())
        MSE_loss =loss(y_pred,train_y.reshape(-1,1).float())
        total_loss+=MSE_loss.item()
        num += len(train_y)
        # 在 PyTorch 中，.backward() 计算梯度时，默认会累加（累加，不是覆盖）到参数的 .grad 属性上。也就是说，多次调用 .backward() 后，梯度会不断叠加。
        #
        # 这种设计是有意为之的：它允许你实现“梯度累积”技术（例如，用多个小 batch 的梯度来模拟一个大 batch 的训练）。但在标准的训练循环中，每个 mini-batch 应该使用该 batch 单独计算的梯度来更新参数，因此需要先清零上一步的梯度。
        opt.zero_grad()
        MSE_loss.backward()
        opt.step()
    loss_list.append(total_loss/num)

plt.plot(range(10),loss_list)
plt.show()
print("coef:",coef)

print("model.weight:",model.weight)
print("model.bias:",model.bias)







# if __name__ == '__main__':
#     x,y,coef=create_dataset()
#     plt.scatter(x,y)
#     x = torch.linspace(x.min(),x.max(),1000)
#     y = torch.tensor([i*coef+1.5 for i in x])
#     plt.plot(x,y)
#     plt.grid()
#     plt.show()
