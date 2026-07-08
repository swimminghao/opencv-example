import torch
import torch.nn as nn
# 优化器
import torch.optim as opt

# Back propagation 反向传播
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(2,2)
        self.layer2 = nn.Linear(2,2)
        self.layer1.weight.data = torch.tensor([[0.15,0.20],[0.25,0.30]])
        self.layer1.bias.data = torch.tensor([0.35,0.35])
        self.layer2.weight.data = torch.tensor([[0.40, 0.45], [0.5, 0.55]])
        self.layer2.bias.data = torch.tensor([0.60, 0.60])

    def forward(self,x):
        x1 =self.layer1(x)
        x1_s = torch.sigmoid(x1)
        x2 =self.layer2(x1_s)
        out = torch.sigmoid(x2)
        return out

# 1.实例化
model =Model()
print("神经网络参数：",model.state_dict())
# 2.数据
inputs = torch.tensor([[0.05,0.10]])
target = torch.tensor([[0.01,0.99]])
# 3.送入网络中
y =model(inputs)

# 4.backward+opt.step()
loss = torch.sum((y-target)**2)/2
opt =opt.SGD(model.parameters(),lr=0.5)
opt.zero_grad()
loss.backward()
print("w1,w2,w3,w4：",model.layer1.weight.grad.data)
print("w5,w6,w7,w8：",model.layer2.weight.grad.data)
opt.step()
print("一次BP之后的神经网络W0、W1等参数：",model.state_dict())




