import torch.nn as nn
import torch

class Model(nn.Module):
    # 初始化
    def __init__(self):
        super(Model,self).__init__()
        self.layer1 = nn.Linear(3,3)
        self.layer2 = nn.Linear(3,2)
        self.output = nn.Linear(2,2)

    # 前向传播
    def forward(self,x):
        x =torch.relu(self.layer1(x))
        x =torch.relu(self.layer2(x))
        x =self.output(x) # scores/logits
        out =torch.softmax(x,dim=-1)
        return out


net = Model()
x = torch.randn(10,3)
out =net(x)
print("outShape：",out.shape)

from torchsummary import summary
summary(net, input_size=(3,), batch_size=10, device='cpu')
# summary(net, input_size=(3,), batch_size=10, device='cpu')
# for name,params in net.named_parameters():
#     print(name,params)
