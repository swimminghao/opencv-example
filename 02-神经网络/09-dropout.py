import torch
import torch.nn as nn
# dropout正则化（随机失活）
x = torch.randint(1,10,[1,4],dtype=torch.float32)
print(x)
layer=nn.Linear(4,5)
y = layer(x)
print("未失活fc层：",y)
layer2=nn.Dropout(p=0.2)
out = layer2(y)
print("失活fc层",out)