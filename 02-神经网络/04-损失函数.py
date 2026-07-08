import torch
import torch.nn as nn

# 1.交叉熵
y_true = torch.tensor([1, 2], dtype=torch.int64)
y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float32)
y_pred = torch.tensor([[0.2, 6, 0.2], [0.1, 0.8, 10]], dtype=torch.float32)
loss = nn.CrossEntropyLoss()
print(loss(y_pred,y_true).numpy())

# 2.二分类交叉熵
y_pred = torch.tensor([0.02, 0.8, 0.01], requires_grad=True)
y_true = torch.tensor([0, 1, 0], dtype=torch.float32)
loss = nn.BCELoss()
print(loss(y_pred,y_true))

# # 3.回归
y_pred = torch.tensor([1.0, 1.0, 1.9])
y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
# L1
loss =nn.L1Loss()
print(loss(y_pred,y_true))
# L2
loss =nn.MSELoss()
print(loss(y_pred,y_true))
# smoothL1
loss =nn.SmoothL1Loss()
print(loss(y_pred,y_true))