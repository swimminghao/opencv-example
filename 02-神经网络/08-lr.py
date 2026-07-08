import torch
import torch.optim as opt
import matplotlib.pyplot as plt

print("1.等间隔学习率.........")
LR = 1
x = torch.tensor([1.0])
w = torch.tensor([1.0],requires_grad=True)
y = torch.tensor([0.0])

optim = opt.SGD([w],lr=LR,momentum=0.9)
lr_s=opt.lr_scheduler.StepLR(optim,step_size=50,gamma=0.1)
lr_list = []
epoch_list = []
for epoch in range(200):
    epoch_list.append(epoch)
    lr_list.append(lr_s.get_last_lr())
    for iter in range(10):
        loss = (w*x-y)**2/2.0
        optim.zero_grad()
        loss.backward()
        optim.step()
    lr_s.step()

plt.plot(epoch_list,lr_list)
plt.grid()
plt.show()

print("2.指定间隔变化学习率.........")
LR = 1
x = torch.tensor([1.0])
w = torch.tensor([1.0],requires_grad=True)
y = torch.tensor([0.0])

optim = opt.SGD([w],lr=LR,momentum=0.9)
lr_s=opt.lr_scheduler.MultiStepLR(optim,milestones=[50,80,120],gamma=0.1)
lr_list = []
epoch_list = []
for epoch in range(200):
    epoch_list.append(epoch)
    lr_list.append(lr_s.get_last_lr())
    for iter in range(10):
        loss = (w*x-y)**2/2.0
        optim.zero_grad()
        loss.backward()
        optim.step()
    lr_s.step()

plt.plot(epoch_list,lr_list)
plt.grid()
plt.show()

print("3.按指数衰减，调整学习率.........")
LR = 1
x = torch.tensor([1.0])
w = torch.tensor([1.0],requires_grad=True)
y = torch.tensor([0.0])

optim = opt.SGD([w],lr=LR,momentum=0.9)
lr_s=opt.lr_scheduler.ExponentialLR(optim,gamma=0.99)
lr_list = []
epoch_list = []
for epoch in range(200):
    epoch_list.append(epoch)
    lr_list.append(lr_s.get_last_lr())
    for iter in range(10):
        loss = (w*x-y)**2/2.0
        optim.zero_grad()
        loss.backward()
        optim.step()
    lr_s.step()

plt.plot(epoch_list,lr_list)
plt.grid()
plt.show()