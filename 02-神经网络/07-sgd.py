import torch
import torch.optim as opt

print("1.momentum.......")
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
y = ((w ** 2) / 2.0).sum()

opt1 = opt.SGD([w], lr=0.01, momentum=0.9)

opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)


y = ((w ** 2) / 2.0).sum()
opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())


print("2.AdaGrad.......")
# 梯度的指数加权平均
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
y = ((w ** 2) / 2.0).sum()

opt1 = opt.Adagrad([w], lr=0.01)
# opt = opt.RMSprop([w],lr=0.01,alpha=0.99)
# opt = opt.Adam([w],lr=0.01,betas=[0.9,0.99])

opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())

y = ((w ** 2) / 2.0).sum()
opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())

print("3.RMSprop.......")
# 梯度平方的指数加权平均
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
y = ((w ** 2) / 2.0).sum()

opt1 = opt.RMSprop([w],lr=0.01,alpha=0.99)
# opt = opt.Adam([w],lr=0.01,betas=[0.9,0.99])

opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())

y = ((w ** 2) / 2.0).sum()
opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())

print("3.Adam.......")
# momentum和rmsprop合体
# momemtum使用指数加权平均计算当前的梯度值
# rmsprop 使用自适应的学习率
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
y = ((w ** 2) / 2.0).sum()

opt1 = opt.Adam([w],lr=0.01,betas=[0.9,0.99])

opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())

y = ((w ** 2) / 2.0).sum()
opt1.zero_grad()
y.backward()
opt1.step()
print(w.grad)
print(w.detach())