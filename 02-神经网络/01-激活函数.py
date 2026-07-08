import torch
import matplotlib.pyplot as plt
#
# # 创建画布和坐标轴
# _, axes =plt.subplots(1,2)
# #函数图像
# x = torch.linspace(-20,20,1000)
# # 输入值x通过sigmoid函数转换成激活值y
# y =torch.sigmoid(x)
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False
# axes[0].plot(x,y)
# axes[0].grid()
# axes[0].set_title('Sigmoid函数图像')
# plt.show()
#
# # 1.sigmoid
# x = torch.linspace(-20,20,10000)
# y =torch.sigmoid(x)
# plt.plot(x,y)
# plt.grid()
# plt.show()
#
# x = torch.linspace(-20,20,1000,requires_grad=True)
# torch.sigmoid(x).sum().backward()
# plt.plot(x.detach(),x.grad)
# plt.grid()
# plt.show()

# 2.tanh
x = torch.linspace(-20,20,1000)
y =torch.tanh(x)
plt.plot(x,y)
plt.grid()
plt.show()

x = torch.linspace(-20,20,1000,requires_grad=True)
torch.tanh(x).sum().backward()
plt.plot(x.detach(),x.grad)
plt.grid()
plt.show()

# 3.relu
x = torch.linspace(-20,20,1000)
y =torch.relu(x)
plt.plot(x,y)
plt.grid()
plt.show()

x = torch.linspace(-20,20,1000,requires_grad=True)
torch.relu(x).sum().backward()
plt.plot(x.detach(),x.grad)
plt.grid()
plt.show()

# 4.softmax
scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
# print(torch.softmax(scores,dim=0))
print(torch.softmax(scores,dim=0))

