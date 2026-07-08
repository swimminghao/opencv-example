import torch
import numpy as np

# 张量创建
# 1.torch.tensor()
# 1.1 标量
print("",torch.tensor(10))
# 1.2 数组
data = torch.tensor(np.random.randn(2,3))
print("# 1.2 数组",data)
# 1.3 列表
list = torch.tensor([[-0.8222, -0.3650,  0.0365],[-0.1228,  0.0695, -0.8740]])
print("# 1.3 列表",list)

# 2.torch.Tensor()
# 2.1 形状
print("# 2.1 形状",torch.Tensor(2,3))

# 2.2 指定数据
print("# 2.2 指定数据",torch.Tensor([2,3]))

# 3.指定类型的数据
print("# 3.指定类型的数据",torch.ShortTensor(2,3))
print("# 3.指定类型的数据",torch.IntTensor([2.8,3.4]))

# 线性张量
print("# 线性张量",torch.arange(0,10,1.5))
print("# 线性张量",torch.linspace(0,10,9))

# 随机张量
# （0附近）标准正太分布
print("# 随机张量",torch.randn(2,3))
print("# 随机张量",torch.random.initial_seed())
torch.random.manual_seed(torch.random.initial_seed())
print("# 随机张量",torch.randn(2,3))
torch.random.manual_seed(100)
print("# 随机张量",torch.randn(2,3))
torch.random.manual_seed(100)
print("# 随机张量",torch.randn(2,3))


# 0/1/指定值
data = torch.randn(2,3)
print("# 0/1/指定值",torch.zeros(4,5))
print("# 0/1/指定值",torch.zeros_like(data))

print("# 0/1/指定值",torch.ones(4,5))
print("# 0/1/指定值",torch.ones_like(data))

print("# 0/1/指定值",torch.full([4,5],10))
print("# 0/1/指定值",torch.full_like(data,20))

# 元素类型转换
data = torch.randn(2,3)
print("# 元素类型转换",data.dtype)
print("# 元素类型转换",data.type(torch.IntTensor).dtype)
print("# 元素类型转换",data.int().dtype)

# 张量->ndarray
data_tensor =torch.ones(4,5)
print("# 张量->ndarray",type(data_tensor))
data_np=data_tensor.numpy().copy()
print("# 张量->ndarray",type(data_np))
data_np[0][0]=100
print("# 张量->ndarray",data_tensor)
print("# 张量->ndarray",data_np)

# ndarray->张量
data_np=np.array([1,1,1])
print("# ndarray->张量",type(data_np))
# data_tensor=torch.from_numpy(data_np.copy())
data_tensor = torch.tensor(data_np)
print("# ndarray->张量",type(data_tensor))
data_np[0]= 20
print("# ndarray->张量",data_np)
print("# ndarray->张量",data_tensor)

# 张量->数值
data_tensor = torch.tensor(30)
print("# 张量->数值",data_tensor.item())


