import torch

print("# 1.基础运算")
data = torch.randint(0,10,[2,3])
print(data)

# print(data.add_(10))
# print(data)
data_div =data.div(10)
print(data_div.dtype)
data =data.float()
data.div_(10)
print(data)

print("# 2 点乘")
data1 = torch.tensor([[1,2],[3,4]])
data2 = torch.tensor([[2,2],[5,4]])
print(torch.mul(data1,data2))
print(data1*data2)

print("# 3.矩阵乘法")
data1 = torch.tensor([[1,2],[3,4]])
data2 = torch.tensor([[2,2],[5,4]])
print(torch.matmul(data1,data2))
print(data1@data2)

print("# 4.统计,指数,对数....")
data = torch.randint(0,10,[2,3],dtype=torch.float64)
print(data)
# dim 0 按列，1按行
print(data.mean(dim=1))
print(data.sum(dim=1))
print(torch.pow(data,0.5))
print(data.sqrt())
print(data.exp())
print(data.log10())

