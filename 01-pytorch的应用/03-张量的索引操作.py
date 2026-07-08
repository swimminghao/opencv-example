import torch
torch.random.manual_seed(42)
data =torch.randint(0,10,[4,5])
print(data)
print(data[1])
print(data[:,1])
print(data[[1,2],[2,3]])
print(data[[[1],[2]],[2,3]])
# 起始值:终止值:步长
print(data[0:3:2,:2])
print(data[:,2]>5)
print(data[0]>5)
print(data[data[:,2]>5,data[0]>5])
torch.random.manual_seed(42)
data =torch.randint(0,10,[4,5,6])
print(data)
print(data[1])
print(data[:,:,1])
print(data[:,1,:])


