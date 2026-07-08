import torch

torch.random.manual_seed(42)
data = torch.randint(0,10,[2,3])
print(data)
print(data.shape)
print(data.size())
print(data.shape[1])
print(data.size(1))
shape = data.reshape(1, 6)
print(shape.shape)
print(shape)
data = torch.tensor([1,2,3,4,5])
print("data-->",data.shape,data)
data = data.unsqueeze(dim=0)
print("data0维度-->",data.shape,data)
unsqueeze = data.unsqueeze(dim=1)
print("data1维度-->",unsqueeze.shape,unsqueeze)
unsqueeze = data.unsqueeze(dim=-1)
print("data-1维度-->",unsqueeze.shape,unsqueeze)
print(data.unsqueeze(dim=-1).unsqueeze(dim=-1).shape)
print(data.unsqueeze(dim=-1).unsqueeze(dim=-1).squeeze().shape)

# [2,5,7]
print(torch.transpose(data,0,1).shape)
# [5,2,7]
print(torch.transpose(torch.transpose(data,1,1),0,1).shape)
# print(torch.permute(data,[2,0,1]).shape)
print(data.permute([0,1]).shape)
data =torch.transpose(data,0,1)
print("continuous:",data)
if data.is_contiguous():
    print('T')
    print(data.view(-1))
else:
    print('F')
    print(data.contiguous().view(-1))

print(data.view(-1))

torch.random.manual_seed(42)
data1 = torch.randint(0,10,[1,3,5])
data2 = torch.randint(0,10,[2,3,5])
print(torch.cat([data1,data2],dim=0).shape)