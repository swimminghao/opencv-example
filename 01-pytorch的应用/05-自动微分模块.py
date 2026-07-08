import torch

def test():
    x = torch.tensor(5.)
    y = torch.tensor(0.)
    w = torch.tensor(1.,requires_grad=True)
    b = torch.tensor(3.,requires_grad=True)
    z = x*w +b
    loss = torch.nn.MSELoss()
    loss =loss(z,y)
    loss.backward()
    print(w.grad)
    print(b.grad)


def test2():
    x = torch.ones(3,5)
    y = torch.zeros(3,3)
    w = torch.randn(5,3,requires_grad=True)
    b = torch.randn(3,requires_grad=True)
    z = torch.matmul(x,w)+b
    loss = torch.nn.MSELoss()
    loss =loss(z,y)
    loss.backward()
    print(w.grad)
    print(b.grad)

if __name__ == '__main__':
    test2()