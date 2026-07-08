import torch
import torch.nn as nn

rnn =nn.RNN(input_size=128,hidden_size=64)
inputs = torch.randn(10,32,128)
hidden0 = torch.zeros(1,32,64)
y,h=rnn(inputs,hidden0)
print(y)
print(h)