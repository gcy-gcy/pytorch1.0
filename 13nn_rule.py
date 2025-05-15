import torch
from torch import nn

input = torch.tensor([[3, -1],
                      [-0.5, 1]])
input = torch.reshape(input, (1, 1, 2, 2))

relu = nn.ReLU()
input_relu = relu(input)

print('input={}\ninput_relu:{}'.format(input, input_relu))
