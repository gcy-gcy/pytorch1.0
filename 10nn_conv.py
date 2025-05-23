import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(input.shape)
print(kernel.shape)

# input、kernel都扩充到4维
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

#
out = F.conv2d(input, kernel, stride=1)
print('out={}'.format(out))

out2 = F.conv2d(input, kernel, stride=2)
print('out2={}'.format(out2))
 
out3 = F.conv2d(input, kernel, stride=1, padding=1)
print('out3={}'.format(out3))
