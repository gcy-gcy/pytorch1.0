import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):  # 模型前向传播
        return self.model(x)


model = Model()  # 定义模型
loss_cross = nn.CrossEntropyLoss()  # 定义损失函数

for data in dataloader:
    imgs, targets = data
    outputs = model(imgs)
    # print(outputs)    # 先打印查看一下结果。outputs.shape=(2, 10) 即(N,C)
    # print(targets)    # target.shape=(2) 即(N)
    # 观察outputs和target的shape，然后选择使用哪个损失函数
    res_loss = loss_cross(outputs, targets)
    res_loss.backward()  # 损失反向传播
    print(res_loss)

#
# inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# targets = torch.tensor([1, 2, 5], dtype=torch.float32)
#
# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))
#
# # -------------L1Loss--------------- #
# loss = nn.L1Loss()
# res = loss(inputs, targets)  # 返回的是一个标量,ndim=0
# print(res)  # tensor(1.6667)
#
# # -------------MSELoss--------------- #
# loss_mse = nn.MSELoss()
# res_mse = loss_mse(inputs, targets)
# print(res_mse)
#
# # -------------CrossEntropyLoss--------------- #
# x = torch.tensor([0.1, 0.2, 0.3])  # (N,C)
# x = torch.reshape(x, (1, 3))
# y = torch.tensor([1])  # (N)
# loss_cross = nn.CrossEntropyLoss()
# res_cross = loss_cross(x, y)
# print(res_cross)
