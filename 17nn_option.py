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
optim = torch.optim.SGD(model.parameters(), lr=0.01)  # lr不能过大或者过小。刚开始的lr可设置得较大一点，后面再对lr进行调节
len = len(dataloader)

for epoch in range(20):
    total_loss = 0.0
    for imgs, targets in dataloader:
        outputs = model(imgs)
        res_loss = loss_cross(outputs, targets)

        optim.zero_grad()  # 优化器对model中的每一个参数进行梯度清零
        res_loss.backward()  # 损失反向传播
        optim.step()  # 对model参数开始调优
        total_loss += res_loss
    print('epoch:{}\ttotal_loss:{}\tmean_loss:{}.'.format(epoch, total_loss, total_loss / len))
# epoch:0	total_loss:9374.806640625	mean_loss:1.8749613761901855.
# epoch:1	total_loss:7721.240234375	mean_loss:1.544248104095459.
# epoch:2	total_loss:6830.775390625	mean_loss:1.3661550283432007.
