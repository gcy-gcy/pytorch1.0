import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class GCY(nn.Module):
    def __init__(self):
        super(GCY, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)  # 默认:stride=kernel_size,ceil_mode=False
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        return self.maxpool1(x), self.maxpool2(x)


model = GCY()

# -------------1.上图例子，查看ceil_mode为True或False的池化结果--------------- #
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
out1, out2 = model(input)
print('out1={}\nout2={}'.format(out1, out2))

# --------------2.加载数据集，并放入tensorboard查看图片----------------------- #
dataset = torchvision.datasets.CIFAR10('data', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

writer = SummaryWriter('maxpool')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('imgs', imgs, step)

    imgs, _ = model(imgs)
    writer.add_images('imgs_maxpool', imgs, step)
    step += 1

writer.close()
