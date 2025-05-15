import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms

# 1. 加载数据
dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


# 2. 构造模型
class GCY(nn.Module):
    def __init__(self):
        super(GCY, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)

    def forward(self, x):
        return self.conv1(x)


writer = SummaryWriter('./logs/Conv2d')

# 3. 实例化一个模型对象，进行卷积
model = GCY()
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images('imgs_ch3', imgs, step)

# 4. 用tensorboard打开查看图像。但是注意，add_images的输入图像的通道数只能是3
#    所以如果通道数>3，则可以先采用小土堆的这个不严谨的做法，在tensorboard中查看一下图片
    outputs = model(imgs)
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images('imgs_ch6', outputs, step)

    step += 1

writer.close()
