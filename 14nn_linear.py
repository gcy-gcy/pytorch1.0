import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载数据集
dataset = torchvision.datasets.CIFAR10('data', train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 修正后的模型定义
class GCY(nn.Module):
    def __init__(self):
        super(GCY, self).__init__()
        # 计算输入特征数：3通道×32高×32宽 = 3072
        self.linear = nn.Linear(3072, 10)

    def forward(self, x):
        # 展平张量：[batch_size, 3, 32, 32] -> [batch_size, 3072]
        x = torch.flatten(x, start_dim=1)
        output = self.linear(x)
        return output


# 初始化模型
model = GCY()

# 创建TensorBoard写入器
writer = SummaryWriter('Linear')
step = 0

for data in dataloader:
    imgs, targets = data

    # 记录原始图像
    writer.add_images('original_imgs', imgs, step)

    # 处理图像
    batch_size = imgs.shape[0]
    # 正确展平图像：[batch_size, 3, 32, 32] -> [batch_size, 3072]
    flattened_imgs = torch.flatten(imgs, start_dim=1)

    # 通过模型
    outputs = model(imgs)  # 模型内部会自动展平

    # 记录处理后的图像（这里记录的是预测结果的可视化）
    # 注意：线性层输出是logits，通常需要经过softmax或可视化处理
    # 这里简化处理，仅作示例
    writer.add_images('processed_imgs', imgs, step)

    step += 1

writer.close()