import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#drop——last设置为false最后为半  true（不足一张直接舍弃）

#测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('dataloader')
step = 0
for data in test_loader:
    img, target = data
    #print(img.shape)
    #print(target)
    writer.add_images("test_data", img, step)
    step += 1

writer.close()