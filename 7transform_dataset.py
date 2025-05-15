import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='./data',train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data',train=False, transform=dataset_transform, download=True)
#train默认为true false则为测试集
#download 设置为true则自动从官网下载


# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#print(test_set[0])

writer = SummaryWriter('CIFAR10')
for i in range(10):
    img, label = test_set[i]  # test_set[i]返回的依次是图像(PIL.Image)和类别(int)
    writer.add_image('test_set', img, i)

writer.close()
