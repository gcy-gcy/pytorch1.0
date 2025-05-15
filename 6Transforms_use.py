from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

image_path = 'images/gcy.jpg'
image = Image.open(image_path)

writer = SummaryWriter('gyx2')

# 1.Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(image)
writer.add_image('ToTensor', img_tensor)  # 这里只传入了tag和image_tensor，没有写入第3个参数global_step，则会默认是第0步

# 2.Normalize 可以改变色调
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm)

trans_norm = transforms.Normalize([1, 1, 4], [3, 2, 1])
img_norm_2 = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm_2, 1)

trans_norm = transforms.Normalize([2, 2, 2], [5, 2.6, 1.5])
img_norm_3 = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm_3, 2)

# 3.Resize 将PIL或者tensor缩放为指定大小然后输出PIL或者tensor
w, h = image.size   # PIL.Image的size先表示的宽再表示的高

trans_resize = transforms.Resize(min(w, h) // 2)    # 缩放为原来的1/2
img_resize = trans_resize(image)  # 对PIL进行缩放
writer.add_image('Resize', trans_totensor(img_resize))  # 因为在tensorboard中显示，所以需要转换为tensor或numpy类型

trans_resize = transforms.Resize(min(w, h) // 4)    # 缩放为原来的1/4
img_resize_tensor = trans_resize(img_tensor)
writer.add_image('Resize', img_resize_tensor, 1)

# 4.compose 组合这些操作
trans_compose = transforms.Compose(
    [transforms.Resize(min(w, h) // 2), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
img_campose = trans_compose(image)  # image是PIL.Image格式
writer.add_image('Compose', img_campose)

# 5.Randomcrop 随机裁剪
trans_randomcrop = transforms.RandomCrop(min(w, h) // 4)    # 从原图中任意位置裁剪1/4
# img_ranomcrop = trans_randomcrop(img_tensor)
for i in range(10):
    img_ranomcrop = trans_randomcrop(img_tensor)
    writer.add_image('RandomCrop', img_ranomcrop, i)

writer.close()
