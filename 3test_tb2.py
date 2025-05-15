from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
#利用opencv读取 获取numpy型图片数据

writer = SummaryWriter('logs_3')  # 实例化一个SummaryWriter为writer，并指定event的保存路径为logs

image_path1 = 'data/train/ants_image/0013035.jpg'
image_path2 = 'data/train/bees_image/16838648_415acd9e3f.jpg'

img = Image.open(image_path2)	# 打开image_path1
img_array = np.array(img) #转为numpy
print(type(img))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
print(type(img_array))  # <class 'numpy.ndarray'>
print(img_array.shape)

# 这里的add_image中的tag为'test_image'没有变化，所以在tensorboard中可通过拖动滑块来展示这两张图像
# writer.add_image('test_image', img_array, 1, dataformats='HWC')
writer.add_image('test_image', img_array, 2, dataformats='HWC')

for i in range(10):	# 这个add_scalar暂时没有管它，虽然tag没有变，但是因为每次写入的数据都是y=3x所以曲线没有显示混乱
    writer.add_scalar('y=3x', 3 * i, i)

writer.close()  # 最后还需要将这个writer关闭
