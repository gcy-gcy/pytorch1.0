import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

image_path = 'data/train/ants_image/0013035.jpg'
image = Image.open(image_path)

# 1.transforms该如何使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image)

print(np.array(image).shape)
print(tensor_img.shape)

# 写入tensorboard
writer = SummaryWriter('gyx1')
writer.add_image('gyx', tensor_img, 1)
writer.close()
