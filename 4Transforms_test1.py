import numpy as np
from torchvision import transforms
from PIL import Image

image_path = 'data/train/ants_image/0013035.jpg'
image = Image.open(image_path)

# 1.transforms该如何使用(python)
tensor_trans = transforms.ToTensor()	# ToTensor()中不带参数
tensor_img = tensor_trans(image)		# 不能直接写成transforms.ToTensor(image)

print(np.array(image).shape)	# (512, 768, 3)
print(tensor_img.shape)			# torch.Size([3, 512, 768])，通道数变到第0维了
