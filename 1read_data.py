from torch.utils.data import Dataset
from PIL import Image
import os

# 构造一个子文件夹数据集类MyData
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):    # root_dir是指整个数据集的根目录，label_dir是指具体某一个类的子目录
        # 在init初始化函数中，定义一些类中的全局变量，即跟在self.后的变量们
        # self相当于船，函数变量相当于人，只有把变量放在船上才能使别的函数使用
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_list = os.listdir(self.path)

    def __getitem__(self, index):   # 传入下标获取元素
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label[:-6]	# 返回的是一个元组
        # 这里进行了截取，因为我不想要label_dir最后面的'_image'这6个元素

    def __len__(self):
        return len(self.img_list)

# --------------实例化ants_data和bees_data------------- #
root_dir = 'data/train'
ants_dir = 'ants_image'
bees_dir = 'bees_image'
ants_data = MyData(root_dir, ants_dir)
bees_data = MyData(root_dir, bees_dir)
# ---------------------------------------------------- #

# -------------返回一个元组，分别赋值给img和label------- #
img, label = ants_data[0]
# ----------------------------------------------------- #

# ---因为是元组，所以可用[0]、[1]直接提取出img、label---- #
print(label == ants_data[0][1])		# true
# ----------------------------------------------------- #

# ----------将ants_data和bees_data相加起来使用---------- #
y = ants_data + bees_data
len_ants = len(ants_data)	# 124
len_bees = len(bees_data)	# 121
len_y = len(y)				# 245
print(len_y == len_ants+len_bees)	# True
print(y[123][1])			# ants
print(y[124][1])			# bees
