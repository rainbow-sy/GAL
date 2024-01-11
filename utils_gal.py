# import math
import os
# import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
# from PIL import Image, ExifTags
from torch.utils.data import Dataset
# from tqdm import tqdm
from torchvision import transforms
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self,
                 path,   # 指向data/my_train_data.txt路径或data/my_val_data.txt路径
                 # 这里设置的是预处理后输出的图片尺寸
                 # 当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
                 # 当为验证集时，设置的是最终使用的网络大小
                 data_path,
                 batch_size=16,
                 dataattr="train"  #验证集写成val
                 ):

        try:
            path = str(Path(path))
            if os.path.isfile(path):  # file
                # 读取对应my_train/val_data.txt文件，读取每一行的图片路劲信息
                with open(path, "r") as f:
                    self.img_files = f.read().splitlines()
            else:
                raise Exception("%s does not exist" % path)
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))

        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path)
        self.class_dict = dict((k, v) for v, k in enumerate(os.listdir(data_path)))
        data_transform = {
        "train": transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize([256, 256]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize([256, 256]),
                                   transforms.ToTensor()])}
        self.transform = data_transform[dataattr]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img, label = load_image(self, index)
        # letterbox
        img = self.transform(img)
        return img, label

def load_image(self, index):
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    #返回一个标签
    class_string = path.split(os.sep)[-2]
    label = self.class_dict[class_string]
    return img, label # img, hw_original, hw_resized
  









