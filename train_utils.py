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
                 img_size=416,
                 batch_size=16,
                 augment=False,  # 训练集设置为True(augment_hsv)，验证集设置为False
                 rect=True,
                 pad=0.0):

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

        # batch index
        # 将数据划分到一个个batch中
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)#[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2....],batch_size=8的情况
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images 图像总数目
        self.batch = bi  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸
        self.rect = rect
        self.augment = augment
        self.class_dict = dict((k, v) for v, k in enumerate(os.listdir(data_path)))
        self.transform = transforms.ToTensor()
        # Read image shapes (wh)
        # 查看data文件下是否缓存有对应数据集的.shapes文件，里面存储了每张图像的 height，width
        sp = path.replace(".txt", ".shapes")  # shapefile path
        with open(sp, "r") as f:  # read existing shapefile
            s = [x.split() for x in f.read().splitlines()]
            # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
            # 如果不相等则认为是不同的数据集，故重新生成shape文件
            assert len(s) == n, "shapefile out of aync"
        # 记录每张图像的原始尺寸
        self.shapes = np.array(s, dtype=np.float64)
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # hw
            # 计算每个图片的高/宽比
            ar = s[:, 0] / s[:, 1]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            irect = ar.argsort()
            # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
            self.img_files = [self.img_files[i] for i in irect]
            # self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # hw
            ar = ar[irect]
            #图像的类别
            
            # set training image shapes
            # 计算每个batch采用的统一尺度
            shapes = [[1, 1]] * nb  # nb: number of batches
            for i in range(nb):
                ari = ar[bi == i]  # bi: batch index
                # 获取第i个batch中，最小和最大高宽比
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        # cache labels
        self.imgs = [None] * n  # n为图像总数

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img, label = load_image(self, index)
        # letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
        # shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)
        img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        # img_size 设置的是预处理后输出的图片尺寸
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        #返回一个标签
        class_string = path.split(os.sep)[-2]
        label = self.class_dict[class_string]
        return img, label # img, hw_original, hw_resized
    else:
        return self.imgs[index] # img, hw_original, hw_resized00

def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(0, 0, 0),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path="./new_folder"):
    # Create floder
    if os.path.exists(path):
        shutil.rmtree(path)  # dalete output folder
    os.makedirs(path)  # make new output folder




