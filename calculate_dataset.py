"""
该脚本有2个功能：
1.统计训练集和验证集的数据并生成相应.txt文件
2.创建.shapes文件，记录图片的长和宽
"""
import os
import random
import cv2
import numpy as np

def main():
    # 统计训练集和验证集的数据并生成相应txt文件
    data_path=r'E:\test_datatest\train'  #包含4个类别的文件夹及对应的图片
    img_path=[]
    for file in os.listdir(data_path):
        imgdir=os.path.join(data_path,file)
        full_path=[os.path.join(imgdir,i) for i in os.listdir(imgdir)]
        img_path.extend(full_path)
    random.seed(1234)
    random.shuffle(img_path)
    train_size=int(len(img_path)*0.8)
    train_data=img_path[:train_size]
    val_data=img_path[train_size:]

    if not os.path.exists('./mydataset'):#确认数据存储的文件夹是否存在
        os.makedirs('./mydataset')

    train_txt_path = "./mydataset/train.txt"  #图片路径存储
    val_txt_path = "./mydataset/val.txt"
    with open(train_txt_path, 'w') as f:
        for data in train_data:
            f.write(data + '\n')

    with open(val_txt_path, 'w') as f:
        for data in val_data:
            f.write(data + '\n')

    train_size_path = "./mydataset/train.shapes"#图片的H,W存储
    val_size_path = "./mydataset/val.shapes"
    train_shapes=[]
    val_shapes=[]
    for imgpath in train_data:
        img=cv2.imread(imgpath)
        h, w = img.shape[:2]
        train_shapes.append((h,w))

    for imgpath in val_data:
        img=cv2.imread(imgpath)
        h, w = img.shape[:2]
        val_shapes.append((h,w))

    train_shapes=np.array(train_shapes).reshape(-1,2)
    val_shapes = np.array(val_shapes).reshape(-1,2)
    np.savetxt(train_size_path, train_shapes, fmt="%g")  # overwrite existing (if any)
    np.savetxt(val_size_path, val_shapes, fmt="%g")

if __name__ == '__main__':
    main()
