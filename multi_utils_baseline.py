import os
import sys
import json
import pickle
import random
import math
# from model import convnext_tiny as create_model
# from Modified_Base_Model import convnext_tiny as create_model
# from final_model import convnext_tiny as create_model
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import torch.nn.functional as F
from torch.cuda import amp
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

####新增的BN稀疏＃＃＃＃＃＃＃＃＃＃＃
# def updateBN(model):
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.weight.grad.data.add_(0.001*torch.sign(m.weight.data))  # L1

# def train_one_epoch(model, optimizer, data_loader, device, epoch):
def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    lr_scheduler1, accumulate, grid_min, grid_max,
                    gs, img_size, warmup, multi_scale=False, scaler=None):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        accumulate = 1
    sample_num = 0
    nb = len(data_loader)  # 一共有多少个batch
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        ni = step + nb * epoch  # 当前属于第几个batch
        #图片多尺寸
        ns=images.shape[2:]
        if multi_scale:
            # 每训练64张图片，就随机修改一次输入图片大小，
            # 由于label已转为相对坐标，故缩放图片不影响label的值
            if ni % accumulate == 0:  # adjust img_size (67% - 150%) every 1 batch
                # 在给定最大最小输入尺寸范围内随机选取一个size(size为32的整数倍)
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(images.shape[2:])  # scale factor

            # 如果图片最大边长不等于img_size, 则缩放图片，并将长和宽调整到32的整数倍
            if sf != 1:
                # gs: (pixels) grid size
                ns = [math.ceil(x * sf / gs) * gs for x in images.shape[2:]]  # new shape (stretched to 32-multiple)
                images = F.interpolate(images, size=ns, mode='bilinear', align_corners=False)
        with amp.autocast(enabled=scaler is not None):
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # loss = loss_function(pred, labels.to(device))
        # backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accu_loss += loss.detach()
        # updateBN(model)
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler1.step()
    print('训练图像的尺寸：', ns)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# def test(data_loader, device):
#     # create model
#     model = create_model(num_classes=4).to(device)
#     # load model weights
#     model_weight_path = "./weights/best_model.pth"
#     model.load_state_dict(torch.load(model_weight_path, map_location=device))
#     model.eval()
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     result_list=[]#存储预测结果
#     labels_list=[]#存储真实结果
#     with torch.no_grad():
#         for step, data in enumerate(data_loader):
#             images, labels = data
#             sample_num += images.shape[0]
#             pred = model(images.to(device))
#             pred_classes = torch.max(pred, dim=1)[1]
#             result_list.append(pred_classes)
#             labels_list.append(labels)
#             accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#         result_final = torch.cat(result_list,dim=0).cpu()
#         labels_final = torch.cat(labels_list,dim=0).cpu()
#     micro_f1 = f1_score(labels_final, result_final, average='micro')
#     macro_f1 = f1_score(labels_final, result_final, average='macro')
#     #添加的处理
#     Macro_precision= precision_score(labels_final, result_final, average='macro')
#     Macro_recall= recall_score(labels_final, result_final, average='macro')
#     #绘制混淆矩阵
#     # plot size setting
#     cm = confusion_matrix(labels_final, result_final)
#     conf_matrix = pd.DataFrame(cm, index=['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'], columns=['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia'])
#     # plot size setting
#     fig, ax = plt.subplots(figsize = (10,10))
#     ax.get_yaxis().get_major_formatter().set_scientific(True)
#     sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues",fmt='.20g')
#     plt.ylabel('True label', fontsize=14)
#     plt.xlabel('Predicted label', fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig('confusion.png', bbox_inches='tight')
#     plt.show()
#     return accu_num.item() / sample_num, micro_f1, macro_f1, Macro_precision, Macro_recall

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def plot_fig(train_loss,valid_loss,metrics='Loss'):
    #Plot the Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.legend(['Training '+metrics, 'Validation '+metrics])
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel(metrics,fontsize=16)
    plt.title(metrics+' Curves',fontsize=16)
    plt.savefig(r'./model_'+metrics+'.png')
    plt.show()