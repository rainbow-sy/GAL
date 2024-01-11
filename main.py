import os
import numpy as np
import sys
import tqdm
# import utils.common as utils
from utils.common import train, test, create_lr_scheduler
# from utils.options import args
from importlib import import_module
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from resnet import resnet_56, resnet_56_sparse
from fista import FISTA
from discriminator import Discriminator
import pdb
from utils_gal import LoadImagesAndLabels

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

#     full_data = datasets.ImageFolder(root=args.data_path, transform=
#     transforms.Compose([transforms.Resize(256),
#                         transforms.ToTensor(),
#                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))  # your dataset

#     train_size = int(len(full_data) * 0.8)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
#     test_size = len(full_data) - train_size
#     print("using {} images for training, {} images for validation.".format(train_size, test_size))
#     train_dataset, test_dataset = random_split(full_data, [train_size, test_size],
#                                                generator=torch.Generator().manual_seed(0))
    train_path = args.train_path_txt
    test_path = args.val_path_txt
    train_dataset = LoadImagesAndLabels(train_path, args.data_path, batch_size,
                                        dataattr="train" # rectangular trainin
                                       )

    # 验证集的图像尺寸指定为img_size(512)
    test_dataset = LoadImagesAndLabels(test_path, args.data_path, batch_size,
                                      dataattr="val" # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)
                                      )

    #使用train set的集合去训练带有mask的模型
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=nw, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=nw, shuffle=False)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Create model
    print('=> Building model...')
    model_t = resnet_56(num_classes=args.num_class).to(device)#model_t教师网络比如resnet56，训练好的未修建的网络

    # Load teacher model
    ckpt_t = torch.load(args.teacher_dir)
    # pdb.set_trace()

    if args.arch == 'densenet':
        state_dict_t = {}
        for k, v in ckpt_t['state_dict'].items():
            new_key = '.'.join(k.split('.')[1:])
            if new_key == 'linear.weight':
                new_key = 'fc.weight'
            elif new_key == 'linear.bias':
                new_key = 'fc.bias'
            state_dict_t[new_key] = v
    else:
        state_dict_t = ckpt_t['state_dict']


    model_t.load_state_dict(state_dict_t)
    model_t = model_t.to(device)

    for para in list(model_t.parameters())[:-2]:   #冻结除全连接层的权重
        para.requires_grad = False

    model_s = resnet_56_sparse(num_classes=args.num_class).to(device)#model_s学生网络，resnet56_sparse。带有mask的需要修剪的pruned网络。

    model_dict_s = model_s.state_dict()
    model_dict_s.update(state_dict_t)
    model_s.load_state_dict(model_dict_s)

    model_d = Discriminator().to(device) #model_d判别器，这里是一个全连接层组成的网络

    models = [model_t, model_s, model_d]#三个模型的集合，model_t不用训练。model_s, model_d需要训练

    optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]#即generator的不包含mask的参数,对应原文中的WG
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]#即generator的mask的参数，对应原文中的m

    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_m = FISTA(param_m, lr=args.lr, gamma=args.sparse_lambda)

    # scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
    # scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
    # scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_d = create_lr_scheduler(optimizer_d, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)
    scheduler_s = create_lr_scheduler(optimizer_s, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)
    scheduler_m = create_lr_scheduler(optimizer_m, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_s.load_state_dict(ckpt['state_dict_s'])
        model_d.load_state_dict(ckpt['state_dict_d'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        optimizer_m.load_state_dict(ckpt['optimizer_m'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        scheduler_s.load_state_dict(ckpt['scheduler_s'])
        scheduler_m.load_state_dict(ckpt['scheduler_m'])
        print('=> Continue from epoch {}...'.format(start_epoch))

    optimizers = [optimizer_d, optimizer_s, optimizer_m]
    schedulers = [scheduler_d, scheduler_s, scheduler_m]
    #模型结果记录
    best_prec1 = 0.0  #测试集最好的准确率
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    if os.path.exists('./模型结果.txt'):
        os.remove('./模型结果.txt')
    filename = './模型结果.txt'

    for epoch in range(args.start_epoch, args.epochs):
        # for s in schedulers:
        #     s.step(epoch)

        train_acc = train(args, train_loader, models, optimizers, schedulers, epoch, device)
        test_prec1, test_loss = test(args, test_loader, model_s, epoch, device)
        all_train_acc.append(train_acc)
        all_val_loss.append(test_loss)
        all_val_acc.append(test_prec1)

        # is_best = best_prec1 < test_prec1
        # best_prec1 = max(test_prec1, best_prec1)
        if best_prec1 < test_prec1:
            best_prec1 = test_prec1
            mask=[]
            for param_tensor in model_s.state_dict():
                if 'mask' in param_tensor:
                    mask.append(model_s.state_dict()[param_tensor].item())
            state = {
                'state_dict_s': model_s.state_dict(),
                'state_dict_d': model_d.state_dict(),
                'best_prec1': best_prec1,
                'optimizer_d': optimizer_d.state_dict(),
                'optimizer_s': optimizer_s.state_dict(),
                'optimizer_m': optimizer_m.state_dict(),
                'scheduler_d': scheduler_d.state_dict(),
                'scheduler_s': scheduler_s.state_dict(),
                'scheduler_m': scheduler_m.state_dict(),
                'mask': mask,
                'epoch': epoch + 1
            }
            #'cfg': checkpoint['cfg']
            torch.save(state, os.path.join(args.save, 'pruned_checkpoint.pth.tar'))
            # best_prec1 = test_prec1
    with open(filename, 'a+') as f:
        f.write('训练集准确率：' + str(all_train_acc) + '\n')
        f.write('验证集损失：' + str(all_val_loss) + '\n')
        f.write('验证集准确率：' + str(all_val_acc) + '\n')
        f.write('验证集最高的准确率：' + str(best_prec1) + '\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=32)
    # parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for MomentumOptimizer.')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='The weight decay of loss.')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--lambda', dest='sparse_lambda', type=float, default=0.6, help='The sparse lambda for l1 loss')
    parser.add_argument('--lr_decay_step', type=int, default=30)
    parser.add_argument('--miu', type=float, default=1, help='The miu of data loss.')
    parser.add_argument('--mask_step', type=int, default=200, help='The frequency of mask to update')
    parser.add_argument('--teacher_dir', default='./logs/epoch60_checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to the weights of pre-trained model without compression')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--data_path', default='./dataset/Mixeddata_noedge', type=str, metavar='PATH',
                        help='data directory')
    parser.add_argument('--train_path_txt', default='./mydataset/train.txt', type=str, metavar='PATH',
                        help='path to the train data recording')
    parser.add_argument('--val_path_txt', default='./mydataset/val.txt', type=str, metavar='PATH',
                        help='path to the val data recording')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-path', default='./dataset/Mixeddata', type=str, metavar='PATH',
                        help='path to the all data restored')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    # parser.add_argument('--arch', default='resnet', type=str,
    #                     help='architecture to use')
    opt = parser.parse_args()

    main(opt)


