from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import models
# from evo_preresnet import resnet
from resnet import resnet_56
from torch.utils.data import DataLoader, Dataset, random_split
import time
from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate,plot_fig

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    full_data = datasets.ImageFolder(root='./dataset/Mixeddata', transform=
    transforms.Compose([transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))  # your dataset

    train_size = int(len(full_data) * 0.8)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
    test_size = len(full_data) - train_size
    print("using {} images for training, {} images for validation.".format(train_size,test_size))
    train_dataset, test_dataset = random_split(full_data, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=nw, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=nw, shuffle=False)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.refine:
        checkpoint = torch.load(args.refine)
        # model = models.__dict__[args.arch](cfg=checkpoint['cfg'])
        model = resnet(cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
        # model = models.__dict__[args.arch]()
        model = resnet_56(num_classes=4)
    # #加载权重
    # checkpoint = torch.load('logs/epoch100_checkpoint.pth.tar')
    # # model = models.__dict__[args.arch]()
    # model.load_state_dict(checkpoint['state_dict'])
    print(model)
    
    model=model.to(device)
    
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=args.warmup_epoch)
    lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=10)  # 上一次的设置为factor=0.1, patience=3
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[30,50,75],gamma=0.1)
    if args.resume:
        if os.path.isfile(args.resume) :
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_acc = 0.
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    inference_time = []
    train_time = []
    lr_record=[]
    if os.path.exists('./模型结果.txt'):
        os.remove('./模型结果.txt')
    filename = './模型结果.txt'
    for epoch in range(args.start_epoch, args.epochs):
        # train
        t1 = time.time()
        lr_record.append(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler1=lr_scheduler
                                                )
                                                
        
        
        
        # lr_scheduler2.step()
        t2 = time.time()
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=test_loader,
                                     device=device,
                                     epoch=epoch)
        t3 = time.time()
        lr_scheduler2.step(val_acc)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)
        train_time.append(t2 - t1)
        inference_time.append(t3 - t2)

        if best_acc < val_acc:
            if args.refine:
                save_files = {
                    'state_dict': model.state_dict(),
                    'best_prec': val_acc,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': checkpoint['cfg']}
            else:
                save_files = {
                    'state_dict': model.state_dict(),
                    'best_prec': val_acc,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch}
            torch.save(save_files, os.path.join(args.save, 'checkpoint.pth.tar'))
            best_acc = val_acc

    with open(filename, 'a+') as f:
        f.write('训练集损失：' + str(all_train_loss) + '\n')
        f.write('训练集准确率：' + str(all_train_acc) + '\n')
        f.write('验证集损失：' + str(all_val_loss) + '\n')
        f.write('验证集准确率：' + str(all_val_acc) + '\n')
        f.write('训练时间：' + str(train_time) + '\n')
        f.write('推理时间：' + str(inference_time) + '\n')
        f.write('学习率：' + str(lr_record) + '\n')
    # additional subgradient descent on the sparsity-induced penalty term

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
    #                     help='train with channel sparsity regularization')
    # parser.add_argument('--s', type=float, default=0.0001,
    #                     help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='eresnet', type=str,
                        help='architecture to use')
    opt = parser.parse_args()

    main(opt)
