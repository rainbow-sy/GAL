import os
import numpy as np
import sys
import tqdm
# import utils.common as utils
from utils.common import create_lr_scheduler, AverageMeter, accuracy
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
import time
from tqdm import tqdm

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    full_data = datasets.ImageFolder(root=args.data_path, transform=
    transforms.Compose([transforms.Resize(256),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))  # your dataset

    train_size = int(len(full_data) * 0.8)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
    test_size = len(full_data) - train_size
    print("using {} images for training, {} images for validation.".format(train_size, test_size))
    train_dataset, test_dataset = random_split(full_data, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(0))
    # 使用train set的集合去训练带有mask的模型
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=nw, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=nw, shuffle=False)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # Create model
    print('=> Building model...')

    # Fine tune from a checkpoint
    refine = args.refine
    assert refine is not None, 'refine is required'
    # checkpoint = torch.load(refine, map_location=device)
    checkpoint = torch.load(refine)
    if args.pruned:
        state_dict = checkpoint['state_dict_s']
        if args.arch == 'vgg':
            cfg = checkpoint['cfg']
            model = vgg_16_bn_sparse(cfg = cfg).to(device)
        # pruned = sum([1 for m in mask if mask == 0])
        # print(f"Pruned / Total: {pruned} / {len(mask)}")
        elif args.arch =='resnet':
            mask = checkpoint['mask']
            # model = resnet_56_sparse(has_mask = mask,num_classes=args.num_class).to(device)
            model = resnet_56(has_mask = mask,num_classes=args.num_class).to(device)
        elif args.arch == 'densenet':
            filters = checkpoint['filters']
            indexes = checkpoint['indexes']
            model = densenet_40_sparse(filters = filters, indexes = indexes).to(device)
        elif args.arch =='googlenet':
            mask = checkpoint['mask']
            model = googlenet_sparse(has_mask = mask).to(device)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, checkpoint['state_dict_s'])
    
    criterion = nn.CrossEntropyLoss()
    test_acc, _ = test(test_loader, model, criterion, 1, device)
    print('测试集准确率：',test_acc)
    if args.test_only:
        return 

    # if args.keep_grad:#不修改mask的参数值，不计算其梯度
    #     for name, weight in model.named_parameters():
    #         if 'mask' in name:
    #             weight.requires_grad = False

    # train_param = [param for name, param in model.named_parameters() if 'mask' not in name]  #不包含mask的参数信息
    train_param = [param for name, param in model.named_parameters()] 
    optimizer = optim.SGD(train_param, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=args.warmup_epoch)
    resume = args.resume
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))

        # 模型结果记录
    best_prec1 = 0.0  # 测试集最好的准确率
    all_train_acc = []
    all_train_loss = []
    all_val_loss = []
    all_val_acc = []
    all_time = []
    if os.path.exists('./模型结果.txt'):
        os.remove('./模型结果.txt')
    filename = './模型结果.txt'
    for epoch in range(args.epochs):
        # scheduler.step(epoch)

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, device)
        t1 = time.time()
        test_prec1, test_loss = test(test_loader, model, criterion, epoch, device)
        t2 = time.time()
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_val_loss.append(test_loss)
        all_val_acc.append(test_prec1)
        all_time.append(t2 - t1)
        # is_best = best_prec1 < test_prec1
        # best_prec1 = max(test_prec1, best_prec1)
        if best_prec1 < test_prec1:
            best_prec1 = test_prec1
            state = {
                'state_dict_s': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, os.path.join(args.save, 'finetune_checkpoint.pth.tar'))

    with open(filename, 'a+') as f:
        f.write('训练集准确率：' + str(all_train_acc) + '\n')
        f.write('训练集损失：' + str(all_train_loss) + '\n')
        f.write('验证集损失：' + str(all_val_loss) + '\n')
        f.write('验证集准确率：' + str(all_val_acc) + '\n')
        f.write('验证集最高的准确率：' + str(best_prec1) + '\n')
        f.write('推理时间：' + str(all_time) + '\n')

def train(loader_train, model, criterion, optimizer, scheduler, epoch, device):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    loader_train = tqdm(loader_train, file=sys.stdout)

    for i, (inputs, targets) in enumerate(loader_train, 1):

        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        prec1= accuracy(logits, targets, topk=(1, ))
        losses.update(loss.item())

        top1.update(prec1[0].item(), inputs.size(0))
        loader_train.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            losses.avg,
            top1.avg,
            optimizer.param_groups[0]["lr"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return top1.avg, losses.avg

def test(loader_test, model, criterion, epoch, device):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    loader_test = tqdm(loader_test, file=sys.stdout)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            prec1 = accuracy(logits, targets, topk=(1,))
            losses.update(loss.item())
            top1.update(prec1[0].item(), inputs.size(0))
            loader_test.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                epoch,
                losses.avg,
                top1.avg
            )
    return top1.avg, losses.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--arch', default='resnet', type=str,
                        help='architecture to use')
    parser.add_argument('--pruned', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for MomentumOptimizer.')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='The weight decay of loss.')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--keep_grad', type=bool, default=True)
    # parser.add_argument('--lambda', dest='sparse_lambda', type=float, default=0.6, help='The sparse lambda for l1 loss')
    # parser.add_argument('--lr_decay_step', type=int, default=30)
    # parser.add_argument('--miu', type=float, default=1, help='The miu of data loss.')
    # parser.add_argument('--mask_step', type=int, default=200, help='The frequency of mask to update')
    # parser.add_argument('--teacher_dir', default='./logs/original_checkpoint.pth.tar', type=str, metavar='PATH',
    #                     help='path to the weights of pre-trained model without compression')
    parser.add_argument('--refine', default='./logs/basepruned_checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-path', default='./dataset/Mixeddata', type=str, metavar='PATH',
                        help='path to the all data restored')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    opt = parser.parse_args()

    main(opt)
