# 导入常用包
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torch.nn as nn


from utils import WarmUpLR

import logging
import time

# 超参数定义
# 批次的大小
batch_size = 32  # 可选32、64、128
# 优化器的学习率
lr = 1e-1
# 运行epoch
MAX_EPOCH = 200
# WARMUP_EPOCH = int(0.05 * MAX_EPOCH) if int(0.05 * MAX_EPOCH) > 0 else 1


# 数据读取
# cifar10数据集为例给出构建Dataset类的方式
# “data_transform”可以对图像进行一定的变换，如翻转、裁剪、归一化等操作，可自己定义
train_mean, train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)

train_data_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
                                            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                                            transforms.Normalize(train_mean, train_std)
                                          ])
train_cifar_dataset = datasets.CIFAR10('cifar10', train=True, download=False, transform=train_data_transform)
train_cifar_dataset, val_cifar_dataset = torch.utils.data.random_split(train_cifar_dataset, [45000, 5000])

# 构建好Dataset后，就可以使用DataLoader来按批次读入数据
train_loader = torch.utils.data.DataLoader(train_cifar_dataset,
                                           batch_size=batch_size, num_workers=4,
                                           shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_cifar_dataset,
                                         batch_size=batch_size, num_workers=4,
                                         shuffle=False)

feature = "torchresnet18"
exp_name = f'{feature}_exp1'
weight_dir = './weights'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

log_dir = f'./runs/train/{exp_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=log_dir + '.txt',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO)

best_weight_dir = weight_dir + f'/resnet101_{exp_name}_MAX_EPOCH{MAX_EPOCH}'
save_dir = weight_dir + f'/resnet101_{exp_name}_MAX_EPOCH{MAX_EPOCH}.pth'

# 训练&验证
# writer = SummaryWriter(log_dir)
hengyuanyun_log_dir = "/tf_logs/" + exp_name
if not os.path.exists(hengyuanyun_log_dir):
    os.makedirs(hengyuanyun_log_dir)
writer = SummaryWriter(hengyuanyun_log_dir)

# Set fixed random number seed
torch.manual_seed(42)
# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(lr=1e-1):
    # 模型
    # 定义模型  使用官方模型
    model = torchvision.models.resnet18(weights=None)

    # 修改模型
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    num_ftrs = model.fc.in_features  # 获取（fc）层的输入的特征数
    model.fc = nn.Linear(num_ftrs, 10)

    # 交叉熵
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * MAX_EPOCH), int(0.8 * MAX_EPOCH)], gamma=0.1)
    total_iter = len(train_loader)
    # warmup_scheduler = WarmUpLR(optimizer, total_iter * WARMUP_EPOCH)
    model = model.to(device)

    # 训练时间
    start = time.time()
    val_last_acc = 0.0
    val_last_loss = np.Inf
    last_train_loss = np.Inf
    epoch_count = 1
    for epoch in range(MAX_EPOCH):
        model.train()
        train_total_loss = 0
        train_total_num = 0
        train_total_correct = 0

        if epoch_count % 10 == 0:
            epoch_count = 0
            lr = lr * 0.5
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        for iter, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            train_total_correct += (outputs.argmax(1) == labels).sum().item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if epoch >= WARMUP_EPOCH:
            #     scheduler.step()
            # else:
            #     warmup_scheduler.step()

            train_total_num += labels.shape[0]
            train_total_loss += loss.item()

        train_loss = train_total_loss / train_total_num
        train_acc = train_total_correct / train_total_num

        if train_loss < last_train_loss:
            last_train_loss = train_loss
            epoch_count = 0
        else:
            epoch_count += 1

        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_num = 0
        for iter, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_total_correct += (outputs.argmax(1) == labels).sum().item()
            val_total_loss += loss.item()
            val_total_num += labels.shape[0]

        val_loss = val_total_loss / val_total_num
        val_acc = val_total_correct / val_total_num

        if val_acc > val_last_acc and val_loss < val_last_loss:
            save_name_t = '{}_epoch{}_trainAcc{:.4f}_valAcc{:.4f}.pth'.format(best_weight_dir, epoch, train_acc, val_acc)
            torch.save(model.state_dict(), save_name_t)
            val_last_acc = val_acc
            val_last_loss = val_loss

        # # Write loss for epoch
        writer.add_scalars('Loss', {
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, epoch)
        writer.add_scalars('acc', {
            'train_acc': train_acc * 100,
            'val_acc': val_acc * 100,
        }, epoch)

    logging.info("Max Epoch:{}, train_val_time:{:.4f}s".format(MAX_EPOCH, time.time() - start))
    writer.close()
    # 保存 模型权重
    torch.save(model.state_dict(), save_dir)


if __name__ == '__main__':

    logging.info("===   !!!START TRAINING!!!   ===")
    logging.info('train_data_num: {}, validation_data_num: {}'.format(len(train_cifar_dataset), len(val_cifar_dataset)))
    train(lr=lr)
    logging.info("===   !!! END TRAINING !!!   ===")
    logging.info("\n\n\n\n")
