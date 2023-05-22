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

from utils import WarmUpLR
from models.resnet import resnet101, resnet18

import logging
import time


def train(lr=1e-1):
    # 模型
    model = resnet18()
    model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = torch.nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    # 交叉熵
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    model = model.to(device)
    # 训练时间
    start = time.time()
    # 保存模型文件的尾标
    save_num = 0

    val_last_acc = 0.0
    last_val_loss = np.Inf
    epoch_count = 0
    for epoch in range(MAX_EPOCH):
        train_total_loss = 0
        train_total_num = 0
        train_total_correct = 0
        if epoch < WARMUP_EPOCH:
            lr = 0.01
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif epoch == WARMUP_EPOCH:
            lr = 0.1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif epoch_count / 10 == 1 and epoch > WARMUP_EPOCH:
            epoch_count = 0
            lr = lr * 0.5
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        logging.info('Epoch {}/{}'.format(epoch, MAX_EPOCH))
        logging.info('-' * 10)
        logging.info("\n")

        with torch.set_grad_enabled(True):
            model.train()
            for iter, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # # Write the network graph at epoch 0, batch 0
                # if epoch == 0 and iter == 0:
                #     writer.add_graph(model, input_to_model=(images, labels)[0])
                # writer.add_images('mini_batch_images', images, iter + (total_iter * epoch))

                outputs = model(images)
                loss = criterion(outputs, labels)
                train_total_correct += (outputs.argmax(1) == labels).sum().item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_total_num += labels.shape[0]
                train_total_loss += loss.item()

            train_loss = train_total_loss / train_total_num
            train_acc = train_total_correct / train_total_num

            time_elapsed = time.time() - start
            logging.info("\n")
            logging.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logging.info('train Loss: {:.4f}[{}], train Acc: {:.4f}'.format(train_loss, epoch_count, train_acc))

        with torch.set_grad_enabled(False):
            model.eval()
            val_total_loss = 0
            val_total_correct = 0
            val_total_num = 0
            # for iter, (images, labels) in enumerate(test_loader):
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

            if val_loss < last_val_loss:
                last_val_loss = val_loss
                epoch_count = 0
                # 只保存最近2次的训练结果
                save_num = 0 if save_num > 1 else save_num
                save_name_t = '{}_{}.pth'.format(best_weight_dir, save_num)
                torch.save(model.state_dict(), save_name_t)
                save_num += 1
            else:
                epoch_count += 1

            time_elapsed = time.time() - start
            logging.info("\n")
            logging.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logging.info('valid Loss: {:.4f}[{}], valid Acc: {:.4f}'.format(val_loss, epoch_count, val_acc))

        logging.info("\n")
        logging.info('当前学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        logging.info("\n")

        # # Write loss for epoch
        writer.add_scalars('Loss', {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning rate': optimizer.param_groups[0]['lr']
        }, epoch)
        writer.add_scalars('acc', {
            'train_acc': train_acc * 100,
            'val_acc': val_acc * 100,
        }, epoch)

    # 训练结束
    time_elapsed = time.time() - start
    logging.info("\n")
    logging.info('任务完成！')
    logging.info('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('最高验证集准确率: {:4f}'.format(val_last_acc))
    save_num = save_num - 1
    save_num = save_num if save_num < 0 else 1
    save_name_t = '{}_{}.pth'.format(best_weight_dir, save_num)
    logging.info('最优模型保存在：{}'.format(save_name_t))

    writer.close()
    # 保存 模型权重
    torch.save(model.state_dict(), save_name_t)


if __name__ == '__main__':
    # 超参数定义
    # 批次的大小
    batch_size = 128  # 可选32、64、128
    # 优化器的学习率
    LR = 1e-1
    # 运行epoch
    MAX_EPOCH = 200
    WARMUP_EPOCH = 2

    # 数据读取
    # cifar10数据集为例给出构建Dataset类的方式
    # “data_transform”可以对图像进行一定的变换，如翻转、裁剪、归一化等操作，可自己定义
    train_mean, train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)
    test_mean, test_std = (0.49421, 0.48513, 0.45041), (0.24665, 0.24289, 0.26159)
    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        transforms.Normalize(train_mean, train_std)
    ])
    test_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)
    ])

    train_cifar_dataset = datasets.CIFAR10('cifar10', train=True, download=False, transform=train_data_transform)
    # test_cifar_dataset = datasets.CIFAR10('cifar10', train=False, download=False, transform=test_data_transform)
    train_cifar_dataset, val_cifar_dataset = torch.utils.data.random_split(train_cifar_dataset, [45000, 5000])

    # 构建好Dataset后，就可以使用DataLoader来按批次读入数据了
    train_loader = torch.utils.data.DataLoader(train_cifar_dataset,
                                               batch_size=batch_size, num_workers=4,
                                               shuffle=True, drop_last=True)
    # test_loader = torch.utils.data.DataLoader(test_cifar_dataset,
    #                                           batch_size=batch_size, num_workers=4,
    #                                           shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_cifar_dataset,
                                             batch_size=batch_size, num_workers=4,
                                             shuffle=False)

    feature = "my_resnet101"
    exp_name = f'{feature}_exp0'
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

    best_weight_dir = weight_dir + f'/{exp_name}_MAX_EPOCH{MAX_EPOCH}'
    save_dir = weight_dir + f'/{exp_name}_MAX_EPOCH{MAX_EPOCH}.pth'

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

    logging.info("===   !!!START TRAINING!!!   ===")
    logging.info('train_data_num: {}, validation_data_num: {}'.format(len(train_cifar_dataset), len(val_cifar_dataset)))
    train(lr=LR)
    logging.info("===   !!! END TRAINING !!!   ===")
    logging.info("\n\n\n\n")
