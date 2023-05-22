#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/15:9:54
@File:           test_func.py
==================================================
"""

# 导入常用包
import os
from datetime import datetime
import logging
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from models.resnet import resnet101


# 超参数定义
# 批次的大小
batch_size = 64  # 可选32、64、128

test_mean, test_std = (0.4942, 0.4851, 0.4504), (0.2467, 0.24295, 0.2616)
test_data_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(test_mean, test_std)
                      ])

test_cifar_dataset = datasets.CIFAR10('cifar10', train=False, download=False, transform=test_data_transform)
test_loader = torch.utils.data.DataLoader(test_cifar_dataset,
                                          batch_size=batch_size, num_workers=4,
                                          shuffle=False)

exp_name = 'exp0'
weight_dir = './weights'
weight_dir = weight_dir + f'/resnet101_{exp_name}.pth'

log_dir = f'./runs/test/{exp_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=log_dir + '.txt',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO)
# writer
writer = SummaryWriter(log_dir)
# Set fixed random number seed
torch.manual_seed(42)
# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def test():
    # 加载模型权重
    model = resnet101()
    model.eval()
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir))

    model.eval()
    test_total_correct = 0
    test_total_loss = 0
    test_total_num = 0
    start = time.time()
    for iter, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_total_loss += loss.item()
        test_total_correct += (outputs.argmax(1) == labels).sum().item()
        test_total_num += labels.shape[0]
        # Write accuracy
        writer.add_scalar('test acc', test_total_correct / test_total_num * 100, iter)
        writer.add_scalar('test loss', test_total_loss / test_total_num * 100, iter)

    logging.info("test_loss:{:.4f}, test_acc:{:.4f}%".format(test_total_loss / test_total_num * 100,
                                                             test_total_correct / test_total_num * 100))
    logging.info("test_time:{:.4f}s".format(time.time() - start))


if __name__ == "__main__":
    logging.info("===   !!!START TESTING!!!   ===")
    logging.info("test data num:{}".format(len(test_cifar_dataset)))
    test()
    logging.info("===   !!! END TESTING !!!   ===")
    logging.info("\n\n\n\n")
