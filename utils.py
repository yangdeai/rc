#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/15:9:54
@File:           utils.py
==================================================
"""

import os
import sys

import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler

from rcProjection import RcProject


def get_network(network):
    """ return given network
    """

    if network == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif network == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif network == 'resnet10':
        from models.resnet import resnet10
        net = resnet10()
    elif network == 'miniresnet_4':
        from models.resnet import miniresnet_4
        net = miniresnet_4()
    elif network == 'mini_resnet10':
        from models.resnet import mini_resnet10
        net = mini_resnet10()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


class PreproRcData:
    """
        对RC处理之后的数据进行处理，作用为dataLoader.
    """
    def __init__(self, rcPro=None, batch_size=128, train=True, seed=1234):

        self.set_seed(seed)

        self.rc_data = rcPro.rc_project()
        self.labels = rcPro.labels
        self.in_num = rcPro.in_num
        self.out_num = rcPro.out_num

        self.batch_size = batch_size
        self.train = train

        self.rc_data = np.reshape(self.rc_data, (-1, 32, 32, 3 * self.out_num))  # 为进入transforms做准备

        self.prepro_data_path = './prepro_rc_data/'
        if not os.path.exists(self.prepro_data_path):
            os.makedirs(self.prepro_data_path)

        self.mean, self.std = self.get_mean_std()
        self.train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])

        self.test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.batch_idxes = BatchSampler(RandomSampler(self.rc_data, replacement=False),
                                        batch_size=self.batch_size,
                                        drop_last=True)

    def dataloader(self):
        """
            根据batch_idx抽取采样数据，生成器。
        :return: batch_size大小的数据
        """
        for batch_idx in self.batch_idxes:
            batch_trans_data = self.transforms(batch_idx).to(torch.float32)
            batch_labels = torch.tensor([self.labels[i] for i in batch_idx], dtype=torch.int64)

            yield batch_trans_data, batch_labels

    def transforms(self, batch_idx):
        batch_trans_data = []
        for idx in batch_idx:
            if self.train:
                x = self.train_trans(self.rc_data[idx])
            else:
                x = self.test_trans(self.rc_data[idx])
            batch_trans_data.append(x)

        batch_trans_data = torch.stack(batch_trans_data, dim=0)

        return batch_trans_data

    def get_mean_std(self):
        if self.train:
            mean_file = self.prepro_data_path + f'train_mean_in{self.in_num}_out{self.out_num}.npy'
            std_file = self.prepro_data_path + f'train_std_in{self.in_num}_out{self.out_num}.npy'
        else:
            mean_file = self.prepro_data_path + f'test_mean_in{self.in_num}_out{self.out_num}.npy'
            std_file = self.prepro_data_path + f'test_std_in{self.in_num}_out{self.out_num}.npy'

        if (not os.path.isfile(mean_file)) or (not os.path.isfile(std_file)):
            mean, std = self.compute_mean_std(mean_file, std_file)
        else:
            mean, std = np.load(mean_file), np.load(std_file)

        return mean, std

    def compute_mean_std(self, mean_file, std_file):
        channel = 3 * self.out_num
        means = [0 for _ in range(channel)]
        stds = [0 for _ in range(channel)]
        num_data = len(self.rc_data)
        for num_idx in range(num_data):
            for channel_idx in range(channel):
                means[channel_idx] += self.rc_data[num_idx][channel_idx, :, :].mean()
                stds[channel_idx] += self.rc_data[num_idx][channel_idx, :, :].std()

        mean = np.array(means) / num_data
        std = np.array(stds) / num_data

        # save data
        np.save(mean_file, mean)
        np.save(std_file, std)

        return mean, std

    def set_seed(self, seed):
        torch.manual_seed(seed)  # 固定随机种子（CPU）
        np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
        if torch.cuda.is_available():  # 固定随机种子（GPU)
            torch.cuda.manual_seed(seed)  # 为当前GPU设置
            torch.backends.cudnn.benchmark = True  # False  # GPU、网络结构固定，可设置为True
            torch.backends.cudnn.deterministic = True  # 固定网络结构


if __name__ == "__main__":

    random_seed = 1234
    BATCH_SIZE = 4

    input_port_num = 1  # 输入端口数
    output_port_num = 6  # 输出端口数
    train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)  # 训练数据集
    test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)  # 测试数据集
    train_rc_project = RcProject(train_dataset, in_num=input_port_num, out_num=output_port_num, seed=random_seed)
    test_rc_project = RcProject(test_dataset, in_num=input_port_num, out_num=output_port_num, seed=random_seed)

    train_rc_dataset = PreproRcData(rcPro=train_rc_project, batch_size=BATCH_SIZE, train=True, seed=random_seed)
    test_rc_dataset = PreproRcData(rcPro=test_rc_project, batch_size=BATCH_SIZE, train=False, seed=random_seed)

    for i in range(2):
        print("epoch {}".format(i))
        train_loader = train_rc_dataset.dataloader()
        for idx, (datas, labels) in enumerate(train_loader):
            if idx == 0:
                print(datas.size(), datas[0].dtype)
                print(labels)
                print(len(labels))
            if idx > 5:
                break

    for i in range(2):
        print("epoch {}".format(i))
        test_loader = test_rc_dataset.dataloader()
        for idx, (datas, labels) in enumerate(test_loader):
            if idx == 0:
                print(datas.size(), datas[0].dtype)
                print(labels)
                print(len(labels))
            if idx > 5:
                break

