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
import pickle
import sys
import re
import datetime
import shutil
import pickle
import matplotlib.pyplot as plt

import numpy as np

import torch
from PIL.Image import Image
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler

from torch.utils.data import dataset


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



"""
把下面的类拆分为函数的形式
"""
class RcProjectPre:
    def __init__(self, userDataset=None, batch_size=8, train=True, trans=True):

        self.set_seed(1234)
        self.train = train
        self.trans = trans

        # 定义储层的相关参数
        self.resSize = 6  # 6
        self.batch_size = batch_size

        self.dataset = userDataset
        self.origin_data = self.dataset.data  # np.array(50000, 32, 32, 3)
        self.inSize = self.origin_data.shape[1] * self.origin_data.shape[2] * self.origin_data.shape[3]
        self.data = self.origin_data.reshape(-1, self.inSize, 1)  # (N,H,W,C)--(50000, 3072, 1)
        self.labels = self.dataset.targets  # (50000,)(list)

        self.x = np.zeros((self.batch_size, self.inSize, self.resSize))
        self.Win = np.random.randn(1, self.resSize)  # - 0.5

        self.rc_project()  # rc project

        self.rc_trans_data_path = './rc_project/'
        if not os.path.exists(self.rc_trans_data_path):
            os.makedirs(self.rc_trans_data_path)

        if self.train:
            self.rc_trans_data = self.rc_trans_data_path + f'train_rc_data_rs{self.resSize}.pt'
            self.mean = self.rc_trans_data_path + f'train_mean_rs{self.resSize}.npy'
            self.std = self.rc_trans_data_path + f'train_std_rs{self.resSize}.npy'
        else:
            self.rc_trans_data = self.rc_trans_data_path + f'test_rc_data_rs{self.resSize}.pt'
            self.mean = self.rc_trans_data_path + f'test_mean_rs{self.resSize}.npy'
            self.std = self.rc_trans_data_path + f'test_std_rs{self.resSize}.npy'

        if not os.path.isfile(self.mean):
            self.mean, self.std = self.get_mean_std()
        else:
            self.mean, self.std = np.load(self.mean), np.load(self.std)

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

        self.memory_out = False
        if not self.memory_out:
            if not os.path.isfile(self.rc_trans_data):
                self.rc_trans_data = self.all_transforms()
            else:
                self.rc_trans_data = torch.load(self.rc_trans_data)

            self.batch_idxes = BatchSampler(RandomSampler(self.rc_trans_data, replacement=False),
                                            batch_size=self.batch_size,
                                            drop_last=True)
        else:
            self.batch_idxes = BatchSampler(RandomSampler(self.data, replacement=False),
                                        batch_size=self.batch_size,
                                        drop_last=True)

    def rc_project(self):
        """ RC rc_project. """

        self.data = np.matmul(self.data, self.Win)  # RC project  (50000, 3072, self.resSize)
        self.data = np.reshape(self.data, (-1, 32, 32, 3 * self.resSize))

    def rc_reprocess(self):
        for batch_idx in self.batch_idxes:
            batch_trans_data = self.transforms(batch_idx).to(torch.float32)
            rc_labels = torch.tensor([self.labels[i] for i in batch_idx], dtype=torch.int64)

            yield batch_trans_data, rc_labels

    def all_rc_reprocess(self):
        """
            生成器，用于取数据用于训练或者测试，不放在这里。
        :return:
        """
        for batch_idx in self.batch_idxes:
            batch_trans_data = []
            for idx in batch_idx:
                x = self.rc_trans_data[idx].to(torch.float32)
                batch_trans_data.append(x)
            batch_trans_data = torch.stack(batch_trans_data, dim=0)
            rc_labels = torch.tensor([self.labels[i] for i in batch_idx], dtype=torch.int64)

            yield batch_trans_data, rc_labels

    def get_mean_std(self):
        channel = 3 * self.resSize
        means = [0 for _ in range(channel)]
        stds = [0 for _ in range(channel)]
        num_data = len(self.data)
        for idx in range(num_data):
            for i in range(channel):
                # print(self.data[idx].shape)  # torch.Size([9, 32, 32])
                means[i] += self.data[idx][i, :, :].mean()
                stds[i] += self.data[idx][i, :, :].std()

        mean = np.array(means) / num_data
        std = np.array(stds) / num_data

        # save data
        np.save(self.mean, mean)
        np.save(self.std, std)

        return mean, std

    def set_seed(self, seed):
        torch.manual_seed(seed)  # 固定随机种子（CPU）
        np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
        if torch.cuda.is_available():  # 固定随机种子（GPU)
            torch.cuda.manual_seed(seed)  # 为当前GPU设置
            torch.backends.cudnn.benchmark = True  # False  # GPU、网络结构固定，可设置为True
            torch.backends.cudnn.deterministic = True  # 固定网络结构

    def imshow(self, imgs):  # self.data[batch_idx]
        imgs = imgs.reshape(-1, 3, 32, 32)
        grid_imgs = torchvision.utils.make_grid(imgs)
        npimg = grid_imgs.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # HWC
        plt.show()

    def transforms(self, batch_idx):
        batch_trans_data = []
        for idx in batch_idx:
            if self.train:
                x = self.train_trans(self.data[idx])
            else:
                x = self.test_trans(self.data[idx])
            batch_trans_data.append(x)

        batch_trans_data = torch.stack(batch_trans_data, dim=0)

        return batch_trans_data

    def all_transforms(self):
        all_trans_data = []
        num_data = len(self.data)
        for idx in range(num_data):
            if self.train:
                x = self.train_trans(self.data[idx])
            else:
                x = self.test_trans(self.data[idx])
            all_trans_data.append(x)

        all_trans_data = torch.stack(all_trans_data, dim=0)

        # save data
        torch.save(all_trans_data, self.rc_trans_data)

        return all_trans_data





if __name__ == "__main__":

    batch_size = 4
    train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)
    test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)
    rcPro = RcProjectPre(train_dataset, batch_size=batch_size, train=True)
    rcPro_test = RcProjectPre(test_dataset, batch_size=batch_size, train=False)
    # print(len(list(train_loader)))  # 390
    for i in range(2):  # 不同的epoch，采样不同
        print("epoch {}".format(i))
        train_loader = rcPro.rc_reprocess()
        # train_loader = rcPro.all_rc_reprocess()
        for idx, (datas, labels) in enumerate(train_loader):
            if idx == 0:
                # print(datas)
                print(datas.size(), datas[0].dtype)  # torch.Size([5, 3, 32, 32]) torch.float32
                # datas_numpy = datas.numpy()
                # img = np.int8(datas_numpy)
                # print(img, img.dtype)
                print(labels)
            # print(len(labels))
            # if idx > 5:
            #     break

        for i in range(2):  # 不同的epoch，采样不同
            print("epoch {}".format(i))
            test_loader = rcPro_test.rc_reprocess()
            # test_loader = rcPro_test.all_rc_reprocess()
            for idx, (datas, labels) in enumerate(test_loader):
                if idx == 0:
                    # print(datas)
                    print(datas.size(), datas[0].dtype)  # torch.Size([5, 3, 32, 32]) torch.float32
                    # datas_numpy = datas.numpy()
                    # img = np.int8(datas_numpy)
                    # print(img, img.dtype)
                    print(labels)
                # print(len(labels))
                # if idx > 5:
                #     break

        # self.normalized_data = self.origin_data / 255.0
        #
        # if self.train:
        #     self.standard_data_r = (self.normalized_data[:, :, :, 0] - self.train_mean[0]) / self.train_std[0]
        #     self.standard_data_g = (self.normalized_data[:, :, :, 1] - self.train_mean[1]) / self.train_std[1]
        #     self.standard_data_b = self.normalized_data[:, :, :, 2] - self.train_mean[2] / self.train_std[2]
        # else:
        #     self.standard_data_r = (self.normalized_data[:, :, :, 0] - self.test_mean[0]) / self.test_std[0]
        #     self.standard_data_g = (self.normalized_data[:, :, :, 1] - self.test_mean[1]) / self.test_std[1]
        #     self.standard_data_b = (self.normalized_data[:, :, :, 2] - self.test_mean[2]) / self.test_std[2]
        #
        # self.origin_data = np.stack([self.standard_data_r, self.standard_data_g, self.standard_data_b], axis=-1)
        # # print(self.origin_data.shape) # (50000, 32, 32, 3)
    # # all save
    # from torchvision import datasets
    # from torch.utils.data import dataloader
    #
    # origin_train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)
    # origin_test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)
    #
    # train_path = './data/train/'
    # test_path = './data/test/'
    #
    # RcPreprocess(origin_train_dataset, train=True, save_path=train_path)
    # RcPreprocess(origin_test_dataset, train=False, save_path=test_path)
    #
    # train_data = train_path + 'train_data.pkl'
    # test_data = test_path + 'test_data.pkl'
    #
    # BATCH_SIZE = 2
    #
    # train_dataset = LoadRcData(train_data)
    # test_dataset = LoadRcData(test_data)
    #
    # train_loader = dataloader.DataLoader(train_dataset,
    #                           batch_size=BATCH_SIZE,
    #                           num_workers=4,
    #                           shuffle=True,
    #                           drop_last=True)
    # test_loader = dataloader.DataLoader(test_dataset,
    #                          batch_size=BATCH_SIZE,
    #                          num_workers=4,
    #                          shuffle=False,
    #                          drop_last=False)
    # for imgs, labels in train_loader:
    #     print(imgs)
    #     print(imgs.shape)
    #     print(labels)
    #     print(len(labels))

    # dummy_dataloader = dataloader.DataLoader(dataset=dummy_dataset,
    #                                          batch_size=BATCH_SIZE,
    #                                          shuffle=True,
    #                                          drop_last=True)
    #
    # for image, label in dummy_dataloader:
    #     print(image)
    #     print(image.shape)
    #     print(label)
    #     print(label.shape)

    # # single save
    # from torchvision import datasets
    # from torch.utils.data import dataloader
    #
    # origin_train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)
    # origin_test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)
    #
    # train_txt_path = './data/train/labels/labels.txt'
    # test_txt_path = './data/test/labels/labels.txt'
    #
    # rc_train_dataset = MyLoadRcData(origin_train_dataset, train=True, txt_path=train_txt_path)
    # rc_test_dataset = MyLoadRcData(origin_test_dataset, train=False, txt_path=test_txt_path)
    #
    # BATCH_SIZE = 2
    #
    # train_loader = dataloader.DataLoader(rc_train_dataset,
    #                           batch_size=BATCH_SIZE,
    #                           num_workers=0,
    #                           shuffle=True,
    #                           drop_last=False)
    # test_loader = dataloader.DataLoader(rc_test_dataset,
    #                          batch_size=BATCH_SIZE,
    #                          num_workers=0,
    #                          shuffle=False,
    #                          drop_last=False)
    #
    # for imgs, labels in train_loader:
    #     print(imgs)
    #     print(imgs.shape)  # torch.Size([2, 18, 32, 32])
    #     print(labels)  # torch.Size([2, 18, 32, 32])
    #     print(len(labels))

    #
    # from torch.utils.data import dataloader
    # BATCH_SIZE = 2
    #
    # pkl_path = './dummy_dataset.pkl'
    # dummy_dataset = LoadImgXOpticalData(pkl_path)
    # print(len(dummy_dataset))
    # dummy_dataloader = dataloader.DataLoader(dataset=dummy_dataset,
    #                                          batch_size=BATCH_SIZE,
    #                                          shuffle=True,
    #                                          drop_last=True)
    #
    # for image, label in dummy_dataloader:
    #     print(image)
    #     print(image.shape)
    #     print(label)
    #     print(label.shape)

    # from torchvision import datasets
    # train_dataset = datasets.CIFAR10('cifar10', train=True, download=False)
    # test_dataset = datasets.CIFAR10('cifar10', train=False, download=False)

    # print(train_dataset.data.shape)  # (50000, 32, 32, 3)
    # train_mean, train_std = compute_mean_std(train_dataset)
    # test_mean, test_std = compute_mean_std(test_dataset)
    # # (0.49139967861519607, 0.48215840839460783, 0.44653091444546567) (0.2470322324632819, 0.24348512800005573, 0.26158784172796434)
    # print(train_mean, train_std)
    # # # (0.4942142800245098, 0.4851313890165441, 0.4504090927542892) (0.24665251509498004, 0.24289226346005355, 0.26159237802202373)
    # # print(test_mean, test_std)

    # # imageNet mean std
    # train_mean, train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)
    # test_mean, test_std = (0.49421, 0.48513, 0.45041), (0.24665, 0.24289, 0.26159)
    #
    # train_data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.RandomCrop(32, padding=4),
    #     # transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.Normalize(train_mean, train_std)
    # ])
    # test_data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(test_mean, test_std)
    # ])
    #
    # train_data_batches = []
    # test_data_batches = []
    # for i in range(train_dataset.data.shape[0]):
    #     # print(train_dataset.data[i].shape)  # (32, 32, 3)
    #     train_data = train_data_transform(train_dataset.data[i])
    #     train_data = train_data.reshape(3072,)
    #     train_data_batches.append(train_data)
    # train_data_stack = np.stack(train_data_batches)
    #
    # for i in range(test_dataset.data.shape[0]):
    #     test_data = test_data_transform(test_dataset.data[i])
    #     test_data = test_data.reshape(3072, )
    #     test_data_batches.append(test_data)
    # test_data_stack = np.stack(test_data_batches)
    #
    # print(train_data_stack.shape, test_data_stack.shape)  # (50000, 3072) (10000, 3072)
    # train_data_mean, train_data_std = check_mean_std(train_data_stack)
    # print(train_data_mean, train_data_std)  # (0.07607781, 0.11612499, 0.109880544) (1.15383, 1.115467, 1.1048968)
    # test_data_mean, test_data_std = check_mean_std(test_data_stack)
    # print(test_data_mean, test_data_std)  # (0.07607781, 0.11612499, 0.109880544) (1.15383, 1.115467, 1.1048968)
    #
    # # 这个使用ImageNet的均值和方差
    # # (0.07591832, 0.11596749, 0.1097227)(1.1538872, 1.1155221, 1.1049513)
    # # (0.060680583, 0.11857232, 0.11297002)(1.1560373, 1.1133307, 1.1015073)

    #     self.train_mean = (0.49144, 0.48222, 0.44652)
    #     self.train_std = (0.24702, 0.24349, 0.26166)
    #     self.train_transforms = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    #         transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
    #         transforms.Normalize(self.train_mean, self.train_std)
    #     ])
    #     self.before_rc_preprocess()
    # # def before_rc_preprocess(self):
    #     if True:
    #         for idx in range(len(self.origin_data)):
    #             self.train_transforms(self.origin_data[idx])  # 只能一个个的处理
    #         pass
    #     else:
    #         pass
    #



