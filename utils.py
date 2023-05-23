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

import numpy as np

import torch
from torch.optim.lr_scheduler import LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.data import dataset


class TemplateLoadData(dataset.Dataset):  # 注意父类的名称，不能写dataset
    def __init__(self, txt_path, train=True):
        super(TemplateLoadData, self).__init__()
        self.img_info = self.get_img(txt_path)
        self.train = train

        # train预处理
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # test预处理
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # 这个函数是用来读txt文档的
    def get_img(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()  # 这里是读取所有文件信息，包括文件路径和标签
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
            return imgs_info

    def __getitem__(self, index):
        img_path, label = self.img_info[index]
        img = Image.open(img_path)  # 这里才是图片，上面的只是读图片路径和标签
        label = int(label)

        # 注意区分预处理
        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        return img, label

    def __len__(self):
        return len(self.img_info)


class LoadImgXOpticalData(dataset.Dataset):  # 注意父类的名称，不能写dataset
    def __init__(self, npy_path):
        super(LoadImgXOpticalData, self).__init__()
        self.img_info = self.get_img(npy_path)

    # 这个函数是用来读txt文档的
    def get_img(self, pkl_path):
        with open(pkl_path, mode="rb") as load_file:
            dummy_dataset = pickle.load(load_file)  # 这里直接拿到图片imgs，和labels: XOptical (50000, 3072) (50000, 3072, 6)

        return dummy_dataset['img'], dummy_dataset['XOptical']  # 这是一个字典

    def __getitem__(self, index):

        img, label = self.img_info[0][index], self.img_info[1][index]

        return img, label

    def __len__(self):
        return len(self.img_info[0])




def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.resnet import resnet101
        net = resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar10 test_func dataset
        std: std of cifar10 test_func dataset
        path: path to cifar10 test_func python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar10_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

def compute_mean_std(dataset):
    """
    pytorch Version to compute mean and std.
    compute the mean and std of cifar10 dataset
    Args:
        cifar10_training_dataset or cifar10_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    mean_r = np.mean(dataset.data[:, :, :, 0]) / 255.0
    mean_g = np.mean(dataset.data[:, :, :, 1]) / 255.0
    mean_b = np.mean(dataset.data[:, :, :, 2]) / 255.0

    std_r = np.std(dataset.data[:, :, :, 0]) / 255.0
    std_g = np.std(dataset.data[:, :, :, 1]) / 255.0
    std_b = np.std(dataset.data[:, :, :, 2]) / 255.0

    mean = mean_r, mean_g, mean_b
    std = std_r, std_g, std_b

    return mean, std


def check_mean_std(data):
    mean_r = np.mean(data[:, 32])
    mean_g = np.mean(data[:, 32: 32 * 2])
    mean_b = np.mean(data[:, 32: 32 * 3])

    std_r = np.std(data[:, 32])
    std_g = np.std(data[:, 32: 32 * 2])
    std_b = np.std(data[:, 32: 32 * 3])

    mean = mean_r, mean_g, mean_b
    std = std_r, std_g, std_b

    return mean, std


def get_mean_std(data):
    mean_r = np.mean(data[:, 0])
    mean_g = np.mean(data[:, :, 1])
    mean_b = np.mean(data[:, :, 2])

    std_r = np.std(data[:, :, 0])
    std_g = np.std(data[:, :, 1])
    std_b = np.std(data[:, :, 2])

    mean = mean_r, mean_g, mean_b
    std = std_r, std_g, std_b

    return mean, std

class WarmUpLR(LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]



if __name__ == "__main__":

    from torch.utils.data import dataloader
    BATCH_SIZE = 2

    pkl_path = './dummy_dataset.pkl'
    dummy_dataset = LoadImgXOpticalData(pkl_path)
    dummy_dataloader = dataloader.DataLoader(dataset=dummy_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             drop_last=True)

    for image, label in dummy_dataloader:
        print(image)
        print(image.shape)
        print(label)
        print(label.shape)

    # from torchvision import datasets
    # train_cifar_dataset = datasets.CIFAR10('cifar10', train=True, download=False)
    # test_cifar_dataset = datasets.CIFAR10('cifar10', train=False, download=False)

    # print(train_cifar_dataset.data.shape)  # (50000, 32, 32, 3)
    # train_mean, train_std = compute_mean_std(train_cifar_dataset)
    # test_mean, test_std = compute_mean_std(test_cifar_dataset)
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
    # for i in range(train_cifar_dataset.data.shape[0]):
    #     # print(train_cifar_dataset.data[i].shape)  # (32, 32, 3)
    #     train_data = train_data_transform(train_cifar_dataset.data[i])
    #     train_data = train_data.reshape(3072,)
    #     train_data_batches.append(train_data)
    # train_data_stack = np.stack(train_data_batches)
    #
    # for i in range(test_cifar_dataset.data.shape[0]):
    #     test_data = test_data_transform(test_cifar_dataset.data[i])
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


    
    
    
    
