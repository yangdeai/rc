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
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
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


class MyLoadRcData(dataset.Dataset):  # 注意父类的名称，不能写dataset
    def __init__(self, userDataset=None, txt_path=None, train=True):
        super(MyLoadRcData, self).__init__()
        self.mean = []
        self.std = []
        self.train = train

        # 定义储层的相关参数
        np.random.seed(42)

        self.inSize = 3 * 32 * 32
        self.resSize = 6

        self.initLen = 0  # int(0.01 * trainLen)  # 0 represents don't use "wash out"
        self.dataLen = len(userDataset.data) - self.initLen  #

        self.a = 0.3
        self.rho = 0.95  # spectral radius
        self.Win = np.random.randn(1, self.resSize)
        self.W = np.random.randn(self.resSize, self.resSize)
        self.X = np.zeros((self.dataLen, self.inSize, self.resSize))
        self.x = np.zeros((self.inSize, self.resSize))

        print('Calculating spectral radius...')
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
        print("Before normalized, spectral radius: rhoW = ", self.rhoW)
        self.W *= self.rho / self.rhoW  # 归一化的方式：除以最大特征的绝对值，乘以0.9 spectral radius 1.25
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))
        print("After normalized, spectral radius: rhoW = ", self.rhoW)

        if userDataset is not None:
            self.rc_reprocess(userDataset)
            # 不保存数据
            # self.save_data()
            self.img_info = self.get_img(txt_path)
        else:
            raise ValueError("The userDataset is None, please check the userDataset!")

        # # If needed, add at rc_processing
        # self.before_rc_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        # ])

        self.rc_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def rc_reprocess(self, userDataset):
        """
            RC preprocessing.
        """
        original_images = userDataset.data  # 原始数据图像: np.array(50000, 32, 32, 3) (50000,)(list)
        # (50000, 32, 32, 3)->(50000, 3, 32, 32)->(50000, 3072, 1)
        # images = original_images.transpose((0, 3, 1, 2)).reshape(-1, 3 * 32 * 32, 1)
        images = original_images.reshape(-1, 3 * 32 * 32, 1)
        self.labels = userDataset.targets
        self.labels = self.labels[self.initLen:]

        origin_num_samples = images.shape[0]
        for idx in range(origin_num_samples):
            u = images[idx]
            self.x = (1 - self.a) * self.x + self.a * np.tanh(np.dot(u, self.Win) + + np.dot(self.x, self.W))

            if idx >= self.initLen:
                self.X[idx - self.initLen] = self.x  # 这里只有x，没有u和偏置, 由于后面没有了Yout,因此不记录输入u

        in_channel = 3 * self.resSize
        self.X = self.X.reshape(-1, 32, 32, in_channel)
        if len(self.X) != len(self.labels):
            raise ValueError(f"The number of data:{len(self.X)} is not equal the number of labels:{len(self.labels)}! "
                             f"Please check them again!")
        print(f"RC samples length : {len(self.X)}, labels length : {len(self.labels)}")

        for i in range(in_channel):
            self.mean.append(np.mean(self.X[:, :, :, i]))
            self.std.append(np.std(self.X[:, :, :, i]))

        return self.X, self.labels

    def save_data(self):
        if self.train:
            rc_npy_path = './data/train/images/'
            rc_txt_path = './data/train/labels/'
        else:
            rc_npy_path = './data/test/images/'
            rc_txt_path = './data/test/labels/'

        if not os.path.exists(rc_npy_path):
            os.makedirs(rc_npy_path)
        if not os.path.exists(rc_txt_path):
            os.makedirs(rc_txt_path)

        if os.listdir(rc_npy_path) and len(os.listdir(rc_npy_path)) == len(self.labels[self.initLen:]):
            return
        elif os.listdir(rc_npy_path) and len(os.listdir(rc_npy_path)) != len(self.labels[self.initLen:]):
            for file in os.listdir(rc_npy_path):
                os.remove(rc_npy_path + file)

            for file in os.listdir(rc_txt_path):
                os.remove(rc_txt_path + file)

        labels_file = rc_txt_path + 'labels.txt'
        for idx, label in enumerate(self.labels[self.initLen:self.initLen + 20]):  # 拿20条数据测试
            imgs_file = rc_npy_path + f'rc_img_idx{idx}_label{label}.npy'
            # print(self.X[idx].shape)  # (18, 32, 32)
            np.save(imgs_file, self.X[idx])

            with open(labels_file, mode='a', encoding='utf-8') as f:
                f.write(imgs_file)
                f.write(' ' + str(label))
                f.write('\n')

        print(f"Save data finish, data number: {len(os.listdir(rc_npy_path))}")

    def get_img(self, txt_path):
        """
            Read txt file.
        """
        with open(txt_path, mode='r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split(' '), imgs_info))
            return imgs_info

    def __getitem__(self, index):
        """
        After RC processing, transforms rc_data and return (rc_images, labels) according to the given indexes.
        从这里可以看出是一个一个的样本本往外拿的，但是在光的仿真中是一个batch,一个batch的往外拿的，所以要写这这种batch的形式。
        """
        img_path, label = self.img_info[index]
        img = np.load(img_path)
        label = int(label)

        # print(img.shape)  # (32, 32, 18)

        img = self.rc_trans(img)  # 转换之后变成
        print(img.shape)  # torch.Size([18, 32, 32]) 这里变成了torch.tensor了,而且维度顺序发生了改变

        return img, label

    def __len__(self):
        return len(self.img_info[0])


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
        # transforms.ToPILImage(),
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


class WarmUpLR(_LRScheduler):
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


class LoadRcData(dataset.Dataset):  # 注意父类的名称，不能写dataset
    def __init__(self, pkl_path):
        super(LoadRcData, self).__init__()

        self.img_info = self.get_img(pkl_path)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    # 这个函数是用来读txt文档的
    def get_img(self, pkl_path):
        with open(pkl_path, mode="rb") as load_file:
            data = pickle.load(load_file)

        self.mean = data['mean']
        self.std = data['std']

        return data['images'], data['labels']

    def __getitem__(self, index):
        img, label = self.img_info[0][index], self.img_info[1][index]
        print(img.shape)
        print(label)
        img = self.transforms(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.img_info[0])


class RcPreprocess:
    def __init__(self, userDataset=None, train=True, save_path=None):
        super(RcPreprocess, self).__init__()
        self.mean = []
        self.std = []
        self.dataset = userDataset
        self.train = train
        self.save_path = save_path

        # 定义储层的相关参数
        np.random.seed(42)

        self.inSize = 3 * 32 * 32
        self.resSize = 6

        self.initLen = 100  # int(0.01 * trainLen)  # the number to "wash out" the initial state x0
        self.dataLen = len(self.dataset.data) - self.initLen

        self.a = 0.3
        self.rho = 0.95  # spectral radius
        self.Win = np.random.randn(1, self.resSize)
        self.W = np.random.randn(self.resSize, self.resSize)
        self.X = np.zeros((self.dataLen, self.inSize, self.resSize))
        self.x = np.zeros((self.inSize, self.resSize))

        print('Calculating spectral radius...')
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
        print("Before normalized, spectral radius: rhoW = ", self.rhoW)
        self.W *= self.rho / self.rhoW  # 归一化的方式：除以最大特征的绝对值，乘以0.9 spectral radius 1.25
        self.rhoW = max(abs(np.linalg.eig(self.W)[0]))
        print("After normalized, spectral radius: rhoW = ", self.rhoW)

        if self.dataset is not None:
            self.rc_reprocess()
            self.save_data()
        else:
            raise ValueError("The userDataset is None, please check the userDataset!")

    def rc_reprocess(self):
        """
            RC preprocessing.
        """
        original_images = self.dataset.data  # (50000, 32, 32, 3) (50000,)(list)
        # (50000, 32, 32, 3)->(50000, 3, 32, 32)->(50000, 3072, 1)
        images = original_images.transpose((0, 3, 1, 2)).reshape(-1, 3 * 32 * 32, 1)
        self.labels = self.dataset.targets
        self.labels = self.labels[self.initLen:]

        origin_num_samples = images.shape[0]
        for idx in range(origin_num_samples):
            u = images[idx]
            self.x = (1 - self.a) * self.x + self.a * np.tanh(np.dot(u, self.Win) + + np.dot(self.x, self.W))

            if idx >= self.initLen:
                self.X[idx - self.initLen] = self.x  # 这里只有x，没有u和偏置, 由于后面没有了Yout,因此不记录输入u

        in_channel = 3 * self.resSize
        self.X = self.X.reshape(-1, in_channel, 32, 32)
        if len(self.X) != len(self.labels):
            raise ValueError(f"The number of data:{len(self.X)} is not equal the number of labels:{len(self.labels)}!")
        print(f"RC samples length : {len(self.X)}, labels length : {len(self.labels)}")

        for i in range(in_channel):
            self.mean.append(np.mean(self.X[:, i, :, :]))
            self.std.append(np.std(self.X[:, i, :, :]))

        return self.X, self.labels

    def save_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.train:
            save_file = self.save_path + 'train_data.pkl'
        else:
            save_file = self.save_path + 'test_data.pkl'
        if os.path.exists(save_file):
            return

        rc_dataset = dict()
        rc_dataset['images'] = self.X
        rc_dataset['labels'] = self.labels
        rc_dataset['mean'] = self.mean
        rc_dataset['std'] = self.std

        with open(save_file, mode="wb") as save_file:
            pickle.dump(rc_dataset, save_file)


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        # torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        torch.backends.cudnn.benchmark = True  # False  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.deterministic = True  # 固定网络结构


class RcProject:
    def __init__(self, userDataset=None, batch_size=8, train=True, trans=True):

        self.set_seed(42)
        self.train = train
        self.trans = trans

        # 定义储层的相关参数
        self.resSize = 3  # 6
        self.batch_size = batch_size

        self.dataset = userDataset
        # np.array(50000, 32, 32, 3) (50000,)(list)
        self.origin_data = self.dataset.data
        self.inSize = self.origin_data.shape[1] * self.origin_data.shape[2] * self.origin_data.shape[3]
        self.labels = self.dataset.targets
        self.dataLen = len(self.origin_data)

        self.a = 1  # 0.3
        self.rho = 0.9  # spectral radius
        # self.Win = torch.randn(1, self.resSize, requires_grad=False)  # requires_grad=False
        # self.W = torch.randn(self.resSize, self.resSize, requires_grad=False)

        # self.Win = torch.ones(1, self.resSize, requires_grad=False)  # requires_grad=False
        # self.Win = torch.randn(1, self.resSize, requires_grad=False)  # requires_grad=False
        self.Win = torch.rand(1, self.resSize, requires_grad=False) - 0.5  # requires_grad=False
        # self.W = torch.ones(self.resSize, self.resSize, requires_grad=False)

        # self.X = torch.zeros((self.dataLen, self.inSize, self.resSize))
        self.x = torch.zeros((self.batch_size, self.inSize, self.resSize))

        # print('Calculating spectral radius for {}'.format('train ...' if self.train else 'test ...'))
        # self.rhoW = max(abs(np.linalg.eig(self.W.numpy())[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
        # print("Before normalized, spectral radius: rhoW = ", self.rhoW)
        # self.W *= self.rho / self.rhoW  # 归一化的方式：除以最大特征的绝对值，乘以 spectral radius 1.25 0.9
        # self.rhoW = max(abs(np.linalg.eig(self.W.numpy())[0]))
        # print("After normalized, spectral radius: rhoW = ", self.rhoW)

        # print("Win :", self.Win, self.Win.dtype)  # Win : tensor([[1.]]) torch.float32

        self.train_mean, self.train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)
        self.test_mean, self.test_std = (0.49421, 0.48513, 0.45041), (0.24665, 0.24289, 0.26159)

        self.train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            # # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # # # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相 全部是随机变化
            # # transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.Normalize(self.train_mean, self.train_std)
        ])

        self.test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.test_mean, self.test_std)
        ])

        if self.trans:
            self.data = self.transforms()
            # print(self.data.shape)  # torch.Size([50000, 3, 32, 32])
            self.data = self.data.reshape(-1, self.inSize, 1)
            # print(self.data.shape)  # torch.Size([50000, 3072, 1])
        else:
            # (N,H,W,C)--(N,C,H,W)--(50000, 3072, 1)
            self.data = self.origin_data.transpose((0, 3, 1, 2)).reshape(-1, self.inSize, 1) / 255.0

        self.batch_idxes = BatchSampler(RandomSampler(self.data, replacement=False),
                                        batch_size=self.batch_size,
                                        drop_last=True)
        # self.batch_idxes = BatchSampler(SequentialSampler(trans_data),
        #                                 batch_size=self.batch_size,
        #                                 drop_last=True)  # always in the same order.

    def rc_reprocess(self):
        """
            RC preprocessing.
        """

        for batch_idx in self.batch_idxes:
            # imgs = torch.from_numpy(self.data[batch_idx]).to(torch.uint8)  # uint8才能显示
            # self.imshow(imgs)
            # self.Win = self.Win.to(torch.uint8)
            # imgs_Win = torch.matmul(imgs, self.Win)
            # print(imgs_Win == imgs)  # True
            # print(imgs_Win.size())
            # self.imshow(imgs_Win)  # 乘以Win之后依然能显示
            # 上面验证reshape(-1, 3, 32, 32)还是能显示图片
            if self.trans:
                u = self.data[batch_idx]
            else:
                u = torch.from_numpy(self.data[batch_idx]).to(torch.float32)

            self.x = torch.matmul(u, self.Win)
            rc_output = self.x.reshape(-1, 3 * self.resSize, 32, 32)
            # self.x = torch.tanh(torch.matmul(u, self.Win))  # 使用激活函数
            # print(u.size())  # torch.Size([4, 3072, 1])
            # self.x = (1 - self.a) * self.x + self.a * torch.tanh(torch.matmul(u, self.Win) + torch.matmul(self.x, self.W))
            # print(u, self.x)

            # print(self.x.size())  # torch.Size([32, 3072, 6])
            # print(self.x.size(), self.x.dtype)  # torch.Size([5, 3072, 1]) torch.float32

            rc_labels = torch.tensor([self.labels[i] for i in batch_idx], dtype=torch.int64)

            yield rc_output, rc_labels

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
        # print(npimg.shape)  # (3, 36, 138)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # HWC
        plt.show()

    def transforms(self):
        trans_data = []
        for idx in range(len(self.origin_data)):
            # print(self.origin_data[idx].shape)  # (32, 32, 3)
            if self.train:
                x = self.train_trans(self.origin_data[idx])
            else:
                x = self.test_trans(self.origin_data[idx])
            trans_data.append(x)
            # print(x.shape)  # torch.Size([3, 32, 32])

        trans_data = torch.stack(trans_data, dim=0)

        return trans_data


class RcProject2:
    def __init__(self, userDataset=None, batch_size=8, train=True, trans=True):

        self.set_seed(42)
        self.train = train
        self.trans = trans

        # 定义储层的相关参数
        self.resSize = 3  # 6
        self.batch_size = batch_size

        self.dataset = userDataset
        # np.array(50000, 32, 32, 3) (50000,)(list)
        self.origin_data = self.dataset.data
        self.inSize = self.origin_data.shape[1] * self.origin_data.shape[2] * self.origin_data.shape[3]
        self.labels = self.dataset.targets
        self.dataLen = len(self.origin_data)

        self.Win = torch.randn(self.resSize, 3, requires_grad=False)  # requires_grad=False  1-->3

        self.train_mean, self.train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)
        self.test_mean, self.test_std = (0.49421, 0.48513, 0.45041), (0.24665, 0.24289, 0.26159)

        self.test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.test_mean, self.test_std)
        ])
        self.train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.train_mean, self.train_std)
        ])

        if self.trans:
            self.data = self.transforms()
            # print(self.data.shape)  # torch.Size([50000, 3, 32, 32])
            self.data = self.data.transpose(1, 2).transpose(2, 3)
            print(self.data.shape)  # torch.Size([50000, 3072, 1])   torch.Size([50000, 32, 32, 3])
        else:
            # (N,H,W,C)--(N,C,H,W)--(50000, 3072, 1)
            self.data = self.origin_data.transpose((0, 3, 1, 2)).reshape(-1, self.inSize, 1) / 255.0

        self.batch_idxes = BatchSampler(RandomSampler(self.data, replacement=False),
                                        batch_size=self.batch_size,
                                        drop_last=True)
        # self.batch_idxes = BatchSampler(SequentialSampler(trans_data),
        #                                 batch_size=self.batch_size,
        #                                 drop_last=True)  # always in the same order.

    def rc_reprocess(self):
        """
            RC preprocessing.
        """

        for batch_idx in self.batch_idxes:
            # imgs = torch.from_numpy(self.data[batch_idx]).to(torch.uint8)  # uint8才能显示
            # self.imshow(imgs)
            # self.Win = self.Win.to(torch.uint8)
            # imgs_Win = torch.matmul(imgs, self.Win)
            # print(imgs_Win == imgs)  # True
            # print(imgs_Win.size())
            # self.imshow(imgs_Win)  # 乘以Win之后依然能显示
            # 上面验证reshape(-1, 3, 32, 32)还是能显示图片
            if self.trans:
                u = self.data[batch_idx]
            else:
                u = torch.from_numpy(self.data[batch_idx]).to(torch.float32)

            self.x = torch.matmul(u, self.Win)  # (4, 32, 32, 1)
            rc_output = self.x.transpose(3, 2).transpose(2, 1)  # (4, 1, 32, 32)
            rc_labels = torch.tensor([self.labels[i] for i in batch_idx], dtype=torch.int64)

            yield rc_output, rc_labels

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
        # print(npimg.shape)  # (3, 36, 138)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # HWC
        plt.show()

    def transforms(self):
        trans_data = []
        for idx in range(len(self.origin_data)):
            # print(self.origin_data[idx].shape)  # (32, 32, 3)
            if self.train:
                x = self.train_trans(self.origin_data[idx])
            else:
                x = self.test_trans(self.origin_data[idx])
            trans_data.append(x)
            # print(x.shape)  # torch.Size([3, 32, 32])

        trans_data = torch.stack(trans_data, dim=0)

        return trans_data


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

        self.memory_out = True
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
    rcPro_train = RcProjectPre(train_dataset, batch_size=batch_size, train=True)
    rcPro_test = RcProjectPre(test_dataset, batch_size=batch_size, train=False)

    # print(len(list(train_loader)))  # 390
    for i in range(2):  # 不同的epoch，采样不同
        print("epoch {}".format(i))
        train_loader = rcPro_train.rc_reprocess()

        # train_loader = rcPro.all_rc_reprocess()
        for idx, (datas, labels) in enumerate(train_loader):
            if idx == 0:
                print(next(iter(rcPro_train.batch_idxes)))
                # print(datas)
                print(datas.size(), datas[0].dtype)  # torch.Size([5, 3, 32, 32]) torch.float32
                # datas_numpy = datas.numpy()
                # img = np.int8(datas_numpy)
                # print(img, img.dtype)
                print(labels)
            # print(len(labels))
            if idx > 5:
                break

    for i in range(2):  # 不同的epoch，采样不同
        print("epoch {}".format(i))
        test_loader = rcPro_test.rc_reprocess()
        # test_loader = rcPro_test.all_rc_reprocess()
        for idx, (datas, labels) in enumerate(test_loader):
            if idx == 0:
                print(rcPro_test.batch_idxes)
                # print(datas)
                print(datas.size(), datas[0].dtype)  # torch.Size([5, 3, 32, 32]) torch.float32
                # datas_numpy = datas.numpy()
                # img = np.int8(datas_numpy)
                # print(img, img.dtype)
                print(labels)
            # print(len(labels))
            if idx > 5:
                break

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



