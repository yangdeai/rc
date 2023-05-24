#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/5/23:9:04
@File:           view_data.py
==================================================
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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
    mean_r = np.mean(data[:, 32]) / 255.0
    mean_g = np.mean(data[:, 32: 32 * 2]) / 255.0
    mean_b = np.mean(data[:, 32: 32 * 3]) / 255.0

    std_r = np.std(data[:, 32]) / 255.0
    std_g = np.std(data[:, 32: 32 * 2]) / 255.0
    std_b = np.std(data[:, 32: 32 * 3]) / 255.0

    mean = mean_r, mean_g, mean_b
    std = std_r, std_g, std_b

    return mean, std


def read_data(data_path, train=True):
    data_batches = []

    if train:
        for i in range(1, 6):
            file = data_path + f'data_batch_{i}'
            data_dict = unpickle(file)
            img = data_dict[b'data']
            data_batches.append(img)
        transform_data = np.vstack(data_batches)
    else:
        file = data_path + f'test_batch'
        data_dict = unpickle(file)
        transform_data = data_dict[b'data']

    return transform_data


def prepared_data(data_path, transforms=None, train=True):
    data_batches = []
    if train:
        for i in range(1, 6):
            file = data_path + f'data_batch_{i}'
            data_dict = unpickle(file)
            img = data_dict[b'data']
            for idx in range(img.shape[0]):
                trans_before = img[idx].reshape(32, 32, 3)
                if transforms is not None:
                    trans_after = transforms(trans_before)
                    trans_after_reshape = trans_after.reshape(3072, )
                else:
                    trans_after_reshape = trans_before
                data_batches.append(trans_after_reshape)
    else:
        file = data_path + f'test_batch'
        data_dict = unpickle(file)
        img = data_dict[b'data']
        for idx in range(img.shape[0]):
            trans_before = img[idx].reshape(32, 32, 3)
            if transforms is not None:
                trans_after = transforms(trans_before)
                trans_after_reshape = trans_after.reshape(3072,)
            else:
                trans_after_reshape = trans_before
            data_batches.append(trans_after_reshape)

    transform_data = np.vstack(data_batches)
    print(transform_data.shape)  # (50000, 3072)  (10000, 3072)
    mean, std = check_mean_std(transform_data)
    print(mean, std)  # (-1.2365744, -0.733376, -0.6329356) (0.004542303085327148, 1.26946, 1.262221)
    if train:
        np.save('./cifar10_train_50000_3072.npy', transform_data)
    else:
        np.save('./cifar10_test_10000_3072.npy', transform_data)


if __name__ == "__main__":
    # torch版的均值和方差
    # train_mean, train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)
    # test_mean, test_std = (0.49421, 0.48513, 0.45041), (0.24665, 0.24289, 0.26159)
    # 3072numpy版的均值和方差
    # train_mean, train_std = (0.5101933333333333, 0.5200862794117648, 0.5185436973039216), (0.2850332067371592, 0.27555628568631496, 0.2729450751468739)
    # test_mean, test_std = (0.5091768627450981, 0.5234558578431373, 0.5220740441176471), (0.28513658697757116, 0.2746030005860156, 0.27168674872169135)

    # imageNet的均值和标准差
    train_mean, train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)
    test_mean, test_std = (0.49421, 0.48513, 0.45041), (0.24665, 0.24289, 0.26159)

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),  # 这个放在进入储层之后出来的的
        transforms.RandomCrop(32, padding=4),  # 由于使用储层进行了预处理这里需不需要还要讨论一下:这里可以看作是一种预处理方式
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(train_mean, train_std)  # 这里减去的是原来数据的均值和标准差,所以应该放在储层之前进行的,但是深度学习哪里是放在上面的处理之后的出来的
    ])
    test_data_transform = transforms.Compose([
        transforms.ToTensor(),  # 放在储层之后出来的
        transforms.Normalize(test_mean, test_std)  #
    ])
    data_path = './cifar10/cifar-10-batches-py/'

    # train_data = read_data(data_path, train=True)
    # train_mean, train_std = get_mean_std(train_data)
    # print(train_mean, train_std) # (0.5101933333333333, 0.5200862794117648, 0.5185436973039216) (0.2850332067371592, 0.27555628568631496, 0.2729450751468739)
    # test_data = read_data(data_path, train=False)
    # test_mean, test_std = get_mean_std(test_data)
    # print(test_mean, test_std) # (0.5091768627450981, 0.5234558578431373, 0.5220740441176471) (0.28513658697757116, 0.2746030005860156, 0.27168674872169135)

    prepared_data(data_path=data_path, transforms=train_data_transform, train=True)
    prepared_data(data_path=data_path, transforms=train_data_transform, train=False)
    # 没有数据增强的均值和标准差，可以看到很接近(0， 1)
    # (0.0657999, 0.08431891, 0.06792943)(1.131884, 1.0584569, 1.0349101)
    # (0.067010574, 0.101131424, 0.08604068)(1.1269815, 1.0567623, 1.0350227)

    # train_dataset = datasets.CIFAR10('cifar10', train=True, download=False, transform=train_data_transform)
    # test_dataset = datasets.CIFAR10('cifar10', train=False, download=False, transform=test_data_transform)

    # input
    # train_data = train_dataset.data.reshape(50000, 32*32*3)  # (50000, H, W, C)
    # test_data = test_dataset.data.reshape(10000, 32*32*3)  # (10000, H, W, C)
    # print(train_data[0, :]) # [ 59  62  63 ... 123  92  72] 还没有transform
    # (50000, 32, 32, 3) (10000, 32, 32, 3) <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    # print(train_data.shape, test_data.shape, type(train_data), type(test_data))
    # x_optical_labels = np.random.randn(50000, 32*32*3, 6)  #
    # print(X_optical_labels.shape)  # (50000, 3072, 6)
    # np.save('./x_optical_labels.npy', x_optical_labels)
    # x_optical_labels_loaded = np.load('./x_optical_labels.npy')
    # print(x_optical_labels_loaded.shape)  # (50000, 3072, 6)
    # dataset_info = [train_data, x_optical_labels_loaded]
    # print(dataset_info)

    # dict = unpickle(data_dir_path + f'{1}')
    # print(dict)
    # print(type(dict[b'data']))  # <class 'numpy.ndarray'> dtype=uint8
    # print(dict[b'data'])  #
    # print(dict[b'data'].shape)  # (10000, 3072)

    # data_info = [(data, x_optical_labels_loaded)]
    # for iter, (data, labels) in enumerate(data_info):
    #     print(iter, data.shape, labels.shape)  #

    # # view one image
    # data_batch_1 = data_path + 'data_batch_1'
    # dict = unpickle(data_batch_1)
    # img = dict[b'data']
    # print(img.shape)  # (10000, 3072)
    # show_img = img[666]
    # print(show_img.shape)  # (3072,)
    # img_reshape = show_img.reshape(3, 32, 32)
    # pic = img_reshape.transpose(1, 2, 0)  # (3, 32, 32) --> (32, 32, 3)
    # import matplotlib.pyplot as plt
    # plt.imshow(pic)
    # plt.show()
    #
    # label = dict[b'labels']
    # image_label = label[666]
    # print(image_label)  # 9














