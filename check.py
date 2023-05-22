#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/15:9:42
@File:           check.py
==================================================
"""


#定义模型
# 方法一：预训练模型
import torchvision
# Resnet50 = torchvision.models.resnet50(pretrained=True)
# Resnet50.fc.out_features = 10
# print(Resnet50)

from torchvision import datasets
import numpy
from datetime import datetime


def compute_mean_std(dataset):
    """compute the mean and std of cifar10 dataset
    Args:
        cifar100_training_dataset or cifar10_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    mean_r = numpy.mean(dataset.data[:, :, :, 0]) / 255.0
    mean_g = numpy.mean(dataset.data[:, :, :, 1]) / 255.0
    mean_b = numpy.mean(dataset.data[:, :, :, 2]) / 255.0

    std_r = numpy.std(dataset.data[:, :, :, 0]) / 255.0
    std_g = numpy.std(dataset.data[:, :, :, 1]) / 255.0
    std_b = numpy.std(dataset.data[:, :, :, 2]) / 255.0

    mean = mean_r, mean_g, mean_b
    std = std_r, std_g, std_b

    return mean, std

# 这样做也可以哈
def get_mean_std(dataset):
    mean = numpy.mean(dataset.data, axis=(0, 1, 2)) / 255.0
    std = numpy.std(dataset.data, axis=(0, 1, 2)) / 255.0

    return mean, std


train_cifar10_dataset = datasets.CIFAR10('cifar10', train=True, download=False)
test_cifar10_dataset = datasets.CIFAR10('cifar10', train=False, download=False)

train_cifar100_dataset = datasets.CIFAR100('cifar100', train=True, download=False)
test_cifar100_dataset = datasets.CIFAR100('cifar100', train=False, download=False)

# print(type(train_cifar10_dataset))
# print(type(train_cifar100_dataset))


train_cifar10_mean, train_cifar10_std = compute_mean_std(train_cifar10_dataset)
test_cifar10_mean, test_cifar10_std = compute_mean_std(test_cifar10_dataset)
print(train_cifar10_mean, train_cifar10_std)
print(test_cifar10_mean, test_cifar10_std)

mean, std = get_mean_std(train_cifar10_dataset)
print(mean, std)

#
# train_cifar100_mean, train_cifar100_std = compute_mean_std(train_cifar100_dataset)
# test_cifar100_mean, test_cifar100_std = compute_mean_std(test_cifar100_dataset)
# print(train_cifar100_mean, test_cifar100_mean)

if __name__ == "__main__":
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    # time of we run the script
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    print(TIME_NOW)
