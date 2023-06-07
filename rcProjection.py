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

import numpy as np
from torchvision import datasets


class RcProject:
    def __init__(self, userDataset=None, in_num=1, out_num=6, seed=1234):

        np.random.seed(seed)  # 随机种子

        self.in_num = in_num  # 输入端口数
        self.out_num = out_num  # 输出端口数

        self.dataset = userDataset
        self.data = self.dataset.data  # 原始图像数据，格式：np.array(50000, 32, 32, 3)
        self.inSize = self.data.shape[1] * self.data.shape[2] * self.data.shape[3]  # 32*32*3=3072
        # 每张图片都拉成一个个的像素：(50000, 32, 32, 3)--(50000, 3072, 1)
        self.data = self.data.reshape(-1, self.inSize, self.in_num)  # 满足条件: self.inSzie * self.in_num == 3072
        self.labels = self.dataset.targets  # 原始图像数据标签，格式：(50000,)，是一个list

        self.Win = np.random.randn(1, self.out_num)  # 正态分布中随机采样，作为储备池升维处理的权重

    def rc_project(self):
        # 矩阵运算模拟储备池处理，数据shape变化：(50000, 3072, 1) X (1, self.out_num) == (50000, 3072, self.out_num)
        rc_data = np.matmul(self.data, self.Win)  
        return rc_data


if __name__ == "__main__":

    batch_size = 4
    input_port_num = 1  # 输入端口数
    output_port_num = 6  # 输出端口数
    train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)  # 训练数据集
    test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)  # 测试数据集
    train_rc_project = RcProject(train_dataset, in_num=input_port_num, out_num=output_port_num)
    test_rc_project = RcProject(test_dataset, in_num=input_port_num, out_num=output_port_num)

    train_rc_data = train_rc_project.rc_project()
    test_rc_data = test_rc_project.rc_project()

    print("经过储备池处理之后，数据shape变化为：\n, train data's shape: {0}, test data's shape: {1}".format(
        train_rc_data.shape, test_rc_data.shape))

    # 打印如下：
    # 经过储备池处理之后，数据shape变化为：
    # , train data 's shape: (50000, 3072, 6), test data's shape: (10000, 3072, 6)

