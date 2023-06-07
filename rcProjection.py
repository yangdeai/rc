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
    def __init__(self, userDataset=None, in_num=1, out_num=6, structure="44", seed=1234):

        np.random.seed(seed)  # 随机种子，用于结果可复现

        self.in_num = in_num  # 输入端口数
        self.out_num = out_num  # 输出端口数
        self.structure = structure  # 芯片结构类型

        self.dataset = userDataset
        self.data = self.dataset.data  # 原始图像数据，格式：np.array(50000, 32, 32, 3)
        self.inSize = self.data.shape[1] * self.data.shape[2] * self.data.shape[3]  # 32*32*3=3072
        # 每张图片都拉成一个个的像素：(50000, 32, 32, 3)--(50000, 3072, 1)
        # 满足条件: self.inSzie * self.in_num == 3072, 不同的数据 shape 对应 输入输出端口组合
        self.data = self.data.reshape(-1, self.inSize, self.in_num)
        self.labels = self.dataset.targets  # 原始图像数据标签，格式：(50000,)，是一个list

        self.Win = np.zeros((1, self.out_num))

    def rc_project(self):
        """
        不同的结构对应不同的光芯片,即对应不同的PDA系统，对应不同的连接权重Win, 也就对应不同的RC projection.
        structure <--> PDA <--> Win <--> RC_project.当芯片制造出来Win也就固定住，相同输入输出端口下，同一结构的芯片的RC_project相同。
        矩阵运算模拟RC_project，数据shape变化：(50000, 3072, 1) X (1, self.out_num) == (50000, 3072, self.out_num)
        """
        assert isinstance(self.structure, str)
        if self.structure == "44":
            self.Win = np.random.randn(1, self.out_num)
            rc_data = np.matmul(self.data, self.Win)
        elif self.structure == "48":
            self.Win = np.random.rand(1, self.out_num)
            rc_data = np.matmul(self.data, self.Win)
        elif self.structure == "12":
            self.Win = np.random.uniform(low=0, high=2, size=(1, self.out_num))
            rc_data = np.matmul(self.data, self.Win)
        elif self.structure == "56":
            self.Win = np.random.uniform(low=-1, high=1, size=(1, self.out_num))
            rc_data = np.matmul(self.data, self.Win)
        elif self.structure == "diffraction":
            self.Win = np.random.uniform(low=1, high=2, size=(1, self.out_num))
            rc_data = np.matmul(self.data, self.Win)
        else:
            raise ValueError("This structure is not supported yet.")

        return rc_data


if __name__ == "__main__":

    random_seed = 1234
    batch_size = 4
    input_port_num = 1  # 输入端口数
    output_port_num = 6  # 输出端口数
    train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)  # 训练数据集
    test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)  # 测试数据集

    structures = ["44", "48", "12", "56", "diffraction"]
    for structure in structures:
        train_rc_project = RcProject(train_dataset, in_num=input_port_num, out_num=output_port_num, structure=structure, seed=random_seed)
        test_rc_project = RcProject(test_dataset, in_num=input_port_num, out_num=output_port_num,  structure=structure, seed=random_seed)

        train_rc_data = train_rc_project.rc_project()
        test_rc_data = test_rc_project.rc_project()

        print("经过储备池处理之后，数据shape变化为：\n structure: {0}, train data's shape: {1}, test data's shape: {2}".format(
              structure, train_rc_data.shape, test_rc_data.shape))

        # 打印如下：
        # 经过储备池处理之后，数据shape变化为：
        # structure: XX, train data 's shape: (50000, 3072, 6), test data's shape: (10000, 3072, 6)

