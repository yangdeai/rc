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


class OpticalOpt:
    """
        模拟光算子操作。
    """
    def __init__(self, data=None, structure="RC_44", in_num=1, out_num=6):

        self.data = data  # 输入数据
        self.in_num = in_num  # 输入端口数
        self.out_num = out_num  # 输出端口数
        self.structure = structure  # 芯片结构类型

    def optical_operator(self):
        """
        根据structure、输入端口数、输出端口数选择对应的光算子操作。
        W是模拟光芯片固化后权重.
        :return: opt_data经过光算子操作后的数据
        """
        if self.structure == "RC_44":
            if self.in_num == 1 and self.out_num == 6:
                np.random.seed(1234)  # 随机种子，用于电仿真结果复现，与光算子操作无关
                W = np.random.randn(1, self.out_num)
                opt_data = np.matmul(self.data, W)
            elif self.in_num == 1 and self.out_num == 2:
                np.random.seed(1234)
                W = np.random.randn(1, self.out_num)
                opt_data = np.matmul(self.data, W)
            elif self.in_num == 1 and self.out_num == 4:
                np.random.seed(1234)
                W = np.random.randn(1, self.out_num)
                opt_data = np.matmul(self.data, W)
            elif self.in_num == 1 and self.out_num == 8:
                np.random.seed(1234)
                W = np.random.randn(1, self.out_num)
                opt_data = np.matmul(self.data, W)
            else:
                pass
        elif self.structure == "RC_48":
            pass
        elif self.structure == "RC_12":
            pass
        elif self.structure == "RC_56":
            pass
        elif self.structure == "diffraction_X1":
            pass
        elif self.structure == "diffraction_X2":
            pass
        elif self.structure == "diffraction_X3":
            pass
        else:
            pass

        return opt_data


