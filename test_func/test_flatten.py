#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/6/9:10:41
@File:           test_flatten.py
==================================================
"""

if __name__ == "__main__":
    import numpy as np

    a = np.arange(1, 25)
    b = a.reshape(2, 2, 2, 3)  # 原始图像数据
    c = b.reshape(2, 2*2*3, 1)  # 拉平为1维数据，为了后面进行矩阵运算，对每个像素升维，这里还要多添加一个维度
    # c = b.reshape(2, 2*2*3)  # 拉平为1维数据

    print(a)
    print(b)
    print(c)
