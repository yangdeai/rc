#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/5/24:8:58
@File:           esn3.py
==================================================
"""

if __name__ == "__main__":
    import numpy as np
    from scipy import linalg
    np.random.seed(42)

    inSize = 32  # 3*32*32
    resSize = 6
    bias = np.random.randn(1, )
    a = 0.3
    rho = 0.95
    data = np.random.randn(500, inSize, 1)

    trainLen = 300
    testLen = 100
    initLen = 100  # int(0.01 * trainLen)
    samples = trainLen - initLen

    Win = np.random.randn(1, resSize)  # - 0.5  #
    W = np.random.randn(resSize, resSize)  # - 0.5  #
    print('计算谱半径...')
    rhoW = max(abs(np.linalg.eig(W)[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
    print("rhoW = ", rhoW)
    W *= rho / rhoW  # 归一化的方式：除以最大特征的绝对值，乘以0.9 spectral radius 1.25
    rhoW = max(abs(np.linalg.eig(W)[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
    print("rhoW = ", rhoW)

    Xout = np.zeros((samples, inSize, resSize))
    x = np.zeros((inSize, resSize))

    for idx in range(trainLen):
        u = data[idx]
        x = (1 - a) * x + a * np.tanh(np.dot(u, Win) + + np.dot(x, W))

        if idx >= initLen:
            Xout[idx - initLen] = x  # 这里只有x，没有u和偏置, # 由于后面没有了Yout,因此不记录输入u

    print(Xout.shape)  # (297, 32, 6) (samples, inSize, resSize)
    # x0 = np.zeros((inSize, resSize))
    # xout = np.dot(x0, W)
    # print(xout.shape) # (32, 6) (inSize, resSize)
    # u = data[0]
    # uin = np.dot(u, Win)
    # print(uin.shape) # (32, 6) (inSize, resSize)

