
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
import numpy as np
import torch
from scipy import linalg


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        # torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        torch.backends.cudnn.benchmark = True # False  # GPU、网络结构固定，可设置为True
        torch.backends.cudnn.deterministic = True  # 固定网络结构


if __name__ == "__main__":

    same_seeds(1234)

    batch_size = 8

    inSize = 32  # 3*32*32
    resSize = 6
    bias = np.random.randn(1, )
    a = 0.3
    rho = 0.95
    data = np.random.randn(500, inSize, 1)

    trainLen = 500
    testLen = 100
    initLen = 100  # int(0.01 * trainLen)
    samples = trainLen  #  - initLen

    Win = np.random.randn(1, resSize)  # - 0.5  # rand()均匀采样的话减去0.5，正态分布的话不用
    W = np.random.randn(resSize, resSize)  # - 0.5  #
    print('计算谱半径...')
    rhoW = max(abs(np.linalg.eig(W)[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
    print("rhoW = ", rhoW)
    W *= rho / rhoW  # 归一化的方式：除以最大特征的绝对值，乘以0.9 spectral radius 1.25
    rhoW = max(abs(np.linalg.eig(W)[0]))  # linalg.eig(W)[0]:特征值 linalg.eig(W)[1]:特征向量
    print("rhoW = ", rhoW)

    print(W.shape, Win.shape)  # (6, 6) (1, 6)

    Xout = np.zeros((samples, inSize, resSize))
    x = np.zeros((batch_size, inSize, resSize))  # (8, 32, 6)
    print(Xout.shape, x.shape)  # (500, 32, 6) (8, 32, 6)

    # # x就是btach_size的数据，处理完成之后直接就丢到模型里面进行训练
    # for idx in range(trainLen):
    #     u = data[idx * batch_size: batch_size * (idx + 1)]
    #     # print(u.shape)  # (8, 32, 1)
    #     print(np.dot(u, Win).shape)  # (8, 32, 1) dot (1, 6) --> (8, 32, 6)
    #     print(np.dot(x, W).shape)  # (8, 32, 6) dot (6, 6) --> (8, 32, 6)
    #     # x = (1 - a) * x + a * np.tanh(np.dot(u, Win) + + np.dot(x, W))  # (8, 32, 6)
    #     x = (1 - a) * x + a * np.tanh(np.matmul(u, Win) + + np.matmul(x, W))  # (8, 32, 6)
    #     # print(np.matmul(u, Win).shape)  # (8, 32, 6)
    #     # print(np.matmul(x, W).shape)  # (8, 32, 6)
    #
    #     # print(np.matmul(u, Win) - np.dot(u, Win))  # (8, 32, 6)
    #     # print((np.matmul(x, W) - np.dot(x, W)))  # (8, 32, 6)
    #
    #     # print(x.shape)
    #     # if idx >= initLen:
    #     #     Xout[idx - initLen] = x  # 这里只有x，没有u和偏置, # 由于后面没有了Yout,因此不记录输入u



    th_bias = torch.from_numpy(bias)
    th_data = torch.from_numpy(data)
    th_Win = torch.from_numpy(Win)
    th_W = torch.from_numpy(W)
    th_Xout = torch.from_numpy(Xout)
    th_x = torch.from_numpy(x)


    for idx in range(trainLen):
        th_u = th_data[idx * batch_size: batch_size * (idx + 1)]
        print(torch.matmul(th_u, th_Win).size())  # (8, 32, 1) dot (1, 6) --> (8, 32, 6)  torch.Size([8, 32, 6])
        print(torch.matmul(th_x, th_W).size())  # (8, 32, 6) dot (6, 6) --> (8, 32, 6)  torch.Size([8, 32, 6])

        th_x = (1 - a) * th_x + a * torch.tanh(np.matmul(th_u, th_Win) + + torch.matmul(th_x, th_W))  # (8, 32, 6)
        print(th_x.size())  # torch.Size([8, 32, 6])


    # print(Xout.shape)  # (297, 32, 6) (samples, inSize, resSize)
    # x0 = np.zeros((inSize, resSize))
    # xout = np.dot(x0, W)
    # print(xout.shape) # (32, 6) (inSize, resSize)
    # u = data[0]
    # uin = np.dot(u, Win)
    # print(uin.shape) # (32, 6) (inSize, resSize)

    # a = np.random.randn(2,3)
    # print(a)
    # a = np.expand_dims(a, 0).repeat(3, axis=0)
    # print(a)

