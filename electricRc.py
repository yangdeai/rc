#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/5/22:14:46
@File:           electricRc.py
==================================================
"""
import torch.nn as nn
import torch
import numpy as np


if __name__ == "__main__":
    # 数据维度 (BATCH_SIZE, in_channel, High, Width)
    BATCH_SIZE = 5
    in_channel = 3
    high = 3
    width = 3
    resSize = 6

    # 图像数据
    img_data = np.random.randn(BATCH_SIZE, in_channel, high, width)
    # 把图像看成一个个像素
    pixel_data = img_data.reshape(BATCH_SIZE, in_channel * high * width, 1)  # (BATCH_SIZE, in_channel * high * width, 1) (5, 27, 1)
    # 输入权重Win
    Win = np.random.randn(1, resSize)  # (1, resSize)  #TODO：根据光的调制确定电的输入权重Win
    # 输入调制：输入像素 与 输入权重 相乘
    ut_win = np.matmul(pixel_data, Win)  # matmul[(BATCH_SIZE, in_channel * high * width, 1), (1, resSize)]
    print(ut_win.shape)  # (BATCH_SIZE, in_channel * high * width, resSize) (5, 27, 6)
    # 储备池连接权重
    Wres = np.random.randn(resSize, resSize)  # (resSize, resSize)
    # 储备池输出： 调制后的输入 与 储备池连接权重 相乘  # TODO: X_pre是光储备池输出X的预测，当X_pred == X时，即可认为点的Wres与光储备池的连接权重相等
    X_pred = np.matmul(ut_win, Wres)  # matmul[(BATCH_SIZE, in_channel * high * width, resSize), (resSize, resSize)]
    print(X_pred.shape)  # (BATCH_SIZE, in_channel * high * width, resSize) (5, 27, 6)





    # input = torch.randn(5, 1, 3*3*3)
    # X_optical = torch.randn(5, 6, 3*3*3)
    # input_layer = nn.Conv1d(1, 6, kernel_size=1, stride=1, bias=False)  # 这里看是 (BATCH_SIZE, 1, in_channel * high * width)
    # input_p = input_layer(input)
    # print(input_p.size())  # torch.Size([5, 6, 27])
    #
    # reservoir_layer = nn.Conv1d(6, 6, kernel_size=1, stride=1, bias=False)
    # output = reservoir_layer(input_p)
    # print(output.size())  # torch.Size([5, 6, 27])
    # for name, weight in reservoir_layer.named_parameters():
    #     print(name, weight.size())
    #
    # optim_wdecay = torch.optim.SGD(reservoir_layer.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-2)
    #
    # loss_fn = torch.nn.MSELoss(reduction='mean')
    # loss = loss_fn(output, X_optical)
    # print(loss.item())
    #
    # # m = nn.Conv1d(16, 33, 3, stride=2)
    # # for name, weight in m.named_parameters():
    # #     print(name, weight.size())
    # # input = torch.randn(20, 16, 50)
    # # output = m(input)
    #
    # # print(output.size())


