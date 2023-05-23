#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/5/22:16:52
@File:           reservoir_66.py
==================================================
"""

import torch.nn as nn


# def get_model(resSize=6):
#     # RNN
#
#     # LSTM
#
#     # GRU
#
#     return model




if __name__ == "__main__":
    # model = get_model()
    # for name, weight in model.named_parameters():
    #     print(name, weight.size())
    #
    # model = get_model(Win=True)
    # for name, weight in model.named_parameters():
    #     print(name, weight.size())

    import numpy as np

    same_dim = 10
    output_dim = 16
    img = np.random.randn(5, 6, 1)
    # Win = np.random.randn(5, 1, 6)
    Win = np.random.randn(1, same_dim)
    modulated_input = np.matmul(img, Win)  # 进行广播
    print(modulated_input.shape)  # (5, 6, 6)

    Wres = np.random.randn(same_dim, 6)
    X_output = np.matmul(modulated_input, Wres)

    print(X_output.shape)
