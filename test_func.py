#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/5/22:18:20
@File:           test_func.py
==================================================
"""

def func():
    for i in range(10):
        yield i


import time
if __name__ == "__main__":
    import numpy as np
    import torch
    gen = func()
    for j in gen:
        print(j)
    # batch_size = 5
    # data_source = np.random.randn(10, 3, 4)
    # random_sampler = torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)
    # data_sampler = torch.utils.data.BatchSampler(sampler=random_sampler, batch_size=batch_size, drop_last=False)
    # print(list(data_sampler))
    # data_loader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, sampler=random_sampler)

    # for sam in data_loader:
    #     print(sam.size())

    # N = 20
    # samples_idx = np.random.permutation(N)
    # # print(samples_idx.shape)
    # for i in range(int(len(samples_idx) / batch_size)):
    #     batch_idx = np.random.choice(samples_idx, size=batch_size, replace=False)
    #     print(batch_idx)
    # img = np.random.randn(5, 3)
    # XOptical = np.random.randn(5, 3, 2)
    # dummy_dataset = dict()
    # dummy_dataset['img'] = img
    # dummy_dataset['XOptical'] = XOptical
    # print(dummy_dataset)
    # import pickle
    # with open('./dummy_dataset.pkl', mode="wb") as save_file:
    #     pickle.dump(dummy_dataset, save_file)
    #
    # with open('./dummy_dataset.pkl', mode="rb") as load_file:
    #     dummy_dataset = pickle.load(load_file)
    #
    # print(dummy_dataset)
    # print(len(dummy_dataset['img']))

    # st = time.time()
    # a = np.random.randn(6, 32*32*3, 50000)
    #
    # print(time.time() - st)
