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

import time
if __name__ == "__main__":
    import numpy as np

    img = np.random.randn(5, 3)
    XOptical = np.random.randn(5, 3, 2)
    dummy_dataset = dict()
    dummy_dataset['img'] = img
    dummy_dataset['XOptical'] = XOptical
    print(dummy_dataset)
    import pickle
    with open('./dummy_dataset.pkl', mode="wb") as save_file:
        pickle.dump(dummy_dataset, save_file)

    with open('./dummy_dataset.pkl', mode="rb") as load_file:
        dummy_dataset = pickle.load(load_file)

    print(dummy_dataset)
    print(len(dummy_dataset['img']))

    # st = time.time()
    # a = np.random.randn(6, 32*32*3, 50000)
    #
    # print(time.time() - st)
