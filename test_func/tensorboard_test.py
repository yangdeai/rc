#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/15:11:36
@File:           tensorboard_test.py
==================================================
"""


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    # writer = SummaryWriter()
    # for n_iter in range(100):
    #     writer.add_scalar('Loss/train', np.random.random(), n_iter)
    #     writer.add_scalar('Loss/test_func', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/test_func', np.random.random(), n_iter)


    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
        # img_batch[i, 2] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i

    writer = SummaryWriter()
    writer.add_images('my_image_batch', img_batch, 0)
    writer.close()
