#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/15:15:34
@File:           os_test.py
==================================================
"""
def exp_dir(log_dir):
    num_exp_list = []
    if not os.path.exists(log_dir + '/exp1'):
        os.makedirs(log_dir + '/exp1')
    else:
        for exp in os.listdir(log_dir):
            if 'exp' in exp:
                num_exp_list.append(int(exp[3:]))
        log_dir = log_dir + '/exp' + str(max(num_exp_list) + 1)
        # print(log_dir)
        os.makedirs(log_dir)
    return log_dir


# log_dir = exp_dir(log_dir)


if __name__ == "__main__":
    import os
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    log_dir = './runs'


    num_exp_list = []
    if not os.path.exists(log_dir + '/exp1'):
        os.makedirs(log_dir + '/exp1')
    else:
        for exp in os.listdir(log_dir):
            if 'exp' in exp:
                num_exp_list.append(int(exp[3:]))
        log_dir = log_dir + '/exp' + str(max(num_exp_list) + 1)
        print(log_dir)
        os.makedirs(log_dir)

    print(os.listdir('./runs'))
    writer = SummaryWriter(log_dir)  # 放在if __name__ == '__main__':之下，writer才不会自动添加1，exp1+++


    for i in range(5):
        for n_iter in range(100):
            writer.add_scalar('Loss/train', np.random.random(), n_iter)
            writer.add_scalar('Loss/test_func', np.random.random(), n_iter)
            writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            writer.add_scalar('Accuracy/test_func', np.random.random(), n_iter)

        writer.add_scalar('Accuracy/test_func', np.random.random(), n_iter)


