#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/6/2:15:50
@File:           test_transform.py
==================================================
"""

if __name__ == "__main__":
    import numpy as np

    import torch
    train_mean, train_std = (0.49144, 0.48222, 0.44652), (0.24702, 0.24349, 0.26166)

    from torchvision.transforms import transforms



    bs = 5
    out_channel = 6
    rc_data = torch.randn((bs, 32, 32, out_channel)).numpy()  # 经过储备池后的数据
    means = [0 for _ in range(out_channel)]
    stds = [0 for _ in range(out_channel)]  # 初始化均值和方差

    print(len(rc_data))  # bs=5

    for sample_idx in range(len(rc_data)):
        for channel_idx in range(out_channel):
            means[channel_idx] += np.mean(rc_data[sample_idx][:, :, channel_idx])
            stds[channel_idx] += np.std(rc_data[sample_idx][:, :, channel_idx])

    mean = np.array(means) / len(rc_data)
    std = np.array(stds) / len(rc_data)

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean, std)
    ])

    b = []
    for i in range(bs):
        print(rc_data[i].shape)  # (32, 32, 6)
        a_trans = train_data_transform(rc_data[i])
        print(a_trans.size())
        print(a_trans)
        a_trans = a_trans.unsqueeze(0)
        print(a_trans)
        print(a_trans.size())  # torch.Size([1, 6, 32, 32])
        b.append(a_trans)

    c = torch.vstack(b)
    print(c.shape)  # torch.Size([5, 6, 32, 32])

    # from torchvision.transforms import ToTensor  # 用于把图片转化为张量
    # import numpy as np  # 用于将张量转化为数组，进行除法
    # from torchvision.datasets import ImageFolder  # 用于导入图片数据集
    #
    # means = [0, 0, 0]
    # std = [0, 0, 0]  # 初始化均值和方差
    # transform = ToTensor()  # 可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
    # dataset = ImageFolder("./data/train/", transform=transform)  # 导入数据集的图片，并且转化为张量
    # num_imgs = len(dataset)  # 获取数据集的图片数量
    # for img, a in dataset:  # 遍历数据集的张量和标签
    #     for i in range(3):  # 遍历图片的RGB三通道
    #         # 计算每一个通道的均值和标准差
    #         means[i] += img[i, :, :].mean()
    #         std[i] += img[i, :, :].std()
    # mean = np.array(means) / num_imgs
    # std = np.array(std) / num_imgs  # 要使数据集归一化，均值和方差需除以总图片数量
    # print(mean, std)  # 打印出结果


