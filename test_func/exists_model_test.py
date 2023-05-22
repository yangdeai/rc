#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/15:16:05
@File:           exists_model_test.py
==================================================
"""

if __name__ == "__main__":
    # 定义模型
    # 方法一：预训练模型
    import torchvision

    Resnet50 = torchvision.models.resnet50(pretrained=True)
    Resnet50.fc.out_features = 10
    print(Resnet50)
