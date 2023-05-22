#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rcProject
@Author:         yang deai
@Time:           2023/5/16:8:58
@File:           logging_test.py
==================================================
"""
import logging
import os


exp_name = 'exp0'
weight_dir = './weights'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

log_dir = f'./runs/train/{exp_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=log_dir + '.txt',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO)



if __name__ == "__main__":
    # # 训练&验证
    # log_dir = f'./runs/train/exp44'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # # log_file_name = 'log.txt'
    # # for exp_name in os.path.split(log_dir):
    # #     if 'exp' in exp_name:
    # #         log_file_name = exp_name
    # #         print(log_file_name)
    #
    # logging.basicConfig(filename=log_dir + '.txt',
    #                     filemode='w',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
    #                     level=logging.INFO)
    # logging.info("logging info test_func!")
    # # logger = logging.getLogger(name="log.txt")

    logging.info("===   !!!START TRAINING!!!   ===")
    logging.info("train data num:{}".format(len([124, 12])))

    logging.info("===   !!! END TRAINING !!!   ===")
    logging.info("\n\n\n\n")

