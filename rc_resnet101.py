# 导入常用包
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.optim.lr_scheduler import MultiStepLR

from utils import WarmUpLR, get_network, RcProject
from models.resnet import resnet101

import logging
import time
import argparse


def train(model=None, loss_fn=None, optimizer=None, lr=1e-1, device=None):
    model = model.to(device)
    # 训练时间
    start = time.time()
    # 保存模型文件的尾标
    save_num = 0

    val_last_acc = 0.0
    val_last_loss = np.Inf
    epoch_count = 0
    # scheduler_optimizer = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 80, 120], gamma=0.5,
    #                                                            last_epoch=-1)
    for epoch in range(MAX_EPOCH):
        train_total_loss = 0
        train_total_num = 0
        train_total_correct = 0
        if epoch_count / 10 == 1:
            epoch_count = 0
            lr = lr * 0.5
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        logging.info('Epoch {}/{}'.format(epoch, MAX_EPOCH))
        logging.info('-' * 10)
        logging.info("\n")

        # next epoch dataloader
        train_loader = train_rc.rc_reprocess()
        writer.add_histogram('train_rc.Win', train_rc.Win, epoch)
        with torch.set_grad_enabled(True):
            model.train()
            for iter, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # write the network graph at epoch 0, batch 0
                if epoch == 0 and iter == 0:
                    writer.add_graph(model, input_to_model=(images, labels)[0])

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                train_total_correct += (outputs.argmax(1) == labels).sum().item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_total_num += labels.shape[0]
                train_total_loss += loss.item()

            train_loss = train_total_loss / train_total_num
            train_acc = train_total_correct / train_total_num

            time_elapsed = time.time() - start
            logging.info("\n")
            logging.info('total time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logging.info('train Loss: {:.4f}[{}], train Acc: {:.4f}'.format(train_loss, epoch_count, train_acc))

        # writer the weights and bias distribution.
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)

        # next epoch dataloader
        test_loader = test_rc.rc_reprocess()
        writer.add_histogram('test_rc.Win', test_rc.Win, epoch)
        with torch.set_grad_enabled(False):
            model.eval()
            val_total_loss = 0
            val_total_correct = 0
            val_total_num = 0
            for iter, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_total_correct += (outputs.argmax(1) == labels).sum().item()
                val_total_loss += loss.item()
                val_total_num += labels.shape[0]

            val_loss = val_total_loss / val_total_num
            val_acc = val_total_correct / val_total_num

            if val_loss < val_last_loss:
                val_last_loss = val_loss
                val_last_acc = val_acc
                epoch_count = 0
                save_num = 0 if save_num > 1 else save_num  # 只保存最近2次的训练结果
                save_name_t = '{}_{}.pth'.format(best_weight_pth, save_num)
                torch.save(model.state_dict(), save_name_t)
                save_num += 1
            else:
                epoch_count += 1

            time_elapsed = time.time() - start
            logging.info("\n")
            logging.info('total time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logging.info('valid Loss: {:.4f}[{}], valid Acc: {:.4f}'.format(val_loss, epoch_count, val_acc))

        # scheduler_optimizer.step()
        logging.info("\n\n")
        logging.info(f"train_rc.Win: {train_rc.Win}, test_rc.Win: {test_rc.Win}, epoch{epoch}/{MAX_EPOCH}")
        logging.info("\n\n")

        logging.info("\n")
        logging.info('current lr: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        logging.info("\n")

        # writer
        writer.add_scalars('Loss', {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        }, epoch)
        writer.add_scalars('Acc', {
            'train_acc': train_acc * 100,
            'val_acc': val_acc * 100,
        }, epoch)

    # train end
    time_elapsed = time.time() - start
    logging.info("\n")
    logging.info('finished！')
    logging.info('total trained time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('the most high accuracy : {:4f}'.format(val_last_acc))
    save_num = save_num - 1
    save_num = save_num if save_num < 0 else 1
    save_name_t = '{}_{}.pth'.format(best_weight_pth, save_num)
    logging.info('the best acc model：{}'.format(save_name_t))
    torch.save(model.state_dict(), save_name_t)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train resnet with cifar10 dataset.')
    parser.add_argument('-net', '--network', type=str, default='resnet101', help='network: resnet101 or rc_resnet_101')
    parser.add_argument('-exp_num', '--exp_num', type=str, default='5', help='the exp num')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-me', '--max_epoch', type=int, default=200, help='total epoch to train')
    parser.add_argument('-we', '--warm_epoch', type=int, default=2, help='warm up training phase')
    args = parser.parse_args()

    # file/dir
    exp_name = f'rc_{args.network}_exp{args.exp_num}'
    weight_dir = f'/tf_logs/weights/{exp_name}'
    best_weight_pth = weight_dir + f'/max_epoch{args.max_epoch}'
    log_dir = f"/tf_logs/runs/train/{exp_name}"

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # log
    logging.basicConfig(filename=log_dir + '.txt',
                        filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO)

    # writer = SummaryWriter(log_dir)
    writer = SummaryWriter(log_dir)

    # seed
    torch.manual_seed(42)
    np.random.seed(42)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # hyper params
    LR = args.learning_rate
    MAX_EPOCH = args.max_epoch
    WARM_EPOCH = args.warm_epoch
    BATCH_SIZE = args.batch_size

    # prepared dataset
    train_dataset = datasets.CIFAR10('cifar10', train=True, download=True)
    test_dataset = datasets.CIFAR10('cifar10', train=False, download=True)

    # rc_projection and dataloader
    train_rc = RcProject(train_dataset, batch_size=BATCH_SIZE, train=True)
    test_rc = RcProject(test_dataset, batch_size=BATCH_SIZE, train=False)

    # train/test Win
    logging.info("\n\n")
    logging.info(f"train_rc.Win: {train_rc.Win}, test_rc.Win: {test_rc.Win}")
    logging.info("\n\n")

    # model
    model = get_network(args.network)
    # model's in_channel equals rc's out_channel
    rc_out_channel = 3 * train_rc.resSize
    model.conv1 = torch.nn.Conv2d(rc_out_channel, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = torch.nn.MaxPool2d(1, 1, 0)  # 通过1x1的池化核让池化层失效

    # check model
    logging.info("============== layers needed to train ==============")
    for name, params in model.named_parameters():
        if params.requires_grad:
            logging.info(name)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    logging.info("===   !!!START TRAINING!!!   ===")
    # logging.info('train_data_num: {}, validation_data_num: {}'.format(len(train_dataset), len(val_cifar_dataset)))
    logging.info('train_data_num: {}, validation_data_num: {}'.format(len(train_dataset), len(test_dataset)))
    train(model=model, loss_fn=loss_fn, optimizer=optimizer, lr=LR, device=device)
    logging.info("===   !!! END TRAINING !!!   ===")
    logging.info("\n\n\n\n")
