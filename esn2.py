#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        esn1
@Author:         yang deai
@Time:           2023/4/20:14:42
@File:           esn.py
==================================================
思考：有没有想过做一下储备池的叠加？预测更长的序列时，而不是扩大矩阵的规模，或者说除了使用岭回归，可以使用决策树，随机森林

"""
import numpy as np
from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import matplotlib.pyplot as plt

# 加载数据
# 前2000个数据用来训练，2001-4000的数据用来测试。训练数据中，前100项用来初始化储备池，以让储备池中形成良好的回声之后再开始训练。
trainLen = 2000
testLen = 2000
initLen = 100  # 前100项用来初始化储备池

data = loadtxt('MackeyGlass_t17.txt')  # (10000,)一万条数据
# print(data.shape)  # (10000,)  -- (50000, 3072)

# 绘制前1000条数据
plt.figure(0).clear()
plot(data[0:1000])
title('A sample of data')

# 生成ESN储层
inSize = outSize = 1  # inSize 输入维数 K  # u(n)是一个输入序列，每个输入u(i)的维度是K，这里设置为=1；y(n)是一个输出序列，每个输出y(i)的维度L也是1
resSize = 1000  # 储备池规模 N  # 储备池规模是储备池神经元个数
a = 0.3  # 可以看作储备池更新的速度，可不加，即设为1.  # soft update, a=1的就是hard update.

random.seed(42)  # 设置随机种子
# 随机初始化 Win 和 W    输入权重Win是输入n(i)和储备池的连接权重,shape=(N,1+K)，dot(Win,u(n)),shape=(N,1+K)X(1+K,), W是储备池神经元连接矩阵，shape=(N,N)
Win = (random.rand(resSize, 1 + inSize) - 0.5) * 1  # 输入矩阵 N * 1+K  (1000, 2)
# print(Win.shape)  # (1000, 2)
W = random.rand(resSize, resSize) - 0.5  # 储备池连接矩阵 N * N (1000, 1000)，???内部权重矩阵W是某时刻神经元与下一时刻神经元的连接，而非普通的互相连接???
# print(W.shape)  # (1000, 1000)

# 对W进行防缩，以满足稀疏的要求。
# 方案 1 - 直接缩放 (快且有脏数据, 特定储层): W *= 0.135
# 方案 2 - 归一化并设置谱半径 (正确, 慢):
print('计算谱半径...')
rhoW = max(abs(linalg.eig(W)[0]))  # linalg.eig(W)[0]:特征值   linalg.eig(W)[1]:特征向量
print("rhoW = ", rhoW)
# w, v = linalg.eig(W)  # (1000,) (1000, 1000)
# print(w.shape, v.shape)  # (1000,) (1000, 1000)

W *= 0.9 / rhoW  # 归一化的方式：除以最大特征的绝对值，乘以0.9 spectral radius
# 为设计（收集状态）矩阵分配内存  状态矩阵表明储备池状态随时间变化的过程：当前状态x(i)是当前输入dot(Win,u(i))、上一次状态dot(W,x(i-1))和偏置v的函数，这里没有上一次输出反馈yout(i-1)
X = zeros((1 + inSize + resSize, trainLen - initLen))  # 储备池的状态矩阵x(t)：每一列是每个时刻的储备池状态。后面会转置
# 直接设置相应的目标矩阵  target
Yt = data[None, initLen + 1:trainLen + 1]  # 输出矩阵:每一行是一个时刻的输出  这是标签。下一时刻data[t+1]就是当前时刻data[t]的标签，target，为什么initLen + 1加1的原因
# print(Yt.shape)  # (1, 1900)

# 输入所有的训练数据，然后得到每一时刻的输入值u和储备池状态x。   !!!这一步获得了储备池的状态X
x = zeros((resSize, 1))  # (1000, 1) x是状态矩阵X的一个元素，表示储备池所有神经元(N)在当前时刻的状态
for t in range(trainLen):
    u = data[t]  # 标量， vstack((1, u)): (2,1)  x:(1000,1)  ### 储备池迭代方程 ###
    x = (1 - a) * x + a * tanh(dot(Win, vstack((1, u))) + dot(W, x))  # vstack((1, u)):将偏置量1加入输入序列  tanh激活函数
    if t >= initLen:  # 空转100次后，开始记录储备池状态     从某一时刻开始记录储备的状态，这里的某一时刻是100
        X[:, t - initLen] = vstack((1, u, x))[:, 0]  # 状态矩阵记录了偏置v、输入u和储备池状态x
        # print(vstack((1, u, x)).shape)  # (1002, 1) 1002 = 1 + inSize + resSize

# 使用Wout根据输入值u和储备池状态x去拟合目标值，这是一个简单的线性回归问题，这里使用的是岭回归(Ridge Regression)。
reg = 1e-8  # 正则化系数
X_T = X.T  # 转置之后dim[0]对应的是每一个时刻，dim[1]对应的是每个储备池神经元

"""
这里直接通过求解逆矩阵或者伪逆矩阵的方式求解Wout,
这里有一个和通过反向传播梯度下降的方式优化不同的一个地方：储备池计算没有模型输出Yout,直接使用标签Yt来求逆矩阵，所以没有优化过程，直接一步到位求解。
"""
#                     Yt == Yout = WoutX + alpha||Wout||^2  要求Yout - Yt = 0，直接使用Yout==Yt
# Wout:  1 * 1+K+N     Wout = [Yt,XT]X[(X,XT),alphaI]_-1，  到这里可以看到输入权重矩阵Win、储备池连接矩阵W（没有反馈连接矩阵）都是随机生成的，偏置(噪声)是固定的1，只有输出矩阵Wout是计算得到
Wout = dot(dot(Yt, X_T), linalg.inv(dot(X, X_T) + reg * eye(1 + inSize + resSize)))  # linalg.inv矩阵求逆；numpy.eye()生成对角矩阵，规模:1+inSize+resSize，默认对角线全1，其余全0
# Wout = dot( Yt, linalg.pinv(X) )

# 使用训练数据进行前向处理得到结果
# run the trained ESN in a generative mode. no need to initialize here,
# because x is initialized with training data and we continue from there. 也就是使用训练数据得到的状态作为测试阶段的储备池的初始化状态，那如果使用的是随机生成的x，就得使用预测模式？
Y = zeros((outSize, testLen))
u = data[trainLen]  # 测试数据最开始输入到储备池中，还是data里面的，生成模式下就使用储备池的输出作为下一时刻的输入
for t in range(testLen):
    x = (1 - a) * x + a * tanh(dot(Win, vstack((1, u))) + dot(W, x))  # (激活函数为tanh)当前时刻t，储备池的状态x，储备池状态迭代方程，最初的x是在训练阶段初始化得到的。
    y = dot(Wout, vstack((1, u, x)))  # (输出层的激活函数为identity)预测阶段: yout = Wout*[1,u,x]    输出矩阵(1 * 1+K+N)*此刻状态矩阵(1+K+N * 1)=此刻预测值
    # print(y.shape) # (1, 1)
    Y[:, t] = y  # t时刻的预测值   Y: 1 * testLen
    # 生成模型    这里使用生成的y，作为输入
    u = y  # 这里使用当前储备池输出y，作为下一次模型的输入，称为为生成模式，generative mode
    # 预测模型    这里还是使用原来的data作为输入，称为预测模式，predict mode
    # u = data[trainLen+t+1]



# 计算第一个errorLen时间步长的MSE
errorLen = 500  # data[trainLen + 1:trainLen + errorLen + 1]这是标签，Y[0, 0: errorLen]这是得到的输出矩阵的储备池生成的或者预测的
mse = sum(square(data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0: errorLen])) / errorLen
print('MSE = {0}'.format(str(mse)))

# 绘制测试集的真实数据和预测数据
figure(1).clear()
plot(data[trainLen + 1:trainLen + testLen + 1], 'g')
plot(Y.T, 'b')
title('Target and generated signals $y(n)$ starting at $n=0$')
legend(['Target signal', 'Free-running predicted signal'])

# 绘制储备池中前200个时刻状态(x(t))的前20个储层结点值
figure(2).clear()
plot(X[0:20, 0:200].T)
title('Some reservoir activations $\mathbf{x}(n)$')

# 绘制输出矩阵
figure(3).clear()
# bar(np.arange(1 + inSize + resSize), Wout.T, 8)
plot(np.arange(1 + inSize + resSize), Wout.T)

title('Output weights $\mathbf{W}^{out}$')
show()