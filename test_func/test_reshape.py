#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
==================================================
@Project:        rc
@Author:         yang deai
@Time:           2023/5/31:17:36
@File:           test_reshape.py
==================================================
"""

if __name__ == "__main__":
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    a = np.arange(1, 13, step=1)
    b = np.arange(10, 130, step=10)
    c = np.arange(100, 1300, step=100)

    ap = a.reshape(3, 2, 2)
    bp = b.reshape(3, 2, 2)
    cp = c.reshape(3, 2, 2)


    # print(ap)
    # print(ap.transpose(1, 2, 0))
    # print(ap.reshape(2, 2, 3))

    ef = torch.rand(1, 2, 3) - 0.5
    ee = torch.reshape(ef, (1, 3*2, 1))
    print(ee)
    # at = torch.rand(1, 3)
    # at = torch.ones(1, 3)
    at = torch.tensor([1, 0, 0]).to(torch.float32).reshape(1, 3)
    # at = torch.tensor([0, 1, 0]).to(torch.float32).reshape(1, 3)
    # at = torch.tensor([0, 0, 1]).to(torch.float32).reshape(1, 3)
    print(at)
    mul = torch.matmul(ee, at)
    print(mul)
    print(mul.reshape(1, 2*3, 3))
    # print(mul)
    # print(torch.sum(ef, dim=1))
    #
    # print(torch.matmul(ef[0], at.t()[0].t()))
    #
    # et = torch.from_numpy(ap).transpose(0, 1).transpose(1, 2)
    # print(et)
    #
    # ettt = et.transpose(2, 1).transpose(1, 0)
    # print(ettt)
    #
    # print(ap)
    # print(bp)
    # print(cp)
    # d = np.vstack([ap, bp, cp])
    # print(d, d.shape)  # (9, 2, 2)
    # e = np.stack([ap, bp, cp], axis=0)
    # print(e, e.shape)  # (3, 3, 2, 2)
    #
    # f = e.reshape(-1, 3*2*2, 1)
    # print(f, f.shape)  # (3, 12, 1)
    #
    # g = f.reshape(-1, 3, 2, 2)
    # print(g, g.shape)  # (3, 3, 2, 2)







