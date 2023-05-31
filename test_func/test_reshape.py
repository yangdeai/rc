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
    a = np.arange(1, 13, step=1)
    b = np.arange(10, 130, step=10)
    c = np.arange(100, 1300, step=100)

    ap = a.reshape(3, 2, 2)
    bp = b.reshape(3, 2, 2)
    cp = c.reshape(3, 2, 2)

    print(ap)
    print(bp)
    print(cp)
    d = np.vstack([ap, bp, cp])
    print(d, d.shape)  # (9, 2, 2)
    e = np.stack([ap, bp, cp], axis=0)
    print(e, e.shape)  # (3, 3, 2, 2)

    f = e.reshape(-1, 3*2*2, 1)
    print(f, f.shape)  # (3, 12, 1)

    g = f.reshape(-1, 3, 2, 2)
    print(g, g.shape)  # (3, 3, 2, 2)







