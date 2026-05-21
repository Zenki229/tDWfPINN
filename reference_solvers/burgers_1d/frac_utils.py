import numpy as np
def frac_BDF_coeff(k, al):
    """
    此函数计算分数阶BDF系数
    p: 近似阶数, p=1,2,...,6
    k: 计算系数直到第k项
    al: 分数阶
    """
    c = np.zeros(k)
    c[0] = 1
    for j in range(k - 1):
        c[j + 1] = -c[j] * (al - j) / (j + 1)
    return c[:]
