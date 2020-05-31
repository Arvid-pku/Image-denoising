import numpy as np
from math import log


def psnr(A, B, row=0, col=0):
    """
    参数A,B是两个图像的像素矩阵
    计算psnr时不考虑最上/下边的row行和最左/右边的col列
    返回三个通道各自的psnr(对于彩色图)以及平均psnr
    """
    A = A.astype(float)
    B = B.astype(float)
    try:
        [n, m, ch] = A.shape
    except:  # 对于灰度图，输入的可能是两个二维数组
        [n, m] = A.shape
        ch = 1

    if ch == 1:
        e = A - B
        e = e[row: n - row, col: m - col]
        me = np.average(e * e)
        s = 10 * log(255 * 255 / me) / log(10)
        return s
    else:
        e = A - B
        e = e[row: n - row, col: m - col, :]
        e1 = e[:, :, 0]
        e2 = e[:, :, 1]
        e3 = e[:, :, 2]
        me1 = np.average(e1 * e1)
        me2 = np.average(e2 * e2)
        me3 = np.average(e3 * e3)
        s1 = 10 * log(255 * 255 / me1) / log(10)
        s2 = 10 * log(255 * 255 / me2) / log(10)
        s3 = 10 * log(255 * 255 / me3) / log(10)
        return [s1, s2, s3, (s1 + s2 + s3) / 3]
