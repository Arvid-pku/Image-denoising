import numpy as np
import cv2


def bgr_to_yuv(img):
    """
    将BGR矩阵转换成YUV
    tips: cv2.imread()返回值按BGR顺序
    """
    # M = np.matrix([[65.738, 129.057, 25.064],
    #                [-37.945, -74.494, 112.439],
    #                [112.439, -94.154, -18.285]]) / 256
    M = np.matrix([[25.064, 129.057, 65.738],
                   [112.439, -74.494, -37.945],
                   [-18.285, -94.154, 112.439]]) / 256
    [H, W, _] = img.shape
    _img = np.matrix(img.reshape(H * W, 3))  # 转换成二维矩阵，利用矩阵运算加快程序运行
    _img = _img * M.T
    bias = np.matrix([16, 128, 128])
    bias = np.tile(bias, H * W).reshape(H * W, 3)  # 复制扩充成二维
    _img += bias
    _img = np.array(_img).reshape(H, W, 3)
    return _img


def yuv_to_bgr(img):
    """
    将YUV矩阵转换成BGR
    tips: cv2.imread()返回值按BGR顺序
    """
    M = np.matrix([[25.064, 129.057, 65.738],
                   [112.439, -74.494, -37.945],
                   [-18.285, -94.154, 112.439]]) / 256
    M = M.I
    [H, W, _] = img.shape
    _img = np.matrix(img.reshape(H * W, 3))  # 转换成二维矩阵，利用矩阵运算加快程序运行
    bias = np.matrix([16, 128, 128])
    bias = np.tile(bias, H * W).reshape(H * W, 3)  # 复制扩充成二维
    _img -= bias
    _img = _img * M.T
    _img = np.array(_img).reshape(H, W, 3)
    return _img


def rgb_to_yuv(img):
    """
    将BGR矩阵转换成YUV
    tips: matplotlib.pyplot.imread()返回值是按RGB顺序,但值介于0~1
    """
    M = np.matrix([[65.738, 129.057, 25.064],
                   [-37.945, -74.494, 112.439],
                   [112.439, -94.154, -18.285]]) / 256
    [H, W, _] = img.shape
    _img = np.matrix(img.reshape(H * W, 3))  # 转换成二维矩阵，利用矩阵运算加快程序运行
    _img = _img * M.T
    bias = np.matrix([16, 128, 128])
    bias = np.tile(bias, H * W).reshape(H * W, 3)  # 复制扩充成二维
    _img += bias
    _img = np.array(_img).reshape(H, W, 3)
    return _img


def yuv_to_rgb(img):
    """
    将BGR矩阵转换成YUV
    tips: matplotlib.pyplot.imread()返回值是按RGB顺序,但值介于0~1
    """
    M = np.matrix([[65.738, 129.057, 25.064],
                   [-37.945, -74.494, 112.439],
                   [112.439, -94.154, -18.285]]) / 256
    M = M.I
    [H, W, _] = img.shape
    _img = np.matrix(img.reshape(H * W, 3))  # 转换成二维矩阵，利用矩阵运算加快程序运行
    bias = np.matrix([16, 128, 128])
    bias = np.tile(bias, H * W).reshape(H * W, 3)  # 复制扩充成二维
    _img -= bias
    _img = _img * M.T
    _img = np.array(_img).reshape(H, W, 3)
    return _img
