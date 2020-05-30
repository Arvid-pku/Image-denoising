import numpy as np
import cv2
import time
from math import exp, floor
from psnr import psnr

N1 = 9  # 块大小N1*N1
N2 = 15  # 搜索窗口大小N2*N2
sigma = 0
h = 1


def weight(x1, y1, x2, y2):
    """输入是两个块的左上角坐标"""
    m1 = _n_img[x1: x1 + N1, y1: y1 + N1, :]
    m2 = _n_img[x2: x2 + N1, y2: y2 + N1, :]
    _flag1 = flag[x1: x1 + N1, y1: y1 + N1]
    _flag2 = flag[x2: x2 + N1, y2: y2 + N1]
    flag_ = _flag1 * _flag2  # 两个块中像素都不为零的位置
    flag_ = flag_.repeat(3).reshape(N1, N1, 3)
    e2 = ((m1 - m2) * flag_) ** 2
    distance2 = np.average(e2[e2 >= 0])  # e2[e2 > 0]为空时，函数返回nan
    if np.isnan(distance2):
        return 0
    return exp(-max(distance2 - 2 * sigma * sigma, 0) / h)


if __name__ == '__main__':
    start = time.clock()
    # rawFile = 'Lena.png'
    # img = cv2.imread('org/' + rawFile)
    # noiseFile = 'Lena_RS1_0.3.png'
    # n_img = cv2.imread('test/' + noiseFile)

    img = cv2.imread('test_raw.png')
    noiseFile = 'test.png'
    n_img = cv2.imread(noiseFile)

    [H, W, _] = n_img.shape
    _n_img = np.zeros((H + N2 - N1, W + N2 - N1, 3))  # 给矩阵周围加空白像素，方便程序的编写
    blank = floor((N2 - N1) / 2)  # _n_img左边和上面的空白行数
    _n_img[blank: H + blank, blank: W + blank, :] = n_img
    flag = (_n_img[:, :, 0] + _n_img[:, :, 1] + _n_img[:, :, 2]) > 0  # 二维，像素不等于零的位置

    final_img = np.zeros_like(_n_img, float)  # 最终估计图像
    num_img = np.zeros((H + N2 - N1, W + N2 - N1, 3), float)  # 记录每个像素点被估计了几次
    for i in range(blank, H + blank - N1 + 1):
        print("\ri=%d" % i, end='')  # 打印程序运行进度
        for j in range(blank, W + blank - N1 + 1):  # (i,j)是中心块左上角坐标
            tem_weight = np.zeros((N1, N1), float)  # 记录权重
            tem_img = np.zeros((N1, N1, 3), float)  # 此次估计结果
            flag1 = (flag[i: i + N1, j: j + N1] == 0)  # 中心块像素等于零的位置
            for x in range(-blank, -blank + N2 - N1 + 1):
                for y in range(-blank, -blank + N2 - N1 + 1):  # (x,y)搜索块相对中心块的位置
                    flag2 = flag[i + x: i + N1 + x, j + y: j + N1 + y]
                    _flag = flag1 * flag2  # 中心块像素为零、搜索块像素不为零的位置
                    _weight = weight(i, j, i + x, j + y) * _flag
                    tem_weight += _weight
                    _weight = _weight.repeat(3).reshape(N1, N1, 3)  # 扩充成三维
                    tem_img += _n_img[i + x: i + N1 + x, j + y: j + N1 + y, :] * _weight

            tem_weight += (tem_weight == 0)  # 避免除以零
            tem_weight = tem_weight.repeat(3).reshape(N1, N1, 3)  # 扩充成三维
            final_img[i: i + N1, j: j + N1:, :] += tem_img / tem_weight
            num_img[i: i + N1, j: j + N1:, :] += tem_weight > 0

    num_img += (num_img == 0)  # 避免除以零
    final_img = final_img / num_img  # 每个像素点都进行了多次估计，直接平均
    final_img = final_img[blank: H + blank, blank: W + blank, :] + n_img

    outFileName = 'result/NLM/' + noiseFile[:-4] + '_' + str(N1) + '_' + str(N2) + '_' + str(sigma) + '_' + str(
        h) + noiseFile[-4:]
    cv2.imwrite(outFileName, final_img)
    final_img = cv2.imread(outFileName)
    print("\nP=%d B=%d sigma=%d h = %f" % (N1, N2, sigma, h), psnr(img, final_img))
    end = time.clock()
    print('time:%d' % (end - start))

# The results
# noiseFile = 'Lena_RS1_0.5.png'
# P=9 B=15 sigma=0 h = 1.000000 [33.802286078114356, 34.37252084511732, 37.22837467495257, 35.13439386606141] time:569
#
# noiseFile = 'Lena_RS1_0.3.png'
# P=15 B=21 sigma=0 h = 1.000000 [31.073116649651563, 31.0775574382444, 33.684435010928624, 31.945036366274863] time:785
