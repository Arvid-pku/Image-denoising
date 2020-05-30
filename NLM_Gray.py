import numpy as np
import cv2
import time
from math import exp, floor
from psnr import psnr

N1 = 15  # 块大小N1*N1
N2 = 21  # 搜索窗口大小N2*N2
sigma = 0
h = 1


def weight(x1, y1, x2, y2):
    """输入是两个块的左上角坐标"""
    m1 = _n_img[x1: x1 + N1, y1: y1 + N1]
    m2 = _n_img[x2: x2 + N1, y2: y2 + N1]
    _flag1 = flag[x1: x1 + N1, y1: y1 + N1]
    _flag2 = flag[x2: x2 + N1, y2: y2 + N1]
    flag_ = _flag1 * _flag2  # 两个块中像素都不为零的位置
    e2 = ((m1 - m2) * flag_) ** 2
    distance2 = np.average(e2[e2 >= 0])  # e2[e2 > 0]为空时，函数返回nan
    if np.isnan(distance2):
        return 0
    return exp(-max(distance2 - 2 * sigma * sigma, 0) / h)


if __name__ == '__main__':
    start = time.clock()
    rawFile = 'barbara.png'
    img = cv2.imread('org/' + rawFile, cv2.IMREAD_GRAYSCALE)
    noiseFile = 'barbara_RS1_0.3.png'
    n_img = cv2.imread('test/' + noiseFile, cv2.IMREAD_GRAYSCALE)

    [H, W] = n_img.shape
    _n_img = np.zeros((H + N2 - N1, W + N2 - N1))  # 给矩阵周围加空白像素，方便程序的编写
    blank = floor((N2 - N1) / 2)  # _n_img左边和上面的空白行数
    _n_img[blank: H + blank, blank: W + blank] = n_img
    flag = _n_img > 0  # 二维，像素不等于零的位置

    final_img = np.zeros((H + N2 - N1, W + N2 - N1), float)  # 最终估计图像
    num_img = np.zeros((H + N2 - N1, W + N2 - N1), float)  # 记录每个像素点被估计了几次
    for i in range(blank, H + blank - N1 + 1):
        print("\ri=%d" % i, end='')  # 打印程序运行进度
        for j in range(blank, W + blank - N1 + 1):  # (i,j)是中心块左上角坐标
            tem_weight = np.zeros((N1, N1), float)  # 记录权重
            tem_img = np.zeros((N1, N1), float)  # 此次估计结果
            flag1 = (flag[i: i + N1, j: j + N1] == 0)  # 中心块像素等于零的位置
            for x in range(-blank, -blank + N2 - N1 + 1):
                for y in range(-blank, -blank + N2 - N1 + 1):  # (x,y)搜索块相对中心块的位置
                    flag2 = flag[i + x: i + N1 + x, j + y: j + N1 + y]
                    _flag = flag1 * flag2  # 中心块像素为零、搜索块像素不为零的位置
                    _weight = weight(i, j, i + x, j + y) * _flag
                    tem_weight += _weight
                    tem_img += _n_img[i + x: i + N1 + x, j + y: j + N1 + y] * _weight

            tem_weight += (tem_weight == 0)  # 避免除以零
            final_img[i: i + N1, j: j + N1] += tem_img / tem_weight
            num_img[i: i + N1, j: j + N1] += tem_weight > 0

    num_img += (num_img == 0)  # 避免除以零
    final_img = final_img / num_img  # 每个像素点都进行了多次估计，直接平均
    final_img = final_img[blank: H + blank, blank: W + blank] + n_img

    outFileName = 'result/NLM/' + noiseFile[:-4] + '_' + str(N1) + '_' + str(N2) + '_' + str(sigma) + '_' + str(
        h) + noiseFile[-4:]
    cv2.imwrite(outFileName, final_img)
    final_img = cv2.imread(outFileName, cv2.IMREAD_GRAYSCALE)
    print("\nN1=%d N2=%d sigma=%d h = %f" % (N1, N2, sigma, h), psnr(img, final_img))
    end = time.clock()
    print('total_time:%d' % (end - start))

# The results
# noiseFile = 'barbara_RS1_0.5.png'
# N1=9 N2=15 sigma=0 h = 1.000000 32.626355238295034 total_time:473
#
# noiseFile = 'barbara_RS1_0.3.png'
# N1=15 N2=21 sigma=0 h = 1.000000 29.081855904076388 total_time:519
