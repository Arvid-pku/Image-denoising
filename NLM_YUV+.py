import numpy as np
import cv2
import time
from math import exp, floor
from psnr import psnr
from YUV import bgr_to_yuv, yuv_to_bgr

N1 = 15  # 块大小N1*N1
N2 = 21  # 搜索窗口大小N2*N2
sigma = 0
h = 1
# Beta_Kaiser = 2.0  # 凯撒窗参数


# 仅用第一个通道来计算权重，YUN模式下，第一个通道就是Y
# def weight(x1, y1, x2, y2):
#     """输入是两个块的左上角坐标"""
#     m1 = _yuv_n_img[x1: x1 + N1, y1: y1 + N1, 0]
#     m2 = _yuv_n_img[x2: x2 + N1, y2: y2 + N1, 0]
#     _flag1 = flag[x1: x1 + N1, y1: y1 + N1]
#     _flag2 = flag[x2: x2 + N1, y2: y2 + N1]
#     flag_ = _flag1 * _flag2  # 两个块中像素都不为零的位置
#     e2 = ((m1 - m2) * flag_) ** 2
#     distance2 = np.average(e2[e2 >= 0])  # e2[e2 > 0]为空时，函数返回nan
#     if np.isnan(distance2):
#         return 0
#     return exp(-max(distance2 - 2 * sigma * sigma, 0) / h)


def weight(x1, y1, x2, y2):
    """输入是两个块的左上角坐标"""
    m1 = _yuv_n_img[x1: x1 + N1, y1: y1 + N1, :]
    m2 = _yuv_n_img[x2: x2 + N1, y2: y2 + N1, :]
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
    rawFile = 'Lena.png'
    img = cv2.imread('org/' + rawFile)
    noiseFile = 'Lena_RS1_0.3.png'
    n_img = cv2.imread('test/' + noiseFile)

    # img = cv2.imread('test_raw.png')
    # noiseFile = 'test.png'
    # n_img = cv2.imread(noiseFile)

    yuv_n_img = bgr_to_yuv(n_img)

    [H, W, _] = yuv_n_img.shape
    _yuv_n_img = np.zeros((H + N2 - N1, W + N2 - N1, 3))  # 给矩阵周围加空白像素，方便程序的编写
    blank = floor((N2 - N1) / 2)  # _yuv_n_img左边和上面的空白行数
    _yuv_n_img[blank: H + blank, blank: W + blank, :] = yuv_n_img

    bias = np.array([16, 128, 128])
    bias = np.tile(bias, H * W).reshape(H, W, 3)  # 复制扩充成三维
    f = yuv_n_img != bias  # 像素点rgb都等于零对应YCbCr等于(16, 128, 128)
    f = (f[:, :, 0] + f[:, :, 1] + f[:, :, 2]) > 0  # 二维，RGB模式中像素不等于零的位置
    flag = np.zeros((H + N2 - N1, W + N2 - N1))  # 给矩阵周围加空白像素，与_yuv_n_img保持一致
    flag[blank: H + blank, blank: W + blank] = f

    # 加凯撒窗
    # K = np.matrix(np.kaiser(N1, Beta_Kaiser))
    # Kaiser = np.array(K.T * K)  # 构造一个凯撒窗
    # Kaiser = Kaiser.repeat(3).reshape(N1, N1, 3)  # 扩充成三维

    final_img = np.zeros((H + N2 - N1, W + N2 - N1, 3), float)  # 最终估计图像
    weight_img = np.zeros((H + N2 - N1, W + N2 - N1, 3), float)  # 权重
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
                    tem_img += _yuv_n_img[i + x: i + N1 + x, j + y: j + N1 + y, :] * _weight

            tem_weight += (tem_weight == 0)  # 避免除以零
            tem_weight = tem_weight.repeat(3).reshape(N1, N1, 3)  # 扩充成三维

            final_img[i: i + N1, j: j + N1:, :] += tem_img / tem_weight
            weight_img[i: i + N1, j: j + N1:, :] += tem_weight > 0  # 记录每个像素点被估计的次数，最后平均
            # 用凯撒窗的代码，对应上面两行
            # final_img[i: i + N1, j: j + N1:, :] += tem_img / tem_weight * Kaiser
            # weight_img[i: i + N1, j: j + N1:, :] += Kaiser

    weight_img += (weight_img == 0)  # 避免除以零
    final_img = final_img / weight_img  # 每个像素点都进行了多次估计，直接平均
    final_img = final_img[blank: H + blank, blank: W + blank, :]
    _bias = np.repeat(f, 3).reshape(H, W, 3) * bias  # 像素点rgb都等于零对应YCbCr等于(16, 128, 128),修正final_img
    final_img = yuv_to_bgr(final_img + _bias) + n_img

    outFileName = 'result/NLM/' + noiseFile[:-4] + '_' + str(N1) + '_' + str(N2) + '_' + str(sigma) + '_' + str(
        h) + noiseFile[-4:]
    cv2.imwrite(outFileName, final_img)
    final_img = cv2.imread(outFileName)
    print("\nP=%d B=%d sigma=%d h = %f" % (N1, N2, sigma, h), psnr(img, final_img))
    end = time.clock()
    print('time:%d' % (end - start))

# The results
# noiseFile = 'Lena_RS1_0.5.png' 用YUV三个通道计算权重,无凯撒窗
# P=9 B=15 sigma=0 h = 1.000000 [33.93627406850448, 34.510636345073365, 37.35516440584259, 35.26735827314015] time:675
#
# noiseFile = 'Lena_RS1_0.3.png'
# P=15 B=21 sigma=0 h = 1.000000 [31.195591935621792, 31.183651976211202, 33.775171616441405, 32.05147184275813]time:685
#
# noiseFile = 'test.png'(150*150取自Lena.png)
# 用RGB计算权重
# P=9 B=15 sigma=0 h = 1.000000 [31.411476278247996, 32.27519874157173, 35.13957722424206, 32.942084081353926]
# 仅Y通道计算权重
# P=9 B=15 sigma=0 h = 1.000000 [31.325063582425457, 32.3117315982166, 35.16094564206194, 32.93258027423467]
# 用YUV三个通道计算权重
# P=9 B=15 sigma=0 h = 1.000000 [31.55947136933814, 32.431314146886955, 35.29763047481411, 33.09613866367974]
# 用YUV三个通道计算权重，并加凯撒窗
# Beta_Kaiser = 2.0
# P=9 B=15 sigma=0 h = 1.000000 [31.527567718939785, 32.394265464283734, 35.271311885481104, 33.06438168956821]
#  Beta_Kaiser = 1.8
# P=15 B=21 sigma=0 h = 1.000000 [31.141995408220758, 32.07005833561392, 34.90664414326643, 32.70623262903371]
# Beta_Kaiser = 2.2
# P=15 B=21 sigma=0 h = 1.000000 [31.129695651452195, 32.055986493199406, 34.88969004183986, 32.69179072883049]
