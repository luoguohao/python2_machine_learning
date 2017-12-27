#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)  # 取出第k个左奇异向量
        vk = v[k].reshape(1, n)  # 取出第k个又奇异向量
        a += sigma[k] * np.dot(uk, vk)  # 取前k个奇异值计算
    a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


def restore2(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K + 1):
        for i in range(m):
            a[i] += sigma[k] * u[i][k] * v[k]
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')


if __name__ == "__main__":

    # broadcast
    # a = np.arange(0, 11)
    # b = np.arange(0, 11)
    #
    # print 'a:', a, 'b:', b
    # print 'a.shape:', a.shape, 'b.shape:', b.shape
    # a = a.reshape(11, 1)
    # b = b.reshape(1, 11)
    # print 'a:', a, 'b:', b
    # c = np.dot(a, b)
    # print c
    # print c[:, 2]
    #
    # arrays = [np.random.randn(2, 3) for _ in range(4)]
    # print arrays
    #
    # axis_0_array = np.stack(arrays, axis=0)  # 将给定array合并起来，组成新的维度，如将2x3的4个数据构成的数组，合并成4x2x3的三维数据
    # print 'axis_0_array shape:', axis_0_array.shape
    # print axis_0_array
    #
    # axis_1_array = np.stack(arrays, axis=1)  # 将给定array合并起来组成新的维度，如将2x3的2个数据构成的数组，合并成2x4x3的三维数据
    # print 'axis_1_array shape:', axis_1_array.shape
    # print axis_1_array
    # exit(0)

    A = Image.open("6.son.png", 'r')
    output_path = r'.\Pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print 'A:', A
    a = np.array(A)
    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
    plt.figure(figsize=(10, 10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    print 'u_r:', u_r.shape
    print 'sigma_r:', sigma_g.shape
    print 'v_r:', v_r.shape

    for k in range(1, K + 1):
        R = restore1(sigma_r, u_r, v_r, k)
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), 2)
        print 'I:', I
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 12:
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')
            plt.title(u'奇异值个数：%d' % k)
    plt.suptitle(u'SVD与图像分解', fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()
