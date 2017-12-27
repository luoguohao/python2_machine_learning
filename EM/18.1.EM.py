# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy import stats

np.set_printoptions(suppress=True)
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    style = 'mysql'

    np.random.seed(0)
    mu1_fact = (0, 0, 0)
    cov_fact = np.identity(3)

    data1 = np.random.multivariate_normal(mu1_fact, cov_fact, 400)
    mu2_fact = (2, 2, 1)
    cov_fact = np.identity(3)
    data2 = np.random.multivariate_normal(mu2_fact, cov_fact, 100)
    data = np.vstack((data1, data2))
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        g.fit(data)
        print '类别概率:\t', g.weights_
        print '均值:\n', g.means_, '\n'
        print '方差:\n', g.covariances_, '\n'
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        n, d = data.shape
        # 随机指定
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5

        print '先验均值:', mu1, mu2, '\n'
        print '先验方差:\n', sigma1, '\n', sigma2, '\n'
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print i, ":\t", mu1, mu2
        print '类别概率:\t', pi
        print '均值:\t', mu1, mu2
        print '方差:\n', sigma1, '\n\n', sigma2, '\n'

    # 预测分类
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'原始数据', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    # 计算通过GMM计算得到的两个高斯分布的均值和实际的均值哪个的欧几里得距离更近，用于区分是哪个分类
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    if order[0] == 0:   # 如果mu1_fact正好和mu1距离最近，即 mu1 预测的就是 mu1_fact
        c1 = tau1 > tau2    # 此时判断tau1 是否大于tau2，如果大于，说明第一分类的概率更大，此时属于第一分类
    else:
        # 如果mu1_fact和mu2距离最近，即 mu2 预测的就是 mu1_fact
        c1 = tau1 < tau2    # 此时判断tau1 是否小于tau2，如果小于，说明第一分类的概率更大，此时属于第一分类

    c2 = ~c1    # 此时可以将测试数据分为两个类别
    acc = np.mean(y == c1)  # 计算为正例的准确度
    print u'准确率：%.2f%%' % (100 * acc)

    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)
    # plt.suptitle(u'EM算法的实现', fontsize=20)
    # plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.show()
