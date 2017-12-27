# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    stock_max, stock_min, stock_close, stock_amount = np.loadtxt('6.SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True)
    N = 100
    stock_close = stock_close[:N]
    print stock_close

    n = 5   # 五日均线， 移动平均
    weight = np.ones(n)
    weight /= weight.sum()
    print weight
    stock_sma = np.convolve(stock_close, weight, mode='valid')  # simple moving average, 使用卷积计算移动平均
    print stock_sma

    # 五日均线; 指数滑动移动平均。表示当前天的数据影响程度比每一个前一天的数据影响程度更高
    weight = np.linspace(1, 0, n)
    weight = np.exp(weight)
    weight /= weight.sum()  # 指数移动均值
    print weight
    stock_ema = np.convolve(stock_close, weight, mode='valid')  # exponential moving average, 一维卷积

    t = np.arange(n-1, N)
    # p(x) = p[0] * x**deg + ... + p[deg]
    poly = np.polyfit(t, stock_ema, 10)     # 最小二乘多项式拟合，10次多项式拟合
    print poly
    stock_ema_hat = np.polyval(poly, t)     # 使用10次多项式拟合，计算预测值

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label=u'原始收盘价')
    t = np.arange(n-1, N)
    plt.plot(t, stock_sma, 'b-', linewidth=2, label=u'简单移动平均线')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(np.arange(N), stock_close, 'r-', linewidth=1, label=u'原始收盘价')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    plt.plot(t, stock_ema_hat, 'm-', linewidth=3, label=u'指数移动平均线估计')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
