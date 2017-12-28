# !/usr/bin/python
# -*- coding:utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import codecs
import random


def load_train():
    f = file(".\\pi.txt", mode="r")
    for line in f:
        pi = map(float, line.split(' ')[:-1])
    f.close()

    f = file(".\\A.txt", mode="r")
    A = [[] for x in range(4)]  # 转移矩阵：B/M/E/S
    i = 0
    for line in f:
        A[i] = map(float, line.split(' ')[:-1])
        i += 1
    f.close()

    f = file(".\\B.txt", mode="r")
    B = [[] for x in range(4)]
    i = 0
    for line in f:
        B[i] = map(float, line.split(' ')[:-1])
        i += 1
    f.close()
    return pi, A, B


def viterbi(pi, A, B, o):
    """
        在计算veterbi算法的时候，此处都是取指数
    :param pi: 初始向量
    :param A:  状态转移概率矩阵
    :param B:   发射矩阵
    :param o:   观测序列
    :return:   返回最优的状态序列
    """

    T = len(o)  # 观测序列
    delta = [[0 for i in range(4)] for t in range(T)]  # 初始化,总共有观测序列长度大小(即时刻)的delta，每个时刻的delta有4种隐状态(B,M,E,S)
    pre = [[0 for i in range(4)] for t in range(T)]  # 前一个状态   # pre[t][i]：t时刻的i状态，它的前一个状态是多少

    for i in range(4):
        # 计算初始delta值:pi[i]*B[i][O[0]] , 因为此处pi,A,B都是指数化的数据，因此，delta计算公式变成了加法，而不是乘法
        delta[0][i] = pi[i] + B[i][ord(o[0])]
    for t in range(1, T):  # 计算每一时刻的delta
        for i in range(4):  # 计算每一时刻每个状态的delta
            # 每一时刻的delta等于前一时刻的 MAX(delta[前一时刻][前一状态] * A[前一状态][当前状态]) * B[当前状态][O[观测序列]] 此处由于已对数化，所以，都变成了加号
            delta[t][i] = delta[t - 1][0] + A[0][i]  # delta[前一时刻][前一状态] * A[前一状态][当前状态]
            for j in range(1, 4):
                vj = delta[t - 1][j] + A[j][i]
                if delta[t][i] < vj:
                    delta[t][i] = vj  # 取最大: MAX(delta[前一时刻][前一状态] * A[前一状态][当前状态])
                    pre[t][i] = j  # 记录下前一时刻取哪个状态，使得当前delta取最大，方便回溯
            delta[t][i] += B[i][ord(o[t])]  # 当前时刻delta的值: MAX(delta[前一时刻][前一状态] * A[前一状态][当前状态]) * B[当前状态][O[观测序列]]

    decode = [-1 for t in range(T)]  # 解码：回溯查找最大路径
    q = 0
    # 计算查找，哪个状态值下，最后时刻的delta最大值
    for i in range(1, 4):
        if delta[T - 1][i] > delta[T - 1][q]:
            q = i
    decode[T - 1] = q  # delta最大值对应的状态值即为最优状态值
    for t in range(T - 2, -1, -1):  # 根据最后时刻的delta最大值，回溯查找前n个的最优状态序列
        q = pre[t + 1][q]  # 取当前时刻的状态为Q的前一个时刻的状态值，即为它的最优路线
        decode[t] = q
    return decode  # 返回最优路线


def segment(sentence, decode):
    """
    根据训练好的状态序列，将文章分词
    需要根据当前词是结束词还是单个词，如果是则使用|分隔；如果是开始词和中间词，那么需要连接起来。
    :param sentence: 句子
    :param decode:  状态序列(0:表示开始词；1：表示中间词；2：表示结束词；3：表示单个词)
    :return:
    """
    N = len(sentence)
    i = 0
    while i < N:  # B/M/E/S
        if decode[i] == 0 or decode[i] == 1:  # Begin or Middle
            j = i + 1
            while j < N:
                if decode[j] == 2:
                    break
                j += 1
            print sentence[i:j + 1], "|",
            i = j + 1
        elif decode[i] == 3 or decode[i] == 2:  # single or end
            print sentence[i:i + 1], "|",
            i += 1
        else:
            print 'Error:', i, decode[i]
            i += 1


if __name__ == "__main__":
    # 使用HMM进行中文分词
    pi, A, B = load_train()
    f = file(".\\24.mybook.txt")
    data = f.read()[3:].decode('utf-8')
    f.close()
    decode = viterbi(pi, A, B, data)
    segment(data, decode)
