# !/usr/bin/python
# -*- coding:utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import codecs
import random

infinite = float(-2 ** 31)


# 归一化及对数化
def log_normalize(a):
    s = 0
    for x in a:
        s += x
    if s == 0:
        print "Error..from log_normalize."
        return
    s = math.log(s)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - s


def log_sum(a):
    if not a:  # a为空
        return infinite
    m = max(a)
    s = 0
    for t in a:
        s += math.exp(t - m)
    return m + math.log(s)


def calc_alpha(pi, A, B, o, alpha):
    """
    给定lambda,定义到时刻t,部分观测序列为O1，O2，O3，...Ot,且状态为q[i]的概率为前向概率.
    记为: alpha[t][i] = P(O1,O2,O3,..,Ot, i[t] = q[i] | lambda)
    此处计算的是已对数化。
    :param pi:
    :param A:
    :param B:
    :param o:观测序列
    :param alpha: 初始前向概率值
    :return:
    """
    for i in range(4):
        # 计算初始时刻(即t=0)的前向概率 alpha[0][i] = B[i][O[0]] * pi[i]，此处都是使用对数化，所以，公式都是加法而不是乘法
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):  # 计算t时刻的前向概率
        for i in range(4):  # 计算t时刻的指定状态的前向概率
            for j in range(4):
                temp[j] = (alpha[t - 1][j] + A[j][i])
            # 计算t时刻指定状态的前向概率: alpha[t][i]= SUM(alpha[t-1][j]*A[j][i])*B[i][O[t]], 此处依然是对数化，使用加法
            alpha[t][i] = log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]


def calc_beta(pi, A, B, o, beta):
    """
    后向概率:给定lambda,定义在时刻t下状态为Q[i](Q为状态序列)的前提下，从t+1到T的部分观测序列为O[t+1],O[t+2],O[t+3]...O[T]的概率
    即为: beta[t][i] = P(O[t+1],O[t+2],O[t+3]...O[T], i[t] = Q[i] | lambda)
    :param pi:
    :param A:
    :param B:
    :param o:
    :param beta: 初始后向概率
    :return:
    """
    T = len(o)
    for i in range(4):
        # 计算beta初始值，由后向概率的定义，可得: beta[T-1][i] = 1
        beta[T - 1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T - 2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                # 根据beta[t+1][j]的值，递推计算beta[t][i]的值
                # beta[t][i] = SUM(A[i][j] * B[j][O[t+1]] * beta[t+1][j]), 此处为对数化
                temp[j] = A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
            # SUM 对数化求和
            beta[t][i] += log_sum(temp)


def calc_gamma(alpha, beta, gamma):
    """
    计算单个状态的概率
    定义: 求给定模型lambda和观测O,在时刻t处于状态Q[i]的概率, I为状态序列
    gamma[t][i] = P(I[t] = Q[i] | O,lambda) = (alpha[t][i]*beta[t][i]) / SUM(alpha[t][i]*beta[t][i])
    :param alpha:
    :param beta:
    :param gamma:
    :return:
    """
    for t in range(len(alpha)):
        for i in range(4):
            # 由公式(alpha[t][i]*beta[t][i]) / SUM(alpha[t][i]*beta[t][i])计算可得
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            # 因为是对数化，所以，变成加减运算
            gamma[t][i] -= s


def calc_ksi(alpha, beta, A, B, o, ksi):
    """
    计算两个状态的联合概率
    定义: 求给定模型lambda和观测序列O,在时刻t出于状态Q[i],并且在时刻t+1，出于状态Q[j]的联合概率
    ksi[t][i,j] = P(I[t]=Q[i], I[t+1]=Q[j] | lambda,O)
    :param alpha:
    :param beta:
    :param A:
    :param B:
    :param o:
    :param ksi:
    :return:
    """
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T - 1):
        k = 0
        for i in range(4):
            for j in range(4):
                # 由公式: ksi[t][i,j] = (alpha[t][i] * A[i][j] * B[j][O[t+1]] * Beta[t+1][j]) /
                # SUM(alpha[t][i] * A[i][j] * B[j][O[t+1]] * Beta[t+1][j]) 计算求得。

                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
                temp[k] = ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s


def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    """
    使用EM算法，计算pi, A, B
    :param pi:
    :param A:
    :param B:
    :param alpha:
    :param beta:
    :param gamma:
    :param ksi:
    :param o:
    :return:
    """
    # 计算pi
    T = len(alpha)
    for i in range(4):
        pi[i] = gamma[0][i]   # pi = gamma[0][i]/ SUM(gamma[0][i])

    s1 = [0 for x in range(T - 1)]
    s2 = [0 for x in range(T - 1)]
    for i in range(4):
        for j in range(4):
            for t in range(T - 1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = log_sum(s1) - log_sum(s2)  # 计算转移概率

    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        print "bw", i
        for k in range(65536):  # UTF-8字体个数
            valid = 0
            if k % 10000 == 0:
                print "bw - k", k
            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)     # 计算发射矩阵


def baum_welch(pi, A, B):
    """
    使用Baum-Welch算法，进行无监督的HMM参数估计，依赖EM算法
    :param pi: 初始pi
    :param A:  初始转移概率矩阵
    :param B:   初始发射矩阵
    :return:    返回，训练得到的pi, A, B
    """
    f = file(".\\news.txt")
    sentence = f.read().decode('utf-8')
    f.close()
    T = len(sentence)
    # 中文分词有四种隐状态
    alpha = [[0 for i in range(4)] for t in range(T)]  # 前向概率
    beta = [[0 for i in range(4)] for t in range(T)]  # 后向概率
    gamma = [[0 for i in range(4)] for t in range(T)]  # 单个状态的概率
    ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T - 1)]  # 两个状态的联合概率
    for time in range(3):
        print "calc_alpha"
        calc_alpha(pi, A, B, sentence, alpha)  # alpha(t,i):给定lamda，在时刻t的状态为i且观测到o(1),o(2)...o(t)的概率
        print "calc_beta"
        calc_beta(pi, A, B, sentence, beta)  # beta(t,i)：给定lamda和时刻t的状态i，观测到o(t+1),o(t+2)...oT的概率
        print "calc_gamma"
        calc_gamma(alpha, beta, gamma)  # gamma(t,i)：给定lamda和O，在时刻t状态位于i的概率
        print "calc_ksi"
        calc_ksi(alpha, beta, A, B, sentence, ksi)  # ksi(t,i,j)：给定lamda和O，在时刻t状态位于i且在时刻i+1，状态位于j的概率
        print "bw"
        bw(pi, A, B, alpha, beta, gamma, ksi, sentence)
        print "time", time
        print "Pi:", pi
        print "A", A
    return pi, A, B


def mle():  # 0B/1M/2E/3S
    # 使用已标注的文档进行最大似然估计，同样可以进行HMM参数估计

    # 一个句子的各个词的隐状态可分为四种:
    # 0: Begin(开始); 1:Middle(中间词)；2：End(结束词)；3：Single(单个词)
    # 如： 我爱中国 你
    # 其中"我"是Begin状态; "爱中"为Middle状态, "国"为 End状态; "你"为 Single状态；

    pi = [0] * 4  # npi[i]：i状态的个数, 隐状态分为四种
    a = [[0] * 4 for x in range(4)]  # na[i][j]：从i状态到j状态的转移个数
    b = [[0] * 65536 for x in range(4)]  # nb[i][o]：从i状态到o字符的个数， UTF-8的词总个数为65536
    f = file(".\\24.pku_training.utf8")
    data = f.read().decode('utf-8')
    f.close()
    tokens = data.split('  ')
    last_q = 2
    old_progress = 0
    print '进度：'
    for k, token in enumerate(tokens):
        progress = float(k) / float(len(tokens))
        if progress > old_progress + 0.1:
            print '%.3f' % progress
            old_progress = progress
        token = token.strip()
        n = len(token)
        if n <= 0:
            continue
        if n == 1:  # 表示一个单个词
            pi[3] += 1  # S 状态加1
            a[last_q][3] += 1  # 上一个词的状态[默认为结束状态](last_q)到当前状态[Single word](3S) 转移概率
            b[3][ord(token[0])] += 1  # 观测概率矩阵(发射矩阵/混淆矩阵)中: 状态为3的所在词加一
            last_q = 3
            continue

        # 初始向量
        # 只要token的长度不为1，那么状态Begin、End都需要增加1
        pi[0] += 1
        pi[2] += 1
        pi[1] += (n - 2)  # token中的中间词有多少个，状态为M的则加上多少个

        # 转移矩阵
        a[last_q][0] += 1
        last_q = 2
        if n == 2:
            a[0][2] += 1  # 如果n==2表示没有middle,直接是从Begin状态到End状态
        else:
            a[0][1] += 1  # 否则是先从Begin状态到Middle状态
            a[1][1] += (n - 3)  # 然后，从Middle状态到Middle状态
            a[1][2] += 1  # 最后，从Middle状态到End状态

        # 发射矩阵/混淆矩阵
        b[0][ord(token[0])] += 1  # 从Begin状态到当前中文字 加一
        b[2][ord(token[n - 1])] += 1  # 从End状态到当前中文字 加一
        for i in range(1, n - 1):
            b[1][ord(token[i])] += 1  # 从Middle状态到当前中文字加一

    # 正则化(将所有的初始向量、转移概率矩阵、混淆矩阵都取对数；因为计算结果中，如果不取对数，计算得到的概率值很小，
    # 极容易溢出，所以，取对数可以避免出现溢出)
    log_normalize(pi)
    for i in range(4):
        log_normalize(a[i])
        log_normalize(b[i])
    return [pi, a, b]  # 计算得到对数化的初始概率、转移概率矩阵、发射矩阵


def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')


def save_parameter(pi, A, B):
    f_pi = open(".\\pi.txt", "w")
    list_write(f_pi, pi)
    f_pi.close()
    f_A = open(".\\A.txt", "w")
    for a in A:
        list_write(f_A, a)
    f_A.close()
    f_B = open(".\\B.txt", "w")
    for b in B:
        list_write(f_B, b)
    f_B.close()


if __name__ == "__main__":
    """
    使用HMM算法对中文分词，生成初始向量、转移概率矩阵、发射矩阵
    若训练数据只有观测序列，那么HMM的学习需要使用EM算法，进行参数估计
    """
    pi, A, B = mle()  # 使用已标注的文档语料进行最大似然估计，同样可以进行HMM参数估计
    pi_n, A_n, B_n = baum_welch(pi, A, B)  # 使用非监督学习算法，依赖EM算法 -- Baum-Welch算法进行HMM参数估计
    save_parameter(pi_n, A_n, B_n)
    print "训练完成..."
