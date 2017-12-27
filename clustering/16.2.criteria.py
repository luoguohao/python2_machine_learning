# !/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn import metrics
import numpy as np


if __name__ == "__main__":
    y =     [0, 0, 0, 1, 1, 1]

    label_idx = np.unique(y, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    print pi
    print pi_sum

    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    print -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))
    print np.log(0.5)

    print metrics.cluster.entropy(y)
    y_hat = [0, 0, 1, 1, 2, 2]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print u'同一性(Homogeneity)：', h
    print u'完整性(Completeness)：', c
    v2 = 2 * c * h / (c + h)
    v = metrics.v_measure_score(y, y_hat)
    print u'V-Measure：', v2, v

    print
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 2, 3, 3]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print u'同一性(Homogeneity)：', h
    print u'完整性(Completeness)：', c
    print u'V-Measure：', v

    # 允许不同值
    print
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [1, 1, 1, 0, 0, 0]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print u'同一性(Homogeneity)：', h
    print u'完整性(Completeness)：', c
    print u'V-Measure：', v

    y = [0, 0, 1, 1]
    y_hat = [0, 1, 0, 1]
    ari = metrics.adjusted_rand_score(y, y_hat)
    print ari

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    ari = metrics.adjusted_rand_score(y, y_hat)
    print ari
