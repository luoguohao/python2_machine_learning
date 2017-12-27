# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw


# 定义f: theta * x
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    # 蘑菇是否有毒预测
    # 读取数据
    # 该文件的数据格式是libsvm-data-format格式,即<label> <index1>:<value1> <index2>:<value2> ...
    data_train = xgb.DMatrix('12.agaricus_train.txt')
    data_test = xgb.DMatrix('12.agaricus_test.txt')


    # 设置参数
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logitraw'} # logitraw
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 4
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    importance_axes = xgb.plot_importance(bst)
    axes = xgb.plot_tree(bst, num_trees=3)
    plt.axes(axes)
    plt.grid(True)
    # xgb.to_graphviz(bst, num_trees=2)

    plt.show()


    # bst.save_model('xgboost_0001.model')  # 保存模型

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print y_hat
    print y
    error = sum(y != (y_hat > 0))
    error_rate = float(error) / len(y_hat)
    print '样本总数：\t', len(y_hat)
    print '错误数目：\t%4d' % error
    print '错误率：\t%.5f%%' % (100*error_rate)
