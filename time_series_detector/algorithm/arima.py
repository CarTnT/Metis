#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
from pmdarima.arima import auto_arima


class Arima(object):
    """
    In statistical quality control, the EWMA chart (or exponentially weighted moving average chart)
    is a type of control chart used to monitor either variables or attributes-type data using the monitored business
    or industrial process's entire history of output. While other control charts treat rational subgroups of samples
    individually, the EWMA chart tracks the exponentially-weighted moving average of all prior sample means.

    WIKIPEDIA: https://en.wikipedia.org/wiki/EWMA_chart
    """

    def __init__(self, start_p=0,start_q=0,test='adf',max_p=2,max_q=2,m=3,d=None,seasonal=True,start_P=0,D=1,trace=True,error_action='ignore',suppress_warnings=True,stepwise=False,coefficient=3):
        """
        :param alpha: Discount rate of ewma, usually in (0.2, 0.3).
        :param coefficient: Coefficient is the width of the control limits, usually in (2.7, 3.0).
        """
        self.start_q = start_q
        self.start_p = start_p
        self.test = test
        self.max_p = max_p
        self.max_q = max_q
        self.m = m
        self.d = d
        self.seasonal = seasonal
        self.start_P = start_P
        self.D = D
        self.trace = trace
        self.error_action = error_action
        self.suppress_warnings = suppress_warnings
        self.stepwise = stepwise
        self.coefficient = coefficient

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :return: 1 denotes normal, 0 denotes abnormal
        """
        data_train = X[:int(0.9*(len(X)))]
        data_test = X[int(0.9*(len(X))):]
        built_arimamodel = auto_arima(data_train,
                                 start_p=self.start_p,   # p最小值
                                 start_q=self.start_q,   # q最小值
                                 test=self.test,  # ADF检验确认差分阶数d
                                 max_p=self.max_p,     # p最大值
                                 max_q=self.max_q,     # q最大值
                                 m=self.m,        # 季节性周期长度，当m=1时则不考虑季节性
                                 d=self.d,      # 通过函数来计算d
                                 seasonal=self.seasonal, start_P=self.start_P, D=self.D, trace=self.trace,
                                 error_action=self.error_action, suppress_warnings=self.suppress_warnings,
                                 stepwise=self.stepwise  # stepwise为False则不进行完全组合遍历
                                 )
        pred_list = []
        for x in range(len(data_test)):
            # 输出索引，值
            pred_list.append(built_arimamodel.predict(n_periods=1))
            # 更新模型，model.update()函数，不断用新观测到的 value 更新模型
            built_arimamodel.update(data_test[x])
        diff_list = list(map(lambda x: x[0]-x[1], zip(pred_list, data_test)))
        sigma = np.sqrt(np.var(X))
        print(sigma)
        for x in range(len(diff_list)): 
            if abs(diff_list[x]) > self.coefficient * sigma:
                return 0, data_test[x], pred_list[x], int(0.9*(len(X)))+x+1
        return 1, 0, 0,0