#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#test

# read the table
# abstract data of city 1 from the table

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  # 
from sklearn.model_selection import train_test_split  # 
from sklearn.preprocessing import StandardScaler  # 
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from scipy.linalg import pinv2

class HiddenLayer:
    def __init__(self, x, num):  # x：输入矩阵   num：隐含层神经元个数
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))  # [columns, num][5,90]
        self.b = np.zeros([row, num], dtype=float)  # 随机设定隐含层神经元阈值，即bi的值 [(3647, 90)]
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)  # 随机产生-0.4 到 0.4 之间的数
            for j in range(row):
                self.b[j, i] = rand_b  # 设定输入层与隐含层的连接权值
        self.h = self.sigmoid(np.dot(x, self.w) + self.b)  # 计算隐含层输出矩阵H
        #(3647, 5) *[5,90] + 3647, 90
        self.H_ = np.linalg.pinv(self.h)  # 获取H的逆矩阵
        # print(self.H_.shape)
 
    # 定义激活函数g(x) ，需为无限可微函数
    def sigmoid(self, x):
        print(x)
        return 1.0 / (1 + np.exp(-x))
 
    '''  若进行分析的训练数据为回归问题，则使用该方式 ，计算隐含层输出权值，即β '''
 
    def regressor_train(self, T):
        C = 2
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        T = T.reshape(-1, 1)
        self.beta = np.dot(all_m, T)
        return self.beta
 
    """
           计算隐含层输出权值，即β 
    """
 
    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        # print(T)
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        # print(T)
        C = 3
        I = len(T)
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        self.beta = np.dot(all_m, T)
        return self.beta
 
    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result
 
    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result
 
        
#df=pd.DataFrame(pd.read_csv('data1.csv',header=1))
data = pd.read_csv("data.csv",encoding="gb2312",index_col ="城市编码") 
# abstract the 3rd line with City1
dt0 = data.loc["City1"]
dt1 = dt0.replace('―', np.NaN) 
dt = dt1.fillna(method='backfill', axis=0, inplace=False)
dt["检测时间"] = pd.to_datetime(dt["检测时间"]) #将数据类型转换为日期类型
dt = dt.set_index("检测时间")
#dt_09 = dt['2019-09']
#dt_09 = dt['2019']
dt_09=dt.loc['2019-09-01':'2019-09-30']
dt_10=dt.loc['2019-10-01':'2019-10-31']
dt_11=dt.loc['2019-11-01':'2019-11-30']
dt_12=dt.loc['2019-12-01':'2019-12-31']
dt_01=dt.loc['2020-01-01':'2020-01-30']
dt_02=dt.loc['2020-02-01':'2020-02-29']

# prepare of training data: data_x， label_x
temp_09 = dt_09.iloc[:,[1,2,3,4,5]]
x_train = temp_09.append(dt_10.iloc[:,[1,2,3,4,5]])
x_train = x_train.append(dt_11.iloc[:,[1,2,3,4,5]])
x_train = x_train.append(dt_12.iloc[:,[1,2,3,4,5]])
x_train = x_train.append(dt_01.iloc[:,[1,2,3,4,5]])

temp_09 = dt_09.iloc[:,[6]]
y_train = temp_09.append(dt_10.iloc[:,[6]])
y_train = y_train.append(dt_11.iloc[:,[6]])
y_train = y_train.append(dt_12.iloc[:,[6]])
y_train = y_train.append(dt_01.iloc[:,[6]])

# prepare for testing data: data_y, label_y
x_test = dt_02.iloc[:,[1,2,3,4,5]]
y_test = dt_02.iloc[:,[6]]

#---------------------------------------------------------------------------------------------------------
## Uniform the data
scaler = StandardScaler()
x_train_1 = scaler.fit_transform(x_train.values[:,:])
y_train_1 = scaler.fit_transform(y_train.values[:,:])
#x_test_1 = scaler.fit_transform(x_test.values[:,:])
#y_test_1 = scaler.fit_transform(y_test.values[:,:])
x_test_1 = np.asarray(x_test.values[:,:],'float64')
y_test_1 = np.asarray(y_test.values[:,:],'float64')

a = HiddenLayer(x_train_1, 1000)  # data_X
a.regressor_train(y_train_1)
 
result = a.regressor_test(x_test_1)
sum_cost = 0
j = 0
for i in result:
    infer_result = i  # 经过预测后的值
    ground_truth = y_test_1[j]  # 真实值
    # if infer_result-ground_truth>0 or infer_result-ground_truth<0:
    #     infer_result = ground_truth
    cost = np.power(infer_result - ground_truth, 2)
    #cost = np.abs(infer_result - ground_truth)/ground_truth
    sum_cost += cost
    j += 1
average_cost = sum_cost/j


###画图###########################################################################
import matplotlib.pyplot as plt
 
fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
axes = fig.add_subplot(1, 1, 1)
line1, = axes.plot(range(len(result)), result, 'b--', label='ELM', linewidth=2)
line3, = axes.plot(range(len(y_test_1)), y_test_1, 'g', label='true')
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line3])
plt.title('ELM')
plt.show()