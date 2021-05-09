#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import numpy as np
from sklearn.preprocessing import OneHotEncoder  # , LabelEncoder
import numpy as np
from sklearn.datasets import load_iris  # 数据集加载
from sklearn.model_selection import train_test_split  # 数据集的分割函数
from sklearn.preprocessing import StandardScaler  # 数据预处理
# 引入包含数据验证方法的包
from sklearn import metrics

class HiddenLayer:
    def __init__(self, x, num):  # x：输入矩阵   num：隐含层神经元个数
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))  #
        self.b = np.zeros([row, num], dtype=float)  # 随机设定隐含层神经元阈值，即bi的值
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)  # 随机产生-0.4 到 0.4 之间的数
            for j in range(row):
                self.b[j, i] = rand_b  # 设定输入层与隐含层的连接权值
        self.h = self.sigmoid(np.dot(x, self.w) + self.b)  # 计算隐含层输出矩阵H
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
 
 
'''
     train_data：被划分的样本特征集
     train_target：被划分的样本标签
     test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
     random_state：是随机数的种子。// 填0或不填，每次都会不一样。
'''
 
data_X = []
data_Y = []
 
with open('boston_house_prices.csv') as f:
    for line in f.readlines():
        line = line.split(',')
        data_X.append(line[:-1])
        data_Y.append(line[-1:])
# 转换为nparray
data_X = np.array(data_X, dtype='float32')
data_Y = np.array(data_Y, dtype='float32')
 
for i in range(data_X.shape[1]):
    _min = np.min(data_X[:, i])  # 每一列的最小值
    _max = np.max(data_X[:, i])  # 每一列的最大值
    data_X[:, i] = (data_X[:, i] - _min) / (_max - _min)  # 归一化到0-1之间
 
# 分割训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(data_X,  # 被划分的样本特征集
                                                    data_Y,  # 被划分的样本标签
                                                    test_size=0.2,  # 测试集占比
                                                    random_state=1)  # 随机数种子，在需要重复试验的时候，保证得到一组一样的随机数
 
a = HiddenLayer(x_train, 90)  # data_X
a.regressor_train(y_train)
 
result = a.regressor_test(x_test)
sum_cost = 0
j = 0
for i in result:
    infer_result = i  # 经过预测后的值
    ground_truth = y_test[j]  # 真实值
    # if infer_result-ground_truth>0 or infer_result-ground_truth<0:
    #     infer_result = ground_truth
    cost = np.power(infer_result - ground_truth, 2)
    sum_cost += cost
    j += 1
print("平均误差为:", sum_cost / j)
 
###画图###########################################################################
import matplotlib.pyplot as plt
 
fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
axes = fig.add_subplot(1, 1, 1)
line1, = axes.plot(range(len(result)), result, 'b--', label='ELM', linewidth=2)
line3, = axes.plot(range(len(y_test)), y_test, 'g', label='true')
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line3])
plt.title('ELM')
plt.show()
