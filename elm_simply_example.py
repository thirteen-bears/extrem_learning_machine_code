#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# the dataset can be downloaded from 'https://www.kaggle.com/c/digit-recognizer/data'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2

train = pd.read_csv('dataset/mnist_train.csv')
test = pd.read_csv('dataset/mnist_test.csv')

onehotencoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()

X_train = scaler.fit_transform(train.values[:,1:]) # all row and except the first column
y_train = onehotencoder.fit_transform(train.values[:,:1]).toarray()

X_test = scaler.fit_transform(test.values[:,1:])
y_test = onehotencoder.fit_transform(test.values[:,:1]).toarray()