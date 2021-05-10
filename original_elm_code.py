#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#correct code

# the dataset can be downloaded from 'https://www.kaggle.com/c/digit-recognizer/data'
# the code is from "https://towardsdatascience.com/build-an-extreme-learning-machine-in-python-91d1e8958599"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2

#data preparation 
train = pd.read_csv('dataset/mnist_train.csv')
test = pd.read_csv('dataset/mnist_test.csv')

onehotencoder = OneHotEncoder(categories='auto')
scaler = StandardScaler()

X_train = scaler.fit_transform(train.values[:,1:-1])
# -1 to solve the mismatch between training and testing
# all row and except the first column
y_train = onehotencoder.fit_transform(train.values[:,:1]).toarray()

X_test = scaler.fit_transform(test.values[:,1:])
y_test = onehotencoder.fit_transform(test.values[:,:1]).toarray()

input_size = X_train.shape[1]
hidden_size = 1000

input_weights = np.random.normal(size=[input_size,hidden_size])
# 783 1000
biases = np.random.normal(size=[hidden_size])
# 1000

def relu(x):
   return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    # (42000, 1000) + (1000,)
    # add every line, keep the row as the same
    H = relu(G)
    return H

output_weights = np.dot(pinv2(hidden_nodes(X_train)), y_train)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)
correct = 0
total = X_test.shape[0]

for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_test[i])
    correct += 1 if predicted == actual else 0
accuracy = correct/total
print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)


