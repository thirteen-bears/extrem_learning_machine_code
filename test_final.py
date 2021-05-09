#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class HiddenLayer():
    def __init__(self,x,num): # x:input data, n:number of hidden layer
        row = x.shape[0]
        col = x.shape[1]
        rnd = np.random.RandomState(4444) # 4444 is the index of the random generator
        self.w = rnd.uniform([-1,1],(col,num))
        self.b = np.zeros([row,num],dtype = float)
    
    def sigmoid():
        

