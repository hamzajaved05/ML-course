# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:37:19 2019

@author: Hamza
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
train = pd.read_csv('train.csv');
test = pd.read_csv('test.csv');
nptrain = train.to_numpy();
nptest = test.to_numpy();
nptrain_x = nptrain[...,2:12];
nptrain_y = nptrain[...,1];
nptest_x = nptest[...,1:];
reg = LinearRegression().fit(nptrain_x, nptrain_y);
test_labels = reg.predict(nptest_x);

#pd.DataFrame(data=test_labels,index = np.arange(10000,11999) columns = );
result = pd.DataFrame({'Id': np.arange(10000,12000), 'y' : test_labels});
result.to_csv('result.csv', index = False);