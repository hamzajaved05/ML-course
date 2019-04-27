"""
Created on Tue Mar  5 14:45:08 2019

@author: Hamza
"""
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
train = pd.read_csv('train.csv');

nptrain = train.to_numpy();
nptrain_x = nptrain[...,2:12];
nptrain_y = nptrain[...,1];
lambda_set = [0.1,1,10,100,1000];
kf = KFold(n_splits=10)
MSE_result = [];
for lambd in lambda_set:
    for train_index, test_index in kf.split(nptrain_x):
        X_train, X_test = nptrain_x[train_index], nptrain_x[test_index];
        y_train, y_test = nptrain_y[train_index], nptrain_y[test_index];
        
    
    
        reg = Ridge(alpha = lambd,fit_intercept=False).fit(X_train, y_train);
        test_labels = reg.predict(X_test);
        MSE_result.append(MSE(test_labels,y_test)**0.5)
        

#pd.DataFrame(data=test_labels,index = np.arange(10000,11999) columns = );
MSE_result = np.array(MSE_result);
MSE_result = MSE_result.reshape([5,10])
result = pd.DataFrame({'labels' : MSE_result.mean(axis = 1)});
result.to_csv('result.csv', index = False, header = False);