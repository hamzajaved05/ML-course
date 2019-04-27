# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:50:13 2019

@author: Hamza
"""

"""
Created on Tue Mar  6 10:57:08 2019

@author: Hamza
"""
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import hypertools as hyp
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
import random
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv');
nptrain = train.to_numpy();
features = nptrain[...,2:];
labels = nptrain[...,1];

lambda_set = np.linspace(0.3,.45,num = 100)

features = np.concatenate((features,np.square(features),np.exp(features),np.cos(features)),axis=1);  #intercept = true
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features);

#features = np.concatenate((features,np.ones((features.shape[0],1))),axis = 1)    #intercept = false

#trainx , testx, trainy, testy = train_test_split(features, labels, test_size = 0.001, random_state = 15)
trainx = features;
trainy = labels;
kf = KFold(n_splits=10)
MSE_result = np.array([]);
MSEs = np.array([]);

MSE_result = [];
for lambd in lambda_set:
    for trainindex, testindex in kf.split(trainx):
        trainxCV, testxCV = trainx[trainindex], trainx[testindex];
        trainyCV, testyCV = trainy[trainindex], trainy[testindex];
    
        test = Lasso(alpha = lambd,fit_intercept = True).fit(trainxCV, trainyCV);
        predicted = test.predict(testxCV);
        MSE_result = np.append(MSE_result,MSE(predicted,testyCV)**0.5)
        
    MSEs = np.append(MSEs,MSE_result.mean());
    MSE_result = np.array([]);    
    
plt.plot(lambda_set,MSEs, color = 'green')
finalmodel = Lasso(alpha = lambda_set[np.argmin(MSEs)],fit_intercept = True).fit(trainx,trainy)
coefficients = np.append(np.divide(finalmodel.coef_,np.sqrt(scaler.var_)),(finalmodel.intercept_ - (np.dot(finalmodel.coef_,np.divide(scaler.mean_,np.sqrt(scaler.var_))))));   #for intercept = true

#coefficients[:-1]= np.divide(finalmodel.coef_[:-1],np.sqrt(scaler.var_))    #intercept = false
#coefficients[-1] = coefficients[-1] - (np.dot(finalmodel.coef_[:-1],np.divide(scaler.mean_,np.sqrt(scaler.var_))))    #intercept = false

result = pd.DataFrame({'labels' : coefficients});
result.to_csv('result.csv', index = False, header = False);
