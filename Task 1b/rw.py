"""
Created on Wed Mar 20 13:04:59 2019

@author: Hamza
"""

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import hypertools as hyp
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
import random
from sklearn.preprocessing import StandardScaler


noise = np.random.normal(-5,5,1000)
line = np.linspace(0,10,1000)
labels = np.add(line*5,noise)+5
labels = labels.reshape((-1,1))
#labels = np.concatenate((labels,np.ones((1000,1))),axis = 1)
features = line.reshape((-1,1))
model = LinearRegression(fit_intercept = True).fit(features,labels)
