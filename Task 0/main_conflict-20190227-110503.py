# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:37:19 2019

@author: Hamza
"""

import sklearn as skl
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
train = pd.read_csv('train.csv');
nptrain = train.to_numpy();
nptrain_x = nptrain[...,2:10];
nptrain_y = nptrain[...,1];
reg = LinearRegression().fit(nptrain_x, nptrain_y);
