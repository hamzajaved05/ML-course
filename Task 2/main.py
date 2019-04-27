# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:53:46 2019

@author: Hamza
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('train.csv');
nptrain = train.to_numpy();

labels = nptrain[:,1]
features = nptrain[:,2:]
del nptrain, train
scaler = StandardScaler()
scaler.fit(features)
zeros , ones, twos = [sum((labels == 0).astype(int)),sum((labels == 1).astype(int)),sum((labels == 2).astype(int))]
#q = np.corrcoef(features.T,labels)

#ratio is 1:1:1 almost
"""" For one vs all  
"""
kf = KFold(n_splits=10)
accuracy = np.array([]);

for trainindex, testindex in kf.split(labels):
    trainxCV, testxCV = features[trainindex], features[testindex];
    trainyCV, testyCV = labels[trainindex], labels[testindex];
    test = MLPClassifier(activation='relu', hidden_layer_sizes= (12,12, 6, 6) , learning_rate_init=0.001,solver = 'adam', learning_rate = 'adaptive', max_iter = 2000).fit(trainxCV,trainyCV)
    predicted = test.predict(testxCV);
    accuracy = np.append(accuracy,accuracy_score(testyCV,predicted))
print(accuracy.mean())

