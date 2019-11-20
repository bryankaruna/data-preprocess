# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 00:08:51 2019

@author: Bryan
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv(r'D:/creditcard.csv')

print(data.head())
print("file read successfuly")
print(data.isnull().sum())
X = data[["Amount"]]#data[["V2","V5","V9","V22","V28"]]
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y)
print("train and test sets created")

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train,y_train)
print("classifier created")
score = knn.score(X_test,y_test)
print("model evaluated")
print(score)