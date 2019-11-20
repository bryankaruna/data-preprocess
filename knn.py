# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:37:17 2019

@author: Bryan
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

#data = data.dropna()
data = data[['Reactions','Strength','Volleys','Position','Nationality','LongPassing','Finishing','BallControl','LongShots','FKAccuracy','Balance','Aggression','Weak Foot','Stamina','SprintSpeed','Overall','Composure','Potential','BallControl','International Reputation','ShortPassing','Vision']]
data.dropna()
print(data.head(5))

print("................................")
print("file read successfuly")
X = data[['Composure','Potential','International Reputation','ShortPassing','Vision']]
y = data["Overall"]

X_train, X_test, y_train, y_test = train_test_split(X,y)
print("train and test sets created")

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
print("classifier created")
score = knn.score(X_test,y_test)
print("model evaluated")
print(score)