# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:29:08 2019

@author: Bryan
"""

import scipy.stats as ss
from collections import Counter
import math 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#READ DATA AND PREPROCESSING

player_df = pd.read_csv("data.csv")
numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']

player_df = player_df[numcols+catcols]
traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()
traindf = pd.DataFrame(traindf,columns=features)
y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']

X.head()

len(X.columns)

#SET SOME PARAMETERS
feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30

#1. PEARSON CORRELATION
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

print("pearson correlation")
print(cor_feature)
print("..............................................")
#2. CHI SQUARE FEATURES
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

print("chi feature")
print(chi_feature)
print("..............................................")

#3.RECURSIVE FEATURE ELIMINATION
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(solver='lbfgs'), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)

rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

#4. LASSO: SELECT FROM MODEL
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

print("lasso model")
print(embeded_lr_feature)
print("..............................................")

#5. TREE BASED SELECT FROM MODEL
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


print("random forest")
print(embeded_rf_feature)
print("..............................................")

#5.2 LIGHT BGM EXTENDS FROM TREE BASED
#from sklearn.feature_selection import SelectFromModel
#from lightgbm import LGBMClassifier

#lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

#embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
#embeded_lgb_selector.fit(X, y)

#embeded_lgb_support = embeded_lgb_selector.get_support()
#embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
#print(str(len(embeded_lgb_feature)), 'selected features')

#embeded_lgb_feature


#OVERALL
pd.set_option('display.max_rows', None)
 #put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
print(feature_selection_df.head(num_feats))