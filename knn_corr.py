# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:55:22 2019

@author: Bryan
"""

import pandas as pd

df = pd.read_csv(r'D:/creditcard.csv')

print(df.head())

print(df.isna().sum())

print(df.info())

df.hist()

# Correlation heatmap
plt.rcParams['figure.figsize']=(25,16)
hm=sns.heatmap(df[["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14",
                   "V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26",
                   "V27","V28","Amount","Class"]].corr(), fmt='.2g',annot = True, vmin=-1, vmax=1, center=0, linewidths=.5, cmap='Blues', square=True)
hm.set_title(label='Heatmap of dataset', fontsize=20)
hm