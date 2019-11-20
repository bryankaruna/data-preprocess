# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 23:37:11 2019

@author: Bryan
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tips_df = sns.load_dataset('tips')
print(tips_df.head())

#correlation and using the annotation argument
sns.heatmap(tips_df.corr(), annot=True)

print(tips_df.info())

#selecting category column
cat_list = tips_df.select_dtypes(include='category').columns.to_list()
print(cat_list)

#Putting all the category columns in a list(using list comprehension except the day column because it has more than two unique variables.
new_cat = [i for i in cat_list if not i == 'day']
print(new_cat)

#To create a list of lists for each of the variables in each category column
great_list = []
for i in new_cat:
  new_list = tips_df[i].unique().to_list()
  great_list.append(new_list)

print(great_list)

