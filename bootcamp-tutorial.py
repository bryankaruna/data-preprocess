# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:43:08 2019

@author: Bryan
"""

import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(100.0,20.0,10000)

plt.hist(incomes, 100)
plt.show();

#mean
mean = np.mean(incomes)
print(mean)

#median
median = np.median(incomes)
print(median)