#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 12:49:44 2023

@author: mouhsine
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

file_path='~/Downloads/canada_per_capita_income.csv'
df=pd.read_csv(file_path)
print(df)
plt.xlabel('year') # plot the corbe
plt.ylabel('incom(US$)')
plt.scatter(df['year'],df['capita income'], color='red', marker='*')
df1=linear_model.LinearRegression()
df1.fit(df[['year']], df['capita income'])
print('in 2018 you will gete this incom capital    : ' , df1.predict(np.array([[2018]])))
