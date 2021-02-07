# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 02:52:52 2021

@author: BERKAY
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#data preprocessing
veriler = pd.read_csv("data.csv") 

outlook = veriler[["outlook"]] 
print(outlook)
#encoding
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
play = le.fit_transform(veriler.iloc[:,-1])
print(play)
windy = le.fit_transform(veriler.iloc[:,-2:-1])
print(windy)
ohe = preprocessing.OneHotEncoder() 
outlook = ohe.fit_transform(outlook).toarray() 
print(outlook)

#numpy arrays
result1 = pd.DataFrame(data=outlook, index = range(14), columns = ["overcast","rainy","sunny"])
result2 = pd.DataFrame(data=windy, index=range(14), columns = ["windy"])
result3 = pd.DataFrame(data=play, index = range(14), columns=["play"])
result = pd.concat([result2,result3,result1,veriler.iloc[:,1:3]],axis=1)
print(result) 
#splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(result.iloc[:,:-1],result.iloc[:,-1:],test_size=0.33, random_state=0)
#linear regression predict
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
#backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values=result.iloc[:,:-1], axis=1 )
X_l = result.iloc[:,[1,2,3,4,5]].values
r_ols = sm.OLS(endog = result.iloc[:,-1:],exog =X_l) 
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:] 

regressor.fit(x_train,y_train) 

y_pred = regressor.predict(x_test)