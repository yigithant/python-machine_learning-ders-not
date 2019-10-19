# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:26:33 2019

@author: yigit
"""
# %% library

#linear regression ile eğitilen makinenin amacı linear olarak değişen durumların öğrenilmesidir

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# %% csv dosyası okuma

original_train_set=pd.read_csv("C:/Users/yigit/Desktop/data_visua/linear_regression_example/train.csv")
original_test_set=pd.read_csv("C:/Users/yigit/Desktop/data_visua/linear_regression_example/test.csv")

#train ve test datalarımızı elde ettik
print(original_test_set.shape)
print("\n", type(original_test_set))
print("\n", original_test_set.info())

# %% dropna

#nan ifadeleri dataset içerisinden çıkarma

train_set=original_train_set.dropna()
test_set=original_test_set.dropna()

print(test_set.shape)
print("\n", type(test_set))
print("\n", test_set.info())

#data içerisinde nan bir ifade bulunmadıgı için data içeriği değişmedi

# %% 
x=train_set.iloc[:,0]
#convert x from pandas series to numpy array so;
x=x.values.reshape(-1,1)

print(x.shape)
print("\n", type(x))

y=train_set.iloc[:,1]
#convert y from pandas series to numpy array so;
y=y.values.reshape(-1,1)





# %% linear regression

reg=LinearRegression()
reg.fit(x,y)
print("reg score : ",reg.score(x, y))
print("\ncoeff : ", reg.coef_)

print("\nb0 : ", reg.predict([[0]]))

# %% visua
array=np.arange(0,100,1).reshape(-1,1)

y_head=reg.predict(array)
plt.plot(array,y_head,color="red")
plt.scatter(x,y,s=5,c='black',marker='*')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
