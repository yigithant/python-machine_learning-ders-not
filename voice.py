# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 05:53:57 2019

@author: yigit
"""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DATANIN HAZIRLANMASI GEREKEN BÖLÜMÜ!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#kütüphaneler import edildi
#numpy bir matematik kütüphanesi
#pandas ise bir data kütüphanesi


# %%

data=pd.read_csv("C:/Users/yigit/Desktop/data_visua/proje/voice.csv")

data.label=[1 if each=="male" else 0 for each in data.label]

print(data.info())

y=data.label.values
x_data=data.drop(["label"],axis=1)

# %% normalization
x=(x_data-np.min(x_data)/(np.max(x_data)-np.min(x_data)))
# (x - min(x))/(max(x)-min(x))
# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#x_train ,x_test y_train y_test değişkenlerine test ve train dataalrı atandı

x_train=x_train.T
y_train=y_train.T
x_test=x_test.T
y_test=y_test.T

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MATEMATİKSEL TARAFI VE FONKSİYONLAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# %% WEİGHT VE BIAS

def weight_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b

def sigm(z):
    y_head=1/(1+np.exp(-z))
    return y_head

# %% forward propagation
    
def forward_propagation(w,b,x_train,y_train):
    z=np.dot(w.T,x_train)+b
    y_head=sigm(z)
    loss=-(1-y)*np.log(1-y_head)-y*np.log(y_head)
    cost=(np.sum(loss))/x_train.shape[1]#coss değeri tüm loss değerlerini yani
    #tüm loss matrisindeki değerleri toplayıp x_train.shape[1] şeklindeki sayıya böler
    #x_train.shape[1] demek 20,2534 olan x_train matrisinin 2534e bölünmesi anlamına gelir
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]
    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost, gradients


# %% backward propagation
    

    

    

    











