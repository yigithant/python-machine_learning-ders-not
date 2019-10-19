# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:06:37 2018

@author: user
"""

# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#kütüphaneler import edildi
#numpy bir matematik kütüphanesi
#pandas ise bir data kütüphanesi

# %% read csv
data = pd.read_csv("C:/Users/yigit/Desktop/data_visua/data.csv")
#dosya okuma işlemi gerçekleştirildi
data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
#Unnamed: 32 ve  id adındaki değişkenler gereksiz görüldüğü için data.drop([],axis=1,inplace=True) şeklinde 
#data içerisinden kaldırıldı
#axis=1 diyerek tüm o sütunların içeriği ortadan kaldırıldı ve inplace=True ifadesi ile oluşturulan yeni data -dataya kayıt edildi
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#data içerisindeki  diagnosis sütunu içerisindeki  m ve b ifadeleri yerine 1 ve 0  değerleri atandı böylece object olan
#bu değerler artık integer haline geldi
print(data.info())
#data hakkında info alınıyor
y = data.diagnosis.values
#y değerine 1 ve 0 olarak değiştirilen m ve b ifadeleri  atıldı.numpy array olarak atandı
#569, luk bir array biçiminde
x_data = data.drop(["diagnosis"],axis=1)
#x_data değişkenine diagnosis haricindeki sütunlar hariç diğer data içeriği atandı.pandas data frame türünde
#burda object olan değerler yani m ve b değerleri integer ifadelere çevrildi yani 1 yada 0 haline çevrildi


# %% normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# (x - min(x))/(max(x)-min(x))
#normalizasyon işlemi gerçekleştirildi
#böylece dataframe içerisindeki 100 değeri ile 0.01 değeri arasında oluşacak olan uçurumdan kaynaklı
#küçük sayının etkisizliği ortadan kaldırılmaya çalışıldı
#bu ise np.min(x_data) featureların içerisindeki her bir minimum değerinin bulur ve bunları satır satır sütun sütun uygun featurelardan çıkarır
#aynı şekilde payda içinde yapar bu işlemi 

# %% train test split
from sklearn.model_selection import train_test_split
#sklearn kütüphanesinden train_test_split çekiliyor
#çünkü mevcut datanın bir kısmı eğitim amaçlı diğer kısmı test amaçlı yapılmaktadır
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
#x_train ve x_test değişkenlerine x normalize edilmiş datanın değerleri atanıyor
#x_train kısmına mevcut x datasının yüzde 80 i atanıyor
#x_test datasına mevcut datanın yüzde 20 si atanıyor
#aynı şekilde y_test ve y_train içinde geçerli 
#test_size=0.2 yaparak test tarafının yüzde 20 olacagı ifade ediliyor
#random_state=42 ise bir id gibi seçimlerin genelde aynı olmasını sağlar

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
#x_train x_test y_test y_train değerlerinin transpozları alınıyor böylece
#satırlar sütun sütunlar satır oluyor
#artık featurelar satır şeklinde dizili
print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# %% parameter initialize and sigmoid function
# dimension = 30
def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b


# w,b = initialize_weights_and_bias(30)
#weight ve bias değerleri oluşturuldu
#weight bir dimension,1 lik bir matris ve içeriği 0.01 olarak atandı
#bias ise sadece 0.0 değeri atandı ve return w,b yapıldı 
#dimension değeri ise fonksiyonun bir parametresi olarak alındı
def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
# print(sigmoid(0))
#sigmoid fonksiyonu z adındaki bir parametreyi alıyor
#ve bunu y_head adındaki bir değişkene atıyor
#weight ve bias değerlerinin toplamı olan z değeri 1/(1+ exp(-z)) işlemi yapılıyor
    #böylece y_head değerine bir olaslık değeri çıkmış oluyor
    #0 ile 1 arasında bir değer döndürüyor
    #böylece olasılık ortaya çıkarılmış oluyor

    

# %%
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    #w.T bir 1,30 luk bir matrisdir. x_train ise 30,455 bir matrisdir
    #np.dot ile matris çarpımı yapıldıgında çıkan sonuç ise 1,455 bir matrisdir
    #bu matris bias değeri olan b ile toplanır ve z matrisi elde edilir bu matris
    #1,455lik bir matrisdir
    y_head = sigmoid(z)
    #z matrisinin her bir değeri sigmoid fonksiyonuna yani 1/(1+exp(-z)) formulü içerisinde işlem görür
    #böylece bir y_head değeri elde edilir
    #y_head adındaki değişkene z ifadesinin 
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    #loss formulüne göre bir loss değeri oluşturulur. loss 1,455 lik bir matrisdir. 
      
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    #cost formulüne göre bir integer ifade elde edilir
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    #weight ve bias ın türevi alınır
    #burası ezber gibi bişey
    #gradients adındaki değişkene atanır. gradients bir dictionarydir
    #
    return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#%%  # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

# %% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300) #num_iterations arttıkça test datasının üzerinde edilen testin
#doğrulugu artmaktadır
#learning_rate ve num_iterations bunlar tune edilir el ile takip edilir



#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))







































































