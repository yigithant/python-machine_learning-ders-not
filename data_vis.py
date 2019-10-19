# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:35:43 2019

@author: yigit
"""

#%%
#linear regression nedir?

import pandas as pd
#import pandas as pd 
#pandas kütüphanesi pd takma adıyla alınıyor
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#matplot kütüphanesi plt takma adıyla alınıyor
df=pd.read_csv("C:/Users/yigit/Desktop/data_visua/linear-regression-dataset.csv",sep=";")
#df=pd.read_csv("csv dosyasının adresi")
#dataframe pandas kütüphanesinden read_csv ile çekiliyor
#sep=";" ile csv içersindeki ayrı ayrı bölümlere ayrılması sağlanıyor
plt.scatter(df.deneyim,df.maas)
#plt.scatter(x,y)
#df içerisindeki deneyim ve maas bilgileri scatter şeklinde yani konumları nokta şeklinde gösterilerek yazılıyor

plt.xlabel("deneyim")
plt.ylabel("maas")
#plt.xlabel ve plt.ylabel x ve y  eksenlerine isimleri veriliyor
plt.show()
#plt.show()
#bu komut ile görsel ekrana basılıyor


#%%


# linear regression 
#linear regression doğrusal olarak bir bağlantı kurulması konusunda önemli
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LinearRegression
#bu kütüphane ile Linear Regression aracı çekiliyor

#linear regression modeli
linear_reg=LinearRegression()
#linear_reg adındaki değişkene sklearn.linear_model içerisinden çekilen araç atandı
#pandas olan x i numpy arraye çevrilir
x=df.deneyim.values.reshape(-1,1)# reshape edildi
#df.deneyim tür olarak seridir
#df.deneyim.values ile numpy.ndarrays biçimine dönüştürülür
#df.deneyim.values.shape şeklinde arrayin biçimine bakıldıgında 14, biçimindedir
#bu biçimi reshape(-1,1) ile 14,1 şeklinde bir arraye dönüştürülür.
y=df.maas.values.reshape(-1,1)
#x için yapılanlar maas içinde yapılır
x.shape#burda çıktı 14, şeklinde yani 14 e 1 lik bir array

linear_reg.fit(x,y)#linear regression için en uygun çizgi bulundu
#linear_reg.fit(x,y)
#bu komut ile yani .fit komutu ile x ve y arasında oluşan linear regression çizgisinin elde edilmesi
#sağlanır. x ve y değerleri bu konumda iyi belirlenmelidir


b0=linear_reg.predict([[0]])#x=0 iken b0 değerini gösterir.
#burda çizilen çizginin x ve y eksenlerinde kesiştiği noktalar gösteriliyor
#b0=linear_reg.predict([[değer]])
#burda değer bölümüne x ekseni içerisinden bir değer sorgulanır. örneğin 5 yıllık bir deneyim için
#y eksenine düşen değeri gösterir yani maaşı. b0 için ise 0 değeri ile bu değere bakılıyor
#grafiğin başladığı en alt taraf

#diğer bir yöntem. yukardakiyle aynı şeyi yapıyor
b0_=linear_reg.intercept_
#b0=linear_reg.intercept_
#bu komut sadece b0 değerine karşılık gelen değeri gösterir
#yani oluşturulan grafikte çizilen linear regression çizgisinin en alt tarafındaki değeri  gösterir

#b1 coefficient bulmak için
b1=linear_reg.coef_
#b1= linear_reg.coef_ 
#burda b1 değerini elde etmek amacıyla bu komut kullanılır
#ÖNEMLİ::::b1 değeri oluşturlan linear regression çizgisinin eğimidir

#%%
import numpy as np
#import numpy as np
#numpy kütüphanesi np takma adıyla çekilir
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)#16, olan şekli 16,1 yapmak gerekiyor
#array=np.array([]).reshape(-1,1)
#bir adet array oluşturuldu bu oluşturulan array reshape ile 16, lık bir arrayden 16,1 lik bir arraya dönüştürüldü

plt.scatter(x,y)
#plt.scatter(x,y)
#scatter olarak x ve y çizilir
y_head=linear_reg.predict(array)
#y_head=linear_reg.predict([[array]])
#y_head arrayine array olarak atanan değerlere karşılık gelen noktalar atanıyor
#linear regression çizgisi üzerinde y eksenine düşen noktalar atanıyor

plt.plot(array,y_head,color="red")
#plt.plot(array,y_head,color="red")
#plot olarak array noktaları ve y_head noktaları çizdiriliyor bu çizginin rengi kırmızı olarak belirleniyor

linear_reg.predict([[100]])
#linear_reg.predict([[değer]])
#sadece burda 100 yıllık bir deneyime karşılık gelen ücretin karşılıgına bakıldı deneme amaçlı
plt.show()
#scatter ve plot çizimleri ekrana aynı pencereden basılması için en sonra 
#plt.show()
#komutu eklendi
#%%
#multiple regression
#iki veya farklı değişkenin birbirleri arasındaki ilişki

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#import pandas as pd
#import numpy as np
#from sklearn.linear_model import LinearRegression
#kütüphaneler çekildi
df=pd.read_csv("C:/Users/yigit/Desktop/data_visua/multiple-linear-regression-dataset.csv")
#df=pd.read_csv("yol")
#burda sep=";" kullanmadan ayrıldı ilginç

x=df.iloc[:,[0,2]].values#iloc kullanımı şu şekilde iloc[satır,sütun]
#satır tarafında : ile tüm satırlar alınırken
#sütun tarafında 0 ve 2.sütunlar alındı
#dataframe içerisindeki bölümleri çekmek için kullanılan komut
#iloc[satir,sütun] şeklinde tanımlanır x adındaki arraye atanır x.shape=14,2 matris haline getirir 14e 2 lik bir matris oluşur

y=df.maas.values.reshape(-1,1)#df.maas bir seridir. bunu values ile array haline getilir.reshape(-1,1) ile 14,1 lik bir matris halini alır


multiple_linear_regression=LinearRegression()
#burda LinearRegression aracı bir değere atandı
multiple_linear_regression.fit(x,y)
#x ve y değerlerinde en uygun olan bir linear doğrultu belirleniyor
#x burda iki farklı değer taşıyor ayrı ayrı bulunuyor
multiple_linear_regression.intercept_#b0 değeri
b1_b2=multiple_linear_regression.coef_#b1 değeri yani grafiğin eğimi b1 ve b2 için farklı eğriler


print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)

# predict
multiple_linear_regression.predict(np.array([[10,35],[5,35]]))
plt.scatter(df.iloc[:,[0]],y)
plt.show()

#%%
#polinomal linear regression

import pandas as pd
import matplotlib.pyplot as plt
#pandas ve matplotlib.pyplot kütüphaneleri plt takma adıyla atandı

df = pd.read_csv("C:/Users/yigit/Desktop/data_visua/polynomial-regression.csv",sep = ";")


y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)
#x ve y değerleri serilerden array durumuna getirildi ve 15,1 lik bir matris haline getirildi

 
plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
# linear regression =  y = b0 + b1*x
# multiple linear regression   y = b0 + b1*x1 + b2*x2
#linear regression ve multiple linear regression formülleri yukardaki gibidir. bu formüller ile regression bulunur

from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LinearRegresssion
#bu şekilde linearregression aracı alınır

lr=LinearRegression()
#lr adlı değişkene araç atanır

lr.fit(x,y)

y_head=lr.predict(x)
#x e göre predict edilen yani noktalar ile bulunan linear regression çizgisi üzerinde aradaki fark elde edilir

plt.plot(x,y_head,color="red",label ="linear")
plt.show()
#linear regression çizgisi çıkartılır. çıktı ekranında görüleceği gibi araba fiyatı ile araba hızı arasındaki bağıntıyı
#linear regression ile bulunamayacaktır
#bu sebeble polinomal linear regression kullanılır

# %% polinomal linear regression

# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n
#formülü yukarıdaki gibidir

from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import PolynomialFeatures
#PolynomialFeatures aracı sklearn.preprocessingden çekilir
polynomial_regression = PolynomialFeatures(degree = 2)
#bu araç polynomial_regression adındaki değişkene atanır. degree olarak gösterilen ise kaç derecede işlem yapılacağıdır
#degreenin artması eğrinin daha yumuşak gösterilmesine yardımcı olacaktır
x_pol=polynomial_regression.fit_transform(x)
#x ekseni için polinomial regression değeri fit_transform ile bulunur


#fit etme
linear_regression2 = LinearRegression()
linear_regression2.fit(x_pol,y)

y_head2 = linear_regression2.predict(x_pol)
#y_head2 ise x_pol değerine göre bir y_head belirlenmesini sağlar. regression çizgisinin y ekseni üzerindeki değeri ile 
#gerçekteki x değerinin y üzerindeki değeri arasındaki ilişkiyi gösterir.



plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()
#degree değeri arttırılırsa oluşturulan regression çzgisi daha yumuşak bir hal alır


# %% Decision_Tree_Regression


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#gerekli olan kütüphaneler import edildi
#pandas matplotlib numpy kütüphaneleri import edildi
df=pd.read_csv("C:/Users/yigit/Desktop/data_visua/decision-tree-regression-dataset.csv",sep=";",header=None)
#dataframe csv dosyasından çekildi. başlıgı olmadıgı için header=None eklendi böylece ilk satır bir feature oldu
x=df.iloc[:,0].values.reshape(-1,1)
#ilk sutün x değerine atandı
y=df.iloc[:,1].values.reshape(-1,1)
#ikinci sütun y değerine atandı
#decision tree

from sklearn.tree import DecisionTreeRegressor
#decisiontreeregressor aracı sklearn den çekildi
tree_reg = DecisionTreeRegressor()   # random sate = 0 default değeri none 
#tree_reg adındaki değişkene atandı
tree_reg.fit(x,y)
#x ve y değerleri fit edildi

tree_reg.predict([[5]])
#!!!!!!!!!UNUTMA!!!!!!!!!!!!!
#PREDICT İFADESİNDE PARANTEZ İÇİNE İKİ ADET KÖŞELİ PARENTEZ KONMASI GEREKMEKTEDİR
#x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x)
#y_head değeri x e göre elde edildi
# %% visualize
plt.scatter(x,y,color="red")
plt.plot(x,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()


x_=np.arange(min(x),max(x),0.01).reshape(-1,1)#minimumdan maximuma kadar 0.01 aralıklar değerler oluşturuyor reshape ile 
y_head = tree_reg.predict(x_)
#bu oluşturulan yeni x_ arrayi ile artık iki nokta arasındaki diğer noktalarda verilecek değerler bir merdiven şeklini alıyor. 
#yani leaf denilen alanlara bölünmüş oluyor

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
# %% random forest
#!!!!!!!!!!!!!!!ÖNEMLİ BİR KONU!!!!!!!!!!!!!!!!!!!!!!
#bir çok farklı algoritamayı birleştiriyor

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("C:/Users/yigit/Desktop/data_visua/random-forest-regression-dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=10,random_state=42)
#n_estimators --> random of tree kaç adet tree kullanılacak
#random_state --> n sayıdaki sample seçimini yapıyor

rf.fit(x,y)

#görselleştirme

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ücret")
plt.show()

# %% Evaluation Regression Models

#R^2 r_square çıkarma
#burda yapılan predict-tahminlerin doğrulugunu bulmak amacıyla yapılır

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#gerekli olan kütüphaneler import edildi
#pandas matplotlib numpy kütüphaneleri import edildi
df=pd.read_csv("C:/Users/yigit/Desktop/data_visua/decision-tree-regression-dataset.csv",sep=";",header=None)
#dataframe csv dosyasından çekildi. başlıgı olmadıgı için header=None eklendi böylece ilk satır bir feature oldu
x=df.iloc[:,0].values.reshape(-1,1)
#ilk sutün x değerine atandı
y=df.iloc[:,1].values.reshape(-1,1)
#ikinci sütun y değerine atandı
#decision tree

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

y_head=rf.predict(x)

#r^2 r_square

from sklearn.metrics import r2_score

print("r_score:", r2_score(y,y_head))

# %% random forest devam
#linear regression örneğini deneme



import pandas as pd
#import pandas as pd 
#pandas kütüphanesi pd takma adıyla alınıyor
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#matplot kütüphanesi plt takma adıyla alınıyor
df=pd.read_csv("C:/Users/yigit/Desktop/data_visua/linear-regression-dataset.csv",sep=";")
#df=pd.read_csv("csv dosyasının adresi")
#dataframe pandas kütüphanesinden read_csv ile çekiliyor
#sep=";" ile csv içersindeki ayrı ayrı bölümlere ayrılması sağlanıyor
plt.scatter(df.deneyim,df.maas)
#plt.scatter(x,y)
#df içerisindeki deneyim ve maas bilgileri scatter şeklinde yani konumları nokta şeklinde gösterilerek yazılıyor

plt.xlabel("deneyim")
plt.ylabel("maas")
#plt.xlabel ve plt.ylabel x ve y  eksenlerine isimleri veriliyor
plt.show()
#plt.show()
#bu komut ile görsel ekrana basılıyor

#%%
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LinearRegression
#bu kütüphane ile Linear Regression aracı çekiliyor

#linear regression modeli
linear_reg=LinearRegression()
#linear_reg adındaki değişkene sklearn.linear_model içerisinden çekilen araç atandı
#pandas olan x i numpy arraye çevrilir
x=df.deneyim.values.reshape(-1,1)# reshape edildi
#df.deneyim tür olarak seridir
#df.deneyim.values ile numpy.ndarrays biçimine dönüştürülür
#df.deneyim.values.shape şeklinde arrayin biçimine bakıldıgında 14, biçimindedir
#bu biçimi reshape(-1,1) ile 14,1 şeklinde bir arraye dönüştürülür.
y=df.maas.values.reshape(-1,1)
#x için yapılanlar maas içinde yapılır
x.shape#burda çıktı 14, şeklinde yani 14 e 1 lik bir array

linear_reg.fit(x,y)#linear regression için en uygun çizgi bulundu
#linear_reg.fit(x,y)
#bu komut ile yani .fit komutu ile x ve y arasında oluşan linear regression çizgisinin elde edilmesi
#sağlanır. x ve y değerleri bu konumda iyi belirlenmelidir

y_head=linear_reg.predict(x)


from sklearn.metrics import r2_score

print("r_score:", r2_score(y,y_head))
#r2_square değerinin 1 e yakın olması gerekiyor
# %% Logistic regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read csv

data=pd.read_csv("C:/Users/yigit/Desktop/data_visua/data.csv")
print(data.info())
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)#axis 1 yapılarak tüm sütunlar gitti sıfır yapılsa satırlar gidecekti. inplace=True ile data içeriğine kayıt edildi

data.diagnosis=[1 if each =="M" else 0 for each in data.diagnosis]
#data.info ile datanın türüne bakıldıktan sonra object olan ifadeleri diagnosis ile 1 veya 0  a çevirdik

y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%normalization

#0 ile 1 arasında tüm değerleri scale edilir

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#x=(x - minimum x)/(maximum x -minimum x)

# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#mevcut datanın %80e train %20si test amaçli
#random_state 42 değerine göre böldü

x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T
#satır ve sütunlar yer değiştirir

# %% parameter initialize and sigmoid function

def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b
#♦weigh ve bias değerleri atandı
#dimension değeri ile weight tarafında dimension,1 lik bir matris oluşturuldu
#ve bunların içerisinde 0.01 lik değerler atandı
#biasa ise sadece 0.0 değeri atandı

def sigm(z):
    y_head=1/(1+np.exp(-z))
    return y_head

# %% Implementing Forward and Backward Propagation
    
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigm(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling deep learning ile analtılacak
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients

# %% updating(learning) paramters
    
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):#learning_rate öğrenme hızı number_of_iteration tekrar
    cost_list=[]
    cost_list2=[]
    index=[]
    
    for i in range(number_of_iteration):
        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w=w-learning_rate*gradients["derivative_weight"]
        b=b-learning_rate*gradients["derivative_bias"]
        if i%10==0:
            cost_list2.append(cost)
            index.append(i)
            print("cost after iteration %i: %f"%(i,cost))
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("number of iteration")
    plt.ylabel("cost")
    plt.show()
    return parameters,gradients,cost_list2

# %% predict 
def predict(w, b,x_test):
    z=sigm(np.dot(w.T,x_test)+b)
    Y_predict=np.zeros(1,x_test.shape[1])
    for i in range(z.shape[1]):
        if z[0,i]<=0.5:
            Y_predict[0,i]=0
        else:
                Y_predict[0,i]=1
    return Y_predict

# %% logistic regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)    





















