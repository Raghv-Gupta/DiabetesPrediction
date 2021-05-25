import pandas as pd                                                                  
import matplotlib.pyplot as plt
import numpy as np     
import math                                                 

data=pd.read_csv("diabetes.csv")                                                      
data.isnull().sum()                                                                   

data.shape                                                                            
data.dtypes                                                                           
X=data.iloc[:,0:8]                                                                     
Y=data.iloc[:, 8]                                                                     

from sklearn.model_selection import train_test_split                                  
X_train,X_test,Y_train,Y_test=train_test_split(X , Y,test_size=0.2,random_state=0)   

from sklearn.preprocessing import StandardScaler                                     
sc=StandardScaler()                                                                                                                       
X_train=sc.fit_transform(X_train)                                                     
X_test=sc.transform(X_test)                                                           


from keras.models import Sequential                                                   
from keras.layers import Dense,Dropout                                                
from keras.optimizers import Adam                                                     

opt= Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07)

classifier=Sequential()                                                                               
classifier.add(Dense(units=128,activation='relu',input_dim=8))
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])                      

history=classifier.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=32,verbose=2,epochs=150)    
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()




