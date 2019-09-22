###Reyes, Marcus
###CoE 197Z Project 1
###Kaggle-https://www.kaggle.com/c/cat-in-the-dat

import pandas as pd
import keras
import numpy as np
from numpy import genfromtxt

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,BatchNormalization,Conv1D
from keras.optimizers import adam
from keras.regularizers import l1,l2
from keras.initializers import he_normal, he_uniform
from keras import initializers
from sklearn import preprocessing


###Data preprocessing
data = pd.read_csv("train.csv")
#For now ignore the data you don't know how to handle
#drop = ['id', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
# drop = ['id', 'nom_7','nom_8', 'nom_9']
drop = ['id', 'nom_9']
mynom_9 =  data['nom_9'].astype(str).apply(lambda x: int(x,16))

data = data.drop(columns = drop)

data = pd.concat([data,mynom_9],axis = 1)
print(data['nom_9'])
#Categorical to one_hot
#https://www.datacamp.com/community/tutorials/categorical-data#encoding
one_hot = ['bin_3', 'bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_3', 'ord_4','ord_5','day','month', 'nom_5']

#Categorical to labelled
labelled = ['nom_6','nom_7', 'nom_8']

#ordinal to normalized
ord_1_mapping = {'ord_1' : {'Novice':1, 'Contributor':2,'Expert':3,'Master':4,'Grandmaster':5}}
ord_2_mapping = {'ord_2' : {'Freezing':1, 'Cold':2, 'Warm':3, 'Hot':4, 'Boiling Hot':5, 'Lava Hot':6}}

data.replace(ord_1_mapping, inplace=True)
data.replace(ord_2_mapping, inplace=True)
# ord_2  ∈{  Freezing, Cold, Warm, Hot, Boiling Hot, Lava Hot  }
# {'carrier': {'AA': 1, 'OO': 
for i,w in enumerate(one_hot):
   data = pd.get_dummies(data, columns=[w], prefix = [w])
print(data.values.shape)

for i,w in enumerate(labelled):
    labels = data[w].astype('category').cat.categories.tolist()

    replace_map_comp = {w: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

    data.replace(replace_map_comp, inplace=True)
    
    del labels, replace_map_comp


y = data['target'].to_numpy()
y = keras.utils.to_categorical(y, 2)

data = data.drop(columns = ['target'])
print(data.columns)
x = data.to_numpy()

#Sample pringting to ensure [0:1]
print(x[7,:])
print(x[1,:])
###Normalize data to large to be one-hot-encoded
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

# print(x[4,:])
# print(x[7,:])
# x_train = x[:255000,:]
# x_pretest = x[255000:,:]

# y_pretest = y[255000:,:] #note y_train is really y_all here
# y_train = y[:255000,:]

val_ind = np.arange(0*30000,(1)*30000)
x_pretest = x[val_ind,:]
x_train = np.delete(x, val_ind, axis = 0)

y_pretest = y[val_ind,:]
y_train = np.delete(y, val_ind, axis = 0)

###Model

dropout = 0.25
(trash, input_dim) = x.shape
print(x.shape)
initializer1 = initializers.he_normal(seed=None)
hidden = 2048
model = Sequential()
shrinking = hidden
activation = 'tanh'
model.add(Dense(hidden,input_dim = input_dim, kernel_initializer=initializer1))
model.add(Dropout(dropout))
model.add(Activation(activation))

#Free up variables
try:
    del data,x,y
except:
    pass
for i in range(1):
    model.add(Dense(hidden))
    model.add(Dropout(dropout))
    model.add(Activation(activation))
    model.add(Dense(hidden,kernel_initializer = 'zeros'))
    model.add(Dropout(dropout))
    model.add(Activation(activation))
    model.add(Dense(hidden,kernel_initializer = 'ones'))
    model.add(Dropout(dropout))
    model.add(Activation(activation))
    # hidden = int(hidden/2)
    
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#It seesm to run into the same place over and over again
#Try different initializers
#Initializers that don't work(sub73max): ones,zeros


    

###To keep track of validation error
for i in range(100):
    # val_ind = np.arange(i*10000,(1+i)*10000)
    # x_pretest = x[val_ind,:]
    # x_train = np.delete(x, val_ind, axis = 0)
    
    # y_pretest = y[val_ind,:]
    # y_train = np.delete(y, val_ind, axis = 0)
    
    model.fit(x_train, y_train, epochs = 1, batch_size = 4096*8)

    score = model.evaluate(x_pretest, y_pretest, batch_size = 512)
    print("\nTest accuacy: %.1f%%" % (100.0 * score[1]))
    print("Iteration: ",i)









###Testing

 
data = pd.read_csv("test.csv")
mynom_9 =  data['nom_9'].astype(str).apply(lambda x: int(x,16))
data = data.drop(columns = drop)
data = pd.concat([data,mynom_9],axis = 1)

data.replace(ord_1_mapping, inplace=True)
data.replace(ord_2_mapping, inplace=True)

for i,w in enumerate(one_hot):
   data = pd.get_dummies(data, columns=[w], prefix = [w])


for i,w in enumerate(labelled):
    labels = data[w].astype('category').cat.categories.tolist()

    replace_map_comp = {w: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

    data.replace(replace_map_comp, inplace=True)
    
    del labels, replace_map_comp



x_test = data.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test)
 
# print("X_testshape",x_test.shape)
y_test = model.predict(x_test)



###Formatting into csv submittable
id = np.arange(start = 300000, stop = 500000)
id = np.transpose(id)
id = id.reshape(200000,1)
y_temp = y_test[:,1].reshape(200000,1)
y_pred = np.concatenate((id, y_temp), axis = 1)
print(id.shape)
print(y_test[:,0].shape)
print(y_pred.shape)
presubmission = pd.DataFrame(y_pred)

presubmission.iloc[:,0] = presubmission.iloc[:,0].astype(int)
presubmission.iloc[:,1] = presubmission.iloc[:,1].astype(float)


presubmission.to_csv("submission.csv",header = ["id","target"],index = False)