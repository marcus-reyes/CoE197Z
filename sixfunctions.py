import pandas as pd

import numpy as np

from numpy import genfromtxt

from sklearn import preprocessing

import matplotlib.pyplot as plt
import string
from keras import backend as K
#Given the dataframe and the column it replaces all zero nonavailable values with the mean
def clean_data_with_mean(data, column):
   
    replace_map = {column:{0:data[column].mean()}}
    # print(replace_map)
    data.replace(replace_map, inplace = True)
    return data
    
def load_train(path,path2, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp):
    ###Data Preprocessing
    data = pd.read_csv(path)
    labels = pd.read_csv(path2)
    data = data.join(labels.set_index('id'), on = 'id')
    append_data = data.loc[data['status_group'] == 'functional needs repair']
    
    # print(data)
    # data = pd.concat([data, append_data], ignore_index = True)
    # data.drop(data.tail(7).index,inplace = True)
    data = data.sample(frac=1).reset_index(drop=True)
    # print(data)
    # print("HELPE")
    y = pd.concat([data['id'], data['status_group']], axis = 1)
    # print(y)
    # print(y.shape)    
    
    y = pd.get_dummies(y, columns=[y.columns[1]], prefix = [y.columns[1]])
    
    y = y.drop(columns = 'id')
    
    y = y.to_numpy()
    data = data.drop(columns = 'status_group')
    for i,v in enumerate(clean_up):
        print("About to clean up")
        data = clean_data_with_mean(data,v)


    #Drop values ot to be used
    data = data.drop(columns = do_not_include)
    data = data.drop(columns = 'id')
    data = data.drop(columns = do_not_include_tent)
    data = data.drop(columns = do_not_include_temp)

    for i in range(len(data['date_recorded'])):
        data['date_recorded'][i] = int(data['date_recorded'][i].replace("-","")[2:6])
    
    
    #Turn the rest into one hot
    for i,w in enumerate(data.columns):
        # print("Before: ",len(data.columns),w)
        if w not in do_not_one_hot:
            prev = len(data.columns)
            data = pd.get_dummies(data, columns=[w], prefix = [w],dummy_na = True)
            now = len(data.columns)
            print("Expanded",w,"Change",prev,now)
        # print("After: ",len(data.columns),w)

    train_col = data.columns
    
    # print(data.columns,"X_test")

    x = data.to_numpy()

    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass
    return x, train_col, y



def load_x_test(path, train_col, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp):
    ###Data Preprocessing
    data = pd.read_csv(path)
    
    for i,v in enumerate(clean_up):
        data = clean_data_with_mean(data,v)

    data_id = data['id']
    #Drop values ot to be used
    data = data.drop(columns = do_not_include)
    data = data.drop(columns = 'id')
    
    
    for i in range(len(data['date_recorded'])):
        data['date_recorded'][i] = int(data['date_recorded'][i].replace("-","")[2:6])
        
        
    data = data.drop(columns = do_not_include_tent)
    data = data.drop(columns = do_not_include_temp)

    #Turn the rest into one hot
    for i,w in enumerate(data.columns):
        # print("Before: ",len(data.columns),w)
        if w not in do_not_one_hot:
            prev = len(data.columns)
            data = pd.get_dummies(data, columns=[w], prefix = [w],dummy_na = True)
            now = len(data.columns)
            print("Expanded",w,"Change",prev,now)

    test_col = data.columns
    
    for i in range(len(test_col)):
        if test_col[i] not in train_col:
            data = data.drop(columns = test_col[i])
            print("Dropped",test_col[i])
    # print(data.columns,"InBetween")   
    data_ins = [0]*(data.shape[0])
    for i in range(len(train_col)):
        if train_col[i] not in test_col:
            index_name = train_col[i]
            index = i
            data.insert(index,index_name,data_ins)
            print("Included",train_col[i])
    
    if(train_col == data.columns).all():
        # print("Similar_Columns")
        pass
    x = data.to_numpy()
    # print("HELLO")  
    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass
    return x, test_col,data_id


# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))