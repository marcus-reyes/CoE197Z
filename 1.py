###Reyes, Marcus

###CoE 197Z Project 1.1

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout,BatchNormalization

from keras.optimizers import adam

import numpy as np

from numpy import genfromtxt


from sklearn import preprocessing

def print_unique_list(data):

    for i, v in enumerate(data.columns):
        print("Unique Values[",data.columns[i],"]:", len(data[v].unique()))

def load_x_train(path):
    ###Data Preprocessing
    data = pd.read_csv(path)

    #for strong reasons
                      #unique      #aka loc, numerous     #messy numerous
    do_not_include = ['wpt_name', 'subvillage',         'scheme_name',
                     #uniform value   
                     'recorded_by']

    do_not_one_hot = ['id','amount_tsh','gps_height','longitude','latitude','construction_year','population']

    do_not_include_tent = ['funder','installer','ward']

    do_not_include_temp = ['date_recorded']
    print(data.columns)


    #Drop values ot to be used
    data = data.drop(columns = do_not_include)
    data = data.drop(columns = 'id')
    data = data.drop(columns = do_not_include_temp)
    data = data.drop(columns = do_not_include_tent)


    #Turn the rest into one hot
    for i,w in enumerate(data.columns):
        # print("Before: ",len(data.columns),w)
        if w not in do_not_one_hot:
            data = pd.get_dummies(data, columns=[w], prefix = [w])
        # print("After: ",len(data.columns),w)

    train_col = data.columns
    
    print(data.columns,"X_test")

    x = data.to_numpy()

    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass
    return x, train_col

def load_y_train(path):
    data = pd.read_csv(path)
    print(data.columns)
    print(data.columns[1])
    print(data.head())
    data = pd.get_dummies(data, columns=[data.columns[1]], prefix = [data.columns[1]])
    print(data.columns)
    print(data.columns[1])
    print(data.head())
    data = data.drop(columns = 'id')
    y = data.to_numpy()

    print(y)

    print(y.shape)
    return y

def load_x_test(path, train_col):
    ###Data Preprocessing
    data = pd.read_csv(path)

    #for strong reasons
                      #unique      #aka loc, numerous     #messy numerous
    do_not_include = ['wpt_name', 'subvillage',         'scheme_name',
                     #uniform value   
                     'recorded_by']

    do_not_one_hot = ['id','amount_tsh','gps_height','longitude','latitude','construction_year','population']

    do_not_include_tent = ['funder','installer','ward']

    do_not_include_temp = ['date_recorded']
    print(data.columns)

    data_id = data['id']
    #Drop values ot to be used
    data = data.drop(columns = do_not_include)
    data = data.drop(columns = 'id')
    data = data.drop(columns = do_not_include_temp)
    data = data.drop(columns = do_not_include_tent)


    #Turn the rest into one hot
    for i,w in enumerate(data.columns):
        # print("Before: ",len(data.columns),w)
        if w not in do_not_one_hot:
            data = pd.get_dummies(data, columns=[w], prefix = [w])
        # print("After: ",len(data.columns),w)

    test_col = data.columns
    
    print(data.columns,"BEFORE")
    for i in range(len(test_col)):
        if test_col[i] not in train_col:
            data = data.drop(columns = test_col[i])
            print("Dropped",test_col[i])
    print(data.columns,"InBetween")   
    data_ins = [0]*(data.shape[0])
    for i in range(len(train_col)):
        if train_col[i] not in test_col:
            index_name = train_col[i]
            index = i
            data.insert(index,index_name,data_ins)
    print(data.columns,"AFTER")
    
    if(train_col == data.columns).all():
        print("Similar_Columns")
    x = data.to_numpy()

    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass
    return x, test_col,data_id

x, train_col = load_x_train("train_set_values.csv")
y = load_y_train("train_set_labels.csv")
x_train = x[:50000,:]

x_pretest = x[50000:,:]

y_train = y[:50000,:]

y_pretest = y[50000:,:]




###Model

hidden = 1024

dropout = 0.30

(trash, input_dim) = x.shape

activation = 'tanh'

model = Sequential()


model.add(Dense(hidden, input_dim = input_dim))

model.add(Dropout(dropout))

model.add(Activation(activation))

model.add(Dense(hidden, input_dim = input_dim))

model.add(Dropout(dropout))

model.add(Activation(activation))

model.add(Dense(hidden, input_dim = input_dim))

model.add(Dropout(dropout))

model.add(Activation(activation))


model.add(Dense(3,input_dim = hidden))

model.add(Activation('softmax'))

model.summary

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


max_score = -1
iter_of_max = 0
for i in range(27):

    model.fit(x_train, y_train, epochs = 1, batch_size = 512)

    score = model.evaluate(x_pretest, y_pretest, batch_size = 512)

    if  float(100 * score[1]) > float(max_score):
        max_score = float(100 * score[1])
        iter_of_max = i
    print("----------",i,"-----------")
    print("Test accuacy: ", (100.0 * score[1]))
    print("Maxscore: ", max_score, "at", iter_of_max)
    if(score[1] > 0.780):
        break
    
x_test, test_col,data_id = load_x_test("test_set_values.csv",train_col)

y_pred = model.predict(x_test)
print(x_test.shape)
status = np.argmax(y_pred, axis = 1)
status = status.reshape(x_test.shape[0],1)
id = x_test[:,0].reshape(x_test.shape[0],1)
y_pred = np.concatenate((id,status), axis = 1)
print(y_pred)
presub_id = pd.DataFrame(data_id)
presub_status = pd.DataFrame(status)
presub_status.replace({0:'functional',1:'functional needs repair',2:'non functional'}, inplace=True)
print(presub_id)
print(presub_status)
presubmission = pd.concat([presub_id,presub_status],axis = 1)

presubmission.iloc[:,0] = presubmission.iloc[:,0].astype(int)


presubmission.to_csv("submission_datadriven.csv",header = ["id", "status_group"],index = False)