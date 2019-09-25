###Reyes, Marcus

###CoE 197Z Project 1.1

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout,BatchNormalization

from keras import regularizers

from keras.optimizers import adam

import numpy as np

from numpy import genfromtxt


from sklearn import preprocessing

import matplotlib.pyplot as plt
import string

#for strong reasons
                  #unique      #aka loc, numerous     #messy numerous
do_not_include = ['wpt_name', 'subvillage',         'scheme_name',
                 #uniform value     #Duplicate of payment type   #Dup of quantity
                 'recorded_by',     'payment',                  'quantity_group',
                 #dup of wtptype less data      #dup of source  #dup of extr_type_group
                 'waterpoint_type_group',       'source_type',  'extraction_type',      'extraction_type_group',
                 #enc in regcode
                 'region']

do_not_one_hot = ['id','gps_height','construction_year','population','amount_tsh','date_recorded','longitude','latitude']

clean_up = ['construction_year','amount_tsh','population']

do_not_include_tent = ['funder','installer','ward','lga']

do_not_include_temp = ['num_private']
def print_unique_list(data):

    for i, v in enumerate(data.columns):
        print("Unique Values[",data.columns[i],"]:", len(data[v].unique()))

#Given the dataframe and the column it replaces all zero nonavailable values with the mean
def clean_data_with_mean(data, column):
    # print(data[column].mean())
    # print(data[column].median())
   
    # data["gps_height"].fillna(data.groupby(['region', 'district_code'])["gps_height"].transform("mean"), inplace=True)
    # data["gps_height"].fillna(data.groupby(['region'])["gps_height"].transform("mean"), inplace=True)
    # data["gps_height"].fillna(data["gps_height"].mean(), inplace=True)
    # data["population"].fillna(data.groupby(['region', 'district_code'])["population"].transform("median"), inplace=True)
    # data["population"].fillna(data.groupby(['region'])["population"].transform("median"), inplace=True)
    # data["population"].fillna(data["population"].median(), inplace=True)
    # data["amount_tsh"].fillna(data.groupby(['region', 'district_code'])["amount_tsh"].transform("median"), inplace=True)
    # data["amount_tsh"].fillna(data.groupby(['region'])["amount_tsh"].transform("median"), inplace=True)
    # data["amount_tsh"].fillna(data["amount_tsh"].median(), inplace=True)
    replace_map = {column:{0:data[column].mean()}}
    print(replace_map)
    data.replace(replace_map, inplace = True)
    # print(data[column].mean())
    # print(data[column].median())
    return data
def load_x_train(path, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp):
    ###Data Preprocessing
    data = pd.read_csv(path)
    
    for i,v in enumerate(clean_up):
        # print("About to clean up")
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



def load_x_test(path, train_col, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp):
    ###Data Preprocessing
    data = pd.read_csv(path)
    
    for i,v in enumerate(clean_up):
        # print("about to clean up")
        data = clean_data_with_mean(data,v)
    # print(data.columns)

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
    print(data.columns,"InBetween")   
    data_ins = [0]*(data.shape[0])
    for i in range(len(train_col)):
        if train_col[i] not in test_col:
            index_name = train_col[i]
            index = i
            data.insert(index,index_name,data_ins)
            print("Included",train_col[i])
    
    if(train_col == data.columns).all():
        print("Similar_Columns")
    x = data.to_numpy()
    print("HELLO")  
    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)

    try:
        del data
    except:
        pass
    return x, test_col,data_id

def load_y_train(path):
    data = pd.read_csv(path)
    
    data = pd.get_dummies(data, columns=[data.columns[1]], prefix = [data.columns[1]])
    
    data = data.drop(columns = 'id')
    
    y = data.to_numpy()
    
    return y



x, train_col = load_x_train("train_set_values.csv", do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp)
y = load_y_train("train_set_labels.csv")


###Model

hidden = 1024

dropout = 0.5

(trash, input_dim) = x.shape

activation = 'relu'

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

# model.summary

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


max_score = -1
iter_of_max = 0
train_plot = []
val_plot = []
x_axis = []
# print(x_train[1,:])
model.save_weights('model.h5')

amount = 50
k_folds = 10

###When i use k_folds instead of 10 for the np.zeros initialization it acts up
val_plot = np.zeros((k_folds,amount))
train_plot = np.zeros((k_folds,amount))
test_plot = np.zeros((k_folds,amount)) 
x_axis = np.zeros((k_folds,amount))

for j in range(k_folds):

    whole = np.arange(0,59400)
    test_range = np.arange((j)*(59400/k_folds), (j)*(59400/k_folds)+(59400/k_folds),dtype = 'int')
    
    train_range = np.delete(whole, test_range)
    # div = 59300
    print(train_range,len(train_range))
    print(test_range, len(test_range))
    x_train = x[train_range,:]

    x_pretest = x[test_range,:]

    y_train = y[train_range,:]

    y_pretest = y[test_range,:]
    for i in range(amount):

        history = model.fit(x_train, y_train, epochs = 1, batch_size = 4096)
        score = model.evaluate(x_pretest, y_pretest, batch_size = 512)

        if  float(100 * score[1]) > float(max_score):
            max_score = float(100 * score[1])
            iter_of_max = i
        print("----------",i,"-----------")
        print("Test accuacy: ", (100.0 * score[1]))
        print("Maxscore: ", max_score, "at", iter_of_max)
        if(score[1] > 0.85):
            break
        x_axis[j,i] = i
        test_plot[j,i] = (score[1])
        train_plot[j,i] = np.array(history.history['acc'])
        
        
    iter_of_max = 887
    if j == k_folds - 1:
        break
    model.load_weights('model.h5')
    print("Reloading Model")
    
    
x_test, test_col,data_id = load_x_test("test_set_values.csv",train_col, do_not_include, do_not_one_hot, clean_up, do_not_include_tent, do_not_include_temp)
# x_axis = np.asarray(x_axis)
# val_plot = np.asarray(val_plot)
# train_plot = np.asarray(train_plot)
axes = plt.gca()
axes.set_ylim([0.7,0.9])
plt.grid(which = 'both')
print(np.mean(test_plot[0:k_folds,:],axis = 0))
print(np.mean(train_plot[0:k_folds,:],axis = 0))
plt.plot(np.mean(test_plot[0:k_folds,:],axis = 0))
plt.plot(np.mean(train_plot[0:k_folds,:],axis = 0))
###Steadist is 0.79 fluctuations
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


plt.show()