###Reyes, Marcus

###CoE 197Z Project 1.1

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout,BatchNormalization,Input,Concatenate

from keras.optimizers import adam

import numpy as np

from numpy import genfromtxt


from sklearn import preprocessing
import matplotlib.pyplot as plt
import string
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
    for i in range(len(data['date_recorded'])):
        data['date_recorded'][i] = int(data['date_recorded'][i].replace("-","")[2:6])
    data = data.drop(columns = do_not_include_tent)
    print(data['date_recorded'])
    arr_frame_inputs = []
    train_col_list = []
    print(data.columns)
    for i,w in enumerate(data.columns):
        if w not in do_not_one_hot:
            print("Onehot",w)
            arr_frame_inputs.append(pd.DataFrame(pd.get_dummies(data[w])))
            train_col_list.append(pd.get_dummies(data[w]).columns)
        else:
            arr_frame_inputs.append(pd.DataFrame(data[w]))
            train_col_list.append(w)
            print("nochange",w)
    b = arr_frame_inputs[0]
    print(arr_frame_inputs[0].columns)
    print(b.columns)
    print("COLUMNS")
    
    #We need this format to fit the test data set
    for i,w in enumerate(data.columns):
        # print("Before: ",len(data.columns),w)
        if w not in do_not_one_hot:
            data = pd.get_dummies(data, columns=[w], prefix = [w])
        # print("After: ",len(data.columns),w)
    train_col = data.columns
    x = data.to_numpy()

    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    x = min_max_scaler.fit_transform(x)
    try:
        del data
    except:
        pass
    return arr_frame_inputs, x, train_col,train_col_list

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
    for i in range(len(data['date_recorded'])):
        data['date_recorded'][i] = int(data['date_recorded'][i].replace("-","")[2:6])
    data = data.drop(columns = do_not_include_tent)
    print(data['date_recorded'])
    arr_frame_inputs = []
    print(data.columns)
    for i,w in enumerate(data.columns):
        if w not in do_not_one_hot:
            print("Onehot",w)
            arr_frame_inputs.append(pd.DataFrame(pd.get_dummies(data[w])))
        else:
            arr_frame_inputs.append(pd.DataFrame(data[w]))
            print("nochange",w)
    print(arr_frame_inputs[0].columns)
    print("COLUMNPROOF")
    data_ins = [0]*(data.shape[0])
    try:
        del data
    except:
        pass
    return arr_frame_inputs, data_ins,data_id

print("CODE PROPER")

arr_train, x_compare_train, train_col, train_col_list = load_x_train("train_set_values.csv")
y = load_y_train("train_set_labels.csv")
y_train = y[:40000,:]
y_pretest = y[40000:,:]
print("POST LOADS")
print(arr_train[0].shape)

inputkeras = []
layer1 = []
activation = 'tanh'
dropout = 0.3
for a in range(len(arr_train)):
    if len(arr_train[a].shape) == 2:
        input_shape = (arr_train[a].shape[1],)
    else:
        input_shape = (1,)
    print(input_shape)
    inputkeras.append(keras.layers.Input(shape = input_shape))
    layer1.append(keras.layers.Dense(1, activation = activation)(inputkeras[a]))
merged = keras.layers.concatenate(layer1,axis = 1)
block1 = keras.layers.Dense(1024, activation = activation)(merged)
block1_1 = keras.layers.Dropout(dropout)(block1)

block2 = keras.layers.Dense(1024, activation = activation)(block1_1)
block2_1 = keras.layers.Dropout(dropout)(block2)

block3 = keras.layers.Dense(1024, activation = activation)(block2_1)
block3_1 = keras.layers.Dropout(dropout)(block3)
block4 = keras.layers.Dense(3, activation = 'softmax')(block3_1)
model = keras.models.Model(inputs=inputkeras, outputs=block4)
    



model.summary()
x_train = []
x_pretest = []
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Manipulate the list of arrays into a normalized array split into train and val
for a in range(len(arr_train)):
    if len(arr_train[a].shape) == 2:
        reshape_shape = (arr_train[a].shape[1],)
    else:
        reshape_shape = (1,)
    temp = np.asarray(arr_train[a])
    temp = np.reshape(temp,(-1,reshape_shape[0]))
    min_max_scaler = preprocessing.MinMaxScaler()
    temp = min_max_scaler.fit_transform(temp)
    print(temp)
    print(temp.shape)
    x_train.append(temp[:40000,:])
    x_pretest.append(temp[40000:,:])
    
max_score = -1
iter_of_max = 0
train_plot = []
val_plot = []
x_axis = []
for i in range(1000):

    history = model.fit(x_train, y_train, epochs = 1, batch_size = 4096)

    score = model.evaluate(x_pretest, y_pretest, batch_size = 512)

    if  float(100 * score[1]) > float(max_score):
        max_score = float(100 * score[1])
        iter_of_max = i
    print("----------",i,"-----------")
    print("Test accuacy: ", (100.0 * score[1]))
    print("Maxscore: ", max_score, "at", iter_of_max)
    if(score[1] > 0.783):
        break
    x_axis.append(i)
    val_plot.append(score[1])
    train_plot.append(history.history['acc'])
x_axis = np.asarray(x_axis)
val_plot = np.asarray(val_plot)
train_plot = np.asarray(train_plot)
axes = plt.gca()
axes.set_ylim([0.5,1])
plt.grid(which = 'both')
plt.plot(val_plot)
plt.plot(train_plot)

arr_test, data_ins,data_id = load_x_test("test_set_values.csv",train_col)


print(arr_train[0].columns)
print("WOW_trainabove")
print(arr_test[0].columns)
print(np.asarray(arr_train[a]).shape)
print(arr_train[a].columns)
print("WOW")



for a in range(len(arr_train)):
    test_col = arr_test[a].columns
    train_col = arr_train[a].columns
    for i in range(len(test_col)):
        if test_col[i] not in train_col:
            arr_test[a] = arr_test[a].drop(columns = test_col[i])
            # arr_test[a] = arr_test[a].drop(columns = arr_test[a].columns[i])
    
    for i in range(len(train_col)):
        if train_col[i] not in test_col:
            index_name = train_col[i]
            index = i
            arr_test[a].insert(index,index_name,data_ins)
    if(arr_test[a].columns == arr_train[a].columns).all():
        print("Similar_Columns")


y_pred = model.predict(arr_test)


status = np.argmax(y_pred, axis = 1)
status = status.reshape(len(data_ins),1)
id = data_id.to_numpy().reshape(len(data_ins),1)
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