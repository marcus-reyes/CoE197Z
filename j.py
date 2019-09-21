###Reyes, Marcus

###CoE 197Z Project 1

###Kaggle-https://www.kaggle.com/c/cat-in-the-dat


#Categorical to one_hot

#https://www.datacamp.com/community/tutorials/categorical-data#encoding

#https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

import pandas as pd

import keras

import numpy as np

from numpy import genfromtxt



from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout

from keras.optimizers import adam



from sklearn import preprocessing


from sklearn.feature_extraction import FeatureHasher

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

###Data preprocessing

data = pd.read_csv("train.csv")
#For now ignore the data you don't know how to handle

#drop = ['id', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

# drop = ['id', 'nom_7','nom_8', 'nom_9']

drop = ['id','nom_9','nom_8']

h = FeatureHasher(n_features = 200, input_type = "string")


#Initialize nom_np
data['nom_9'] = data['nom_9'].astype('str')
nom_np = (h.transform(data['nom_9'].values)).todense()
    
data = data.drop(columns = drop)



#Categorical to one_hot

#https://www.datacamp.com/community/tutorials/categorical-data#encoding

one_hot = ['bin_3', 'bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5','day','month', 'nom_5','nom_6','nom_7']



#Categorical to labelled

labelled = ['nom_7', 'nom_8']


duplicate = ['bin_0','bin_1','bin_2','bin_3_F','bin_3_T','bin_4_Y','bin_4_N']
duplicatecount = 2
for i,w in enumerate(one_hot):

    data = pd.get_dummies(data, columns=[w], prefix = [w])



# for i,w in enumerate(labelled):

    # labels = data[w].astype('category').cat.categories.tolist()



    # replace_map_comp = {w: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}



    # data.replace(replace_map_comp, inplace=True)

    

    # del labels, replace_map_comp

    

    # print(data[w])


for i,w in enumerate(duplicate):
    for j in range(duplicatecount):
        data = pd.concat([data, data[w]],axis = 1)

y_train = data['target'].to_numpy()

# print(y_train.shape)

y_train = keras.utils.to_categorical(y_train, 2)



data = data.drop(columns = ['target'])


columns_list = data.columns
x = data.to_numpy()

print(nom_np.shape,"NOM_NP_SHAP")
print(x.shape)
x = np.concatenate((x,nom_np), axis = 1)
print(x.shape)

try:
    del data, nom_np
    print("cleared memory")
except:
    pass
###Normalize data to large to be one-hot-encoded

min_max_scaler = preprocessing.MinMaxScaler()

x = min_max_scaler.fit_transform(x)



# print(x[4,:])

# print(x[7,:])

x_train = x[:300000,:]

x_pretest = x[240000:,:]



y_pretest = y_train[240000:,:]

y_train = y_train[:300000,:]



###Model



hidden = 1024-128

dropout = 0.55

(trash, input_dim) = x.shape

model = Sequential()



model.add(Dense(hidden, input_dim = input_dim))

model.add(Dropout(dropout))

model.add(Activation('tanh'))



model.add(Dense(hidden,input_dim = hidden))

model.add(Dropout(dropout))

model.add(Activation('tanh'))



model.add(Dense(hidden,input_dim = hidden))

model.add(Dropout(dropout))

model.add(Activation('tanh'))



model.add(Dense(2,input_dim = hidden))

model.add(Activation('softmax'))



model.summary()



model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



###To keep track of validation error

for i in range(6):



    model.fit(x_train, y_train, epochs = 1, batch_size = 4096*4)



    score = model.evaluate(x_pretest, y_pretest, batch_size = 512)

    print("\nTest accuacy: %.4f%%" % (100.0 * score[1]))
    





###Testing
# delete_list = ['data','x_pretest','x_train','y_pretest','y_train','x']
# for i,v in enumerate(delete_list):        
try:
    del data
    print("Cleared data")
except:
    pass
    
try:
    del x
    print("Cleared x")
except:
    pass
try:
    del x_pretest,x_train,y_pretest,y_train
    print("Cleared others")
except:
    pass
 

data = pd.read_csv("test.csv")

#Initialize nom_np
data['nom_9'] = data['nom_9'].astype('str')
nom_np = (h.transform(data['nom_9'].values)).todense()
data = data.drop(columns = drop)





for i,w in enumerate(one_hot):

   data = pd.get_dummies(data, columns=[w], prefix = [w])





# for i,w in enumerate(labelled):

    # labels = data[w].astype('category').cat.categories.tolist()

    # replace_map_comp = {w: {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

    # data.replace(replace_map_comp, inplace=True)

    # del labels, replace_map_comp

    # print(data[w])


for i,w in enumerate(duplicate):
    for j in range(duplicatecount):
        data = pd.concat([data, data[w]],axis = 1)





columns_list2 = data.columns
data_ins = [0]*200000
for i in range(len(columns_list)):
    if columns_list[i] not in columns_list2:
        index_name = columns_list[i]
        index = i
data.insert(index,index_name,data_ins)
print(columns_list)
print(columns_list2)
if(columns_list == data.columns).all():
    print("equal hdeaders")
x_test = data.to_numpy()
try:
    del data,NAlist
except:
    pass
print(nom_np.shape,"NOM_NP_SHAP")
print(x_test.shape)
x_test = np.concatenate((x_test,nom_np), axis = 1)
print(x_test.shape)

try:
    del nom_np
except:
    pass

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