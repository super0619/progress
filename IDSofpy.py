from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import sys
import sklearn




import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


#pd.set_option('display.width', 1000) # 设置字符显示宽度
#pd.set_option('display.max_rows', None) # 设置显示最大行


#print(pd.__version__)
#print(np.__version__)
#print(sys.version)
#print(sklearn.__version__)


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
# these have already been removed.
df = pd.read_csv("KDDTrain+_2.csv", header=None, names = col_names)
df_test = pd.read_csv("KDDTest+_2.csv", header=None, names = col_names)

# shape, this gives the dimensions of the dataset
print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)

#print(df.head(5))
#print(df.describe())

#print('Label distribution Training set:')
#print(df['label'].value_counts())
#print()
#print('Label distribution Test set:')
#print(df_test['label'].value_counts())

print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())

print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag'] 
 # Get the categorical values into a 2D numpy array
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
print(df_categorical_values.head())       

# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

#do same for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2
print(testdumcols)

df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc.head())
# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

print(df_cat_data.head())

trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
print(difference)

print(testdf_cat_data.shape)

for col in difference:
    testdf_cat_data[col] = 0
print(testdf_cat_data.shape)

newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)

print(newdf.shape)
print(newdf_test.shape)

# take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test

#print(newdf.shape)
#print(newdf_test.shape)


#print(newdf['label'])

#adjust1,调整train数据集里面多出来的的6行至最后面，然后将标签和数据分开，本来分成了四个类，但现在放在一个里面就好了

yoftrain=newdf.label

col8001=newdf['service_http_8001']
colharvest=newdf['service_harvest']
colaol=newdf['service_aol']
col27=newdf['service_http_2784']
colui=newdf['service_urh_i']
colri=newdf['service_red_i']
xoftrain=newdf.drop(['label','service_http_8001','service_harvest','service_aol','service_http_2784','service_urh_i','service_red_i'],1)

xoftrain['service_http_8001'] =col8001
xoftrain['service_harvest'] =colharvest
xoftrain['service_aol'] =colaol
xoftrain['service_http_2784'] =col27
xoftrain['service_urh_i'] =colui
xoftrain['service_red_i'] =colri


xoftest =newdf_test.drop('label',1)
yoftest =newdf_test.label


colNames=list(xoftrain)
colNames_test=list(xoftest)

print(colNames)
print(colNames_test)

from sklearn import preprocessing
scaler1 = preprocessing.StandardScaler().fit(xoftrain)
xoftrain=scaler1.transform(xoftrain) 

scaler5 = preprocessing.StandardScaler().fit(xoftest)
xoftest=scaler5.transform(xoftest) 

import keras

from keras import backend as K #转换为张量
xoftrain1 = K.cast_to_floatx(xoftrain)
print(xoftrain1)
yoftrain1 = K.cast_to_floatx(yoftrain)
print(yoftrain1)

xoftest1 = K.cast_to_floatx(xoftest)
yoftest1 = K.cast_to_floatx(yoftest)

def to_image(array0):
    array0 = np.array(array0)
    array0 = np.column_stack((array0, np.zeros((array0.shape[0], 22))))
    return array0.reshape((array0.shape[0],12,12,1))

train_list = [xoftrain]

def to_label(array0):
    array0 = np.array(array0)
    return array0.reshape((array0.shape[0],1))

train_images, train_labels, test_images, test_labels = to_image(xoftrain1), to_label(yoftrain1), to_image(xoftest1), to_label(yoftest1)

print(train_images.shape)
print(train_labels.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(12, 12, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))   #softmax

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=0, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)