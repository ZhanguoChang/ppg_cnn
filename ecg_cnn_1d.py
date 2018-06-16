import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import scipy.io as sio
import pprint
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.utils import plot_model
import os

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
####
####！！！！用神经网络做预测时数据必须进行归一化处理，之后进行反归一化！！！！
####
####
#h5py.File(r'F:\temp\myh5.hdf5')
#model=load_model(r'F:\temp\myh5.hdf5')
# Generate dummy data

data=sio.loadmat(r'C:\Users\Dell\Desktop\ECG\ecg2_1.mat')
x_train,x_test,x_l,y_l=train_test_split(data['ecg2_1'],data['label_1'],test_size=0.20)
'''
num=len(data['feature1'])
num1=int(num*0.8)

x_train=data['feature1'][0:num1]
x_test=data['feature1'][num1+1:]
'''
dim=len(x_train[0])
# x_train_t=x_train.T
# x_test_t=x_test.T
a=preprocessing.MinMaxScaler()  #sklearn按列归一化
x_t=a.fit_transform(x_train)
y_t=a.transform(x_test)

# x_t=x_t.T
# y_t=y_t.T


'''
x_l=data['output_bp1'][0:num1]
y_l=data['output_bp1'][num1+1:]
'''
b=preprocessing.MinMaxScaler()
y_t1=b.fit_transform(x_l)
y_l1=b.transform(y_l)


#increase the dims
#x_t=np.expand_dims(x_t,0)
x_t=np.expand_dims(x_t,2)
#x_t(1,1,none,100)
y_t=np.expand_dims(y_t,2)
#y_t=np.expand_dims(y_t,1)
print(x_t.shape)
#y_train=keras.utils.to_categorical(y_train1, 2)
#y_test=keras.utils.to_categorical(y_test1, 2)
#print('x_test:',x_test)
#print('y_test:',y_test)

model = Sequential()
#model.add(Conv2D(64, (2, 2), activation='relu',input_shape=(100,20)))
#model.add(Dense(10, input_dim=10, activation='relu',init='uniform'))
model.add(Conv1D(64,5,strides=2,padding='same',input_shape=(dim,1)))

model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
#model.add(Conv1D(64,5,strides=2,padding='same'))
#model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
#model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Conv1D(64,5,activation='relu',strides=2,padding='same'))
#model.add(Conv1D(128,3,border_mode='same',activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Conv1D(128,5,strides=2,border_mode='same'))
model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
#model.add(Conv1D(128,3,activation='relu',strides=2,border_mode='same'))
#model.add(Dropout(0.25))
#model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
#model.add(Conv1D(64,3,strides=2,border_mode='same'))
#model.add(Dropout(0.25))

#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Flatten())
#model.output_shape
#model.add(Dense(64,activation='sigmoid'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

plot_model(model,show_shapes=True ,to_file=r'C:\Users\Dell\Desktop\test\model.png')
print(model.summary())



#print('the 7 layers weight is :',model.get_weights())

model.compile(#loss='binary_crossentropy',
              loss='mean_squared_error',
              #optimizer='rmsprop',
              optimizer='sgd',
                #optimizer='sgd',
              metrics=['accuracy']
     )


x_t=np.array(x_t)
l1=len(x_t)
x_t=x_t.reshape(l1,dim,1)
l2=len(y_t)
y_t=np.array(y_t)
y_t=y_t.reshape(l2,dim,1)

print ('----Training----')
# 训练过程

#earlystoping=keras.callbacks.EarlyStopping(monitor='loss',mode='auto')
model.fit(x_t, y_t1,
           epochs=20,
           verbose=2,
           #callbacks=[earlystoping],
           batch_size=64)
escore = model.evaluate(y_t, y_l1, batch_size=64)
print(escore)
#print('y_test:',y_l1)
real_test=b.inverse_transform(y_l1)
print('real_test:',real_test[0:10])
#predict = model.predict(x_test)
#x_test=a.inverse_transform(x_test)
predict1=model.predict_on_batch(y_t)
real_pre=b.inverse_transform(predict1)
#print('predict:',predict)
print('predict1:',real_pre[0:10])
#print('predict:',predict)
#print('###########')
#keras.metrics.binary_accuracy(y_test, predict)
'''
json_string=model.to_json()
print(json_string)
#print(model.save_weights())
h5py.File(r'F:\temp\myh5.hdf5','w')
model.save(r'F:\temp\myh5.hdf5')
'''

plt.figure()
plt.plot(real_test)
plt.plot(real_pre)
plt.show()



