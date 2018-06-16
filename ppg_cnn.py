import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import scipy.io as sio
import pprint
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.utils import plot_model
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
start_time=datetime.datetime.now()
#显存配置
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction=0.8 #占用80%的显存
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
#加载数据
bp_h=[]
bp_f=[]
path='F:\\ecg_data\\data_test'
file=os.listdir(path)
for f in file:

    data=sio.loadmat(path+'\\'+f)
    data=data['fea']
    num=len(data)
    for i in range(num):
        bp_high=data[i][1][0][0]
        bp_feature=data[i][0]
        bp_h.append(bp_high)
        bp_f.append(bp_feature)

x_train,y_test,x_train_label,y_test_label=train_test_split(bp_f,bp_h,test_size=0.20)

x_train=np.expand_dims(x_train,3)
y_test=np.expand_dims(y_test,3)
x_train_label=np.array(x_train_label)
y_test_label=np.array(y_test_label)
x_train_label=x_train_label.reshape(-1,1)
y_test_label=y_test_label.reshape(-1,1)

bp=preprocessing.MinMaxScaler()
x_label=bp.fit_transform(x_train_label)
y_label=bp.transform(y_test_label)

#模型构建
model = Sequential()
#model.add(Conv2D(64, (2, 2), activation='relu',input_shape=(100,20)))
#model.add(Dense(10, input_dim=10, activation='relu',init='uniform'))
model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2),padding='same',init='uniform',input_shape=(1000,100,1)))

model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2), padding='same'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Conv2D(64,kernel_size=(5,5),strides=(2,2),padding='same',activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2), padding='same'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Conv2D(64,kernel_size=(5,5),strides=(2,2),padding='same',activation='relu'))
#model.add(Conv1D(128,3,border_mode='same',activation='relu'))
model.add(Dropout(0.25))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2), padding='same'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Conv2D(64,kernel_size=(3,3),strides=(2,2),border_mode='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2), padding='same'))
#model.add(Conv1D(64,3,strides=2,border_mode='same'))
#model.add(Dropout(0.25))

#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Flatten())
#model.output_shape
#model.add(Dense(64,activation='sigmoid'))
#model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one'))
model.add(Dense(128))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(32))
model.add(Dense(1))

#plot_model(model,show_shapes=True ,to_file=r'C:\Users\Dell\Desktop\test\model_ppg.png')
print(model.summary())

#模型训练
sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)

model.compile(#loss='binary_crossentropy',
              loss='mean_squared_error',
              #optimizer='rmsprop',
              optimizer='sgd',
                #optimizer='sgd',
              metrics=['accuracy']
     )
earlystoping=keras.callbacks.EarlyStopping(monitor='loss',mode='auto')
model.fit(x_train, x_label,
          #callbacks=[earlystoping],
           verbose=2,
           epochs=100,
           batch_size=32)

#结果显示
#为防止OOM，batch_size不宜设置过大
escore = model.evaluate(y_test, y_label, batch_size=32)
print(escore)

real_bp=bp.inverse_transform(y_label)
print('real_bp:',real_bp[0:10])

#predict=model.predict_on_batch(y_test)
predict=model.predict(y_test,batch_size=16)
predict_bp=bp.inverse_transform(predict)
print('predict_bp:',predict_bp[0:10])

error=abs(real_bp-predict_bp)/real_bp
err=sum(error)/len(error)
print('error:',err)
mse=sum(abs(real_bp-predict_bp)**2)/len(real_bp)
print('mse:',mse)

plt.figure()
plt.plot(real_bp, 'o-', label='real_bp')
plt.plot(predict_bp, '*-', label='predict')
plt.legend(loc='upper left')
plt.show()

end_time=datetime.datetime.now()
print('run time:',(end_time-start_time).seconds//60)
#save model
json_string=model.to_json()
print(json_string)
#print(model.save_weights())
h5py.File(r'F:\temp\ppg_cnn_0408.hdf5','w')
model.save(r'F:\temp\ppg_cnn_0408.hdf5')
'''
#download model
h5py.File(r'F:\temp\ppg_cnn_0408.hdf5')
model=load_model(r'F:\temp\ppg_cnn_0408.hdf5')
'''
