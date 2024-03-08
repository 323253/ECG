import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import re
from scipy.io import loadmat
from tensorflow import keras
from keras.utils import to_categorical#独热编码
import matplotlib.pyplot as plt

loadata1=loadmat(r"D:\毕设数据\input_lables\input_data.mat")
loadata2=loadmat(r"D:\毕设数据\input_lables\label_data_hot.mat")

input_data=loadata1['input_data']
label_data=loadata2['label_data_hot']


def CNN_Model(input_shape):
    model = models.Sequential()
    #原参考32,32,48
    model.add(layers.Conv1D(6, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(6, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(12, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))  # 4个类别
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


input_data=np.expand_dims(input_data,axis=-1)#单通道三维
print('input_data,label_data,finish')
# label_data=np.squeeze(label_data)#转一维
# print("Shape of input_data:", input_data.shape)
# print("Shape of label_data:", label_data.shape)


X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)
print("split")

input_shape = input_data.shape[1:]  # 根据实际情况调整
model = CNN_Model(input_shape)
print("CNN_Model finish")

# predictions = model.predict(X_train)
# print(predictions)
# print(predictions.shape)#(8788, 3)
#画出模型
keras.utils.plot_model(model, "CNN_in_out_model121224.png", show_shapes=True)
#训练
RESULT=model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
print("train")
loss_=RESULT.history['loss']
val_loss_=RESULT.history['val_loss']
accuracy=RESULT.history['accuracy']

# print("训练损失：",loss_)
# print("验证损失：",val_loss_)
echos=range(1,len(loss_)+1)
plt.plot(echos,loss_,'r',label='loss_')
plt.plot(echos,val_loss_,'b',label='val_loss_')
plt.title('LOSS AND VAL_LOSS')
plt.xlabel('echos')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(echos,accuracy,'g',label='accuracy')
plt.title('ACCURACY')
plt.xlabel('echos')
plt.ylabel('accuracy')
plt.legend()
plt.show()


