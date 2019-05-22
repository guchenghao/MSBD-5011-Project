#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/CodeWarehouse/MSBD5011_Project/method2/CNN_training_model.py
# Project: /Users/guchenghao/CodeWarehouse/MSBD5011_Project/method2
# Created Date: Tuesday, May 7th 2019, 8:12:43 pm
# Author: Harold Gu
# -----
# Last Modified: Tuesday, 7th May 2019 8:12:43 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2019 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np
import pandas as pd
from os.path import join as pjoin

import matplotlib.pyplot as plt
import cv2 as cv
from skimage.io import imshow, imread, imsave

import warnings
warnings.filterwarnings('ignore')

work_dir = './data'
file_path = './data/faces/'
label_data = './data/labels.csv'
WIDTH, HEIGHT = 48, 28  # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT  # dimension of feature vector

# ! 读入预处理文件
data_img_vec = pd.read_csv('/Users/guchenghao/CodeWarehouse/MSBD5011_Project_Code/data/face_mouth_data.csv', index_col=0)
pd_face = pd.read_csv('/Users/guchenghao/CodeWarehouse/MSBD5011_Project_Code/data/face_mouth_data_index.csv', index_col=0)

# ! 处理图像数据
data_img_vec_norm = data_img_vec / 255
data_img_vec_norm.head()
data_imgs = data_img_vec_norm.values.reshape(-1, HEIGHT, WIDTH, 1)
pd_face['is_smile'].value_counts()
data_labels = to_categorical(pd_face['is_smile'].values, num_classes=2)


random_seed = 2019
X_train, X_val, Y_train, Y_val = train_test_split(
    data_imgs, data_labels, test_size=0.1, random_state=random_seed)

# ! 搭建CNN模型的架构
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same',
                 activation='relu', input_shape=(28, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))


# ! 优化器
optimizer = adam()
model.compile(optimizer=optimizer,
              loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ! 读入图像数据
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False)
datagen.fit(X_train)


# ! 早停防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
# ! 学习率衰减
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

# ! 设置训练参数并开始训练
epochs = 50
batch_size = 128
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(X_val, Y_val),
                              verbose=2,
                              steps_per_epoch=10,
                              callbacks=[learning_rate_reduction])


# ! 利用matplotlib画出训练曲线图
fig_res, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label='loss')
ax[0].plot(history.history['val_loss'], color='r',
           label='val_loss', axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label='acc')
ax[1].plot(history.history['val_acc'], color='r', label='val_acc')
legend = ax[1].legend(loc='best', shadow=True)
fig_res.show()

# ! 保存模型
model.save('./model/model_smile_rec_by_nn.h5')
