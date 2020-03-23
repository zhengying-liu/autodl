#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/24 15:12
# @Author:  Mecthew

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.layers import (Activation, Flatten, Conv2D,
                                            MaxPooling2D, BatchNormalization)
from tensorflow.python.keras.layers import (Conv1D, Dense, Dropout, MaxPool1D)
from tensorflow.python.keras.models import Sequential

from CONSTANT import IS_CUT_AUDIO, MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import ohe2cat, extract_mfcc, get_max_length, pad_seq, extract_mfcc_parallel
from models.my_classifier import Classifier
from tools import log
from tools import timeit


class CnnModel2D(Classifier):
    def __init__(self):
        # clear_session()
        self.max_length = None

        self._model = None
        self.is_init = False

    def init_model(self,
                   input_shape,
                   num_classes,
                   max_layer_num=5,
                   **kwargs):
        # FIXME: keras sequential model is better than keras functional api,
        # why???
        model = Sequential()
        min_size = min(input_shape[:2])
        for i in range(max_layer_num):
            if i == 0:
                model.add(
                    Conv2D(
                        64,
                        3,
                        input_shape=input_shape,
                        padding='same'))
            else:
                model.add(Conv2D(64, 3, padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            min_size //= 2
            if min_size < 2:
                break

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(rate=0.5))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6)

        optimizer = tf.keras.optimizers.Adam()
        # optimizer = optimizers.SGD(lr=1e-3, decay=2e-4, momentum=0.9, clipvalue=5)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        self.is_init = True
        self._model = model

    def preprocess_data(self, x):
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]
        # extract mfcc
        x = extract_mfcc_parallel(x, n_mfcc=96)
        if self.max_length is None:
            self.max_length = get_max_length(x)
        x = pad_seq(x, self.max_length)

        # if self.scaler is None:
        #     self.scaler = []
        #     for i in range(x.shape[2]):
        #         self.scaler.append(StandardScaler().fit(x[:, :, i]))
        # for i in range(x.shape[2]):
        #     x[:, :, i] = self.scaler[i].transform(x[:, :, i])

        # feature scale
        # if self.mean is None or self.std is None:
        #     self.mean = np.mean(x)
        #     self.std = np.std(x)
        #     x = (x - self.mean) / self.std

        # s0, s1, s2 = x.shape[0], x.shape[1], x.shape[2]
        # x = x.reshape(s0 * s1, s2)
        # if not self.scaler:
        #     self.scaler = MinMaxScaler().fit(x)
        # x = self.scaler.transform(x)
        # x = x.reshape(s0, s1, s2)

        # 4 dimension?
        # (120, 437, 24) to (120, 437, 24, 1)
        # 120 is the number of instance
        # 437 is the max length
        # 24 frame in mfcc
        # log(f"max {np.max(x)} min {np.min(x)} mean {np.mean(x)}")

        x = x[:, :, :, np.newaxis]
        return x

    def fit(self, train_x, train_y, validation_data_fit,
            train_loop_num, **kwargs):
        val_x, val_y = validation_data_fit

        # if train_loop_num == 1:
        #     patience = 2
        #     epochs = 8
        # elif train_loop_num == 2:
        #     patience = 3
        #     epochs = 10
        # elif train_loop_num < 10:
        #     patience = 4
        #     epochs = 16
        # elif train_loop_num < 15:
        #     patience = 4
        #     epochs = 24
        # else:
        #     patience = 8
        #     epochs = 32

        epochs = 3
        patience = 2

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)]

        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)


# Input of CNN1D is speech raw data， shape of each sample is
# (sample_rate*default_duration, 1)
class CnnModel1D(Classifier):
    def __init__(self):
        clear_session()
        self.max_length = None
        self.mean = None
        self.std = None

        self._model = None
        self.is_init = False

    def preprocess_data(self, x):
        # extract mfcc
        x = extract_mfcc(x)
        if self.max_length is None:
            self.max_length = get_max_length(x)
        x = pad_seq(x, self.max_length)
        # feature scale
        if self.mean is None or self.std is None:
            self.mean = np.mean(x)
            self.std = np.std(x)
            x = (x - self.mean) / self.std
        # calculate mean of mfcc
        x = np.mean(x, axis=-1)
        x = x[:, :, np.newaxis]
        return x

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        # New model
        model = Sequential()
        model.add(
            Conv1D(256, 8, padding='same', input_shape=(input_shape[0], 1)))  # X_train.shape[0] = No. of Columns
        model.add(Activation('relu'))
        model.add(Conv1D(256, 8, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPool1D(pool_size=8))
        model.add(Conv1D(128, 8, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 8, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 8, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 8, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(MaxPool1D(pool_size=8))
        model.add(Conv1D(64, 8, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(64, 8, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(num_classes))  # Target class number
        model.add(Activation('softmax'))
        # opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=1e-6, nesterov=False)
        # opt = keras.optimizers.Adam(lr=0.0001)
        opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=['acc'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit,
            train_loop_num, **kwargs):
        val_x, val_y = validation_data_fit
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3)]
        epochs = 10 if train_loop_num == 1 else 30
        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)


# use raw wave data as input
class CnnModelRawData(Classifier):
    def __init__(self):
        clear_session()
        self.max_length = None

        self._model = None
        self.is_init = False

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        model = Sequential()
        model.add(Conv2D(100, (3, 1), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 1), strides=2, padding='same'))

        model.add(Conv2D(64, (3, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1), strides=2, padding='same'))

        model.add(Conv2D(128, (3, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1), strides=2, padding='same'))

        model.add(Conv2D(128, (3, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1), strides=2, padding='same'))

        model.add(Conv2D(128, (3, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1), strides=2, padding='same'))

        model.add(Conv2D(128, (3, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 1), strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(1024, 'relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        optimizer = optimizers.SGD(
            lr=1e-4, decay=5e-5, momentum=0.9, clipnorm=4)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()

        self._model = model
        self.is_init = True

    @timeit
    def preprocess_data(self, x):
        if self.max_length is None:
            self.max_length = 8000 * 2
        x_resample = []
        for sample in x:
            data = librosa.core.resample(
                sample, orig_sr=16000, target_sr=8000, res_type='scipy')
            if len(data) > self.max_length:
                embedded_data = data[:self.max_length]
            elif len(data) < self.max_length:
                embedded_data = np.zeros(self.max_length)
                offset = np.random.randint(
                    low=0, high=self.max_length - len(data))
                embedded_data[offset:offset + len(data)] = data
            else:
                # nothing to do here
                embedded_data = data
            embedded_data /= (np.percentile(embedded_data, 95) + 0.001)
            x_resample.append(embedded_data)

        x_resample = np.array(x_resample)[:, :, np.newaxis, np.newaxis]
        return x_resample

    def fit(self, train_x, train_y, validation_data_fit,
            train_loop_num, **kwargs):
        val_x, val_y = validation_data_fit
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3)]
        epochs = 10 if train_loop_num == 1 else 30
        log('train_x: {}; train_y: {}'.format(train_x.shape, train_y.shape))
        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
