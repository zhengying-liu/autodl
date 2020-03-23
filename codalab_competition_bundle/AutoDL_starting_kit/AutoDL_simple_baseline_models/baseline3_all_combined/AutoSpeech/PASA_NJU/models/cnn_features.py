#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/5 16:43
# @Author:  Mecthew

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import (Activation, Flatten, Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D,
                                            MaxPooling2D, BatchNormalization, Concatenate)
from tensorflow.python.keras.layers import (
    Input, Dense, Dropout)
from tensorflow.python.keras.models import Model as TFModel

from CONSTANT import *
from data_process import ohe2cat, get_max_length, pad_seq, extract_mfcc_parallel, extract_melspectrogram_parallel
from models.my_classifier import Classifier
from tools import timeit


class CnnFeatures(Classifier):
    def __init__(self):
        # clear_session()
        self.max_length = None

        self._model = None
        self.is_init = False

    @timeit
    def preprocess_data(self, x):
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]
        # extract mfcc
        x_mfcc = extract_mfcc_parallel(x, n_mfcc=64)
        x_mel = extract_melspectrogram_parallel(
            x, n_mels=64, use_power_db=True)
        if self.max_length is None:
            self.max_length = get_max_length(x_mfcc)
            self.max_length = min(MAX_FRAME_NUM, self.max_length)
        x_mfcc = pad_seq(x_mfcc, self.max_length)
        x_mel = pad_seq(x_mel, self.max_length)
        x_feas = np.concatenate([x_mfcc, x_mel], axis=-1)
        x_feas = x_feas[:, :, :, np.newaxis]
        # x_mel = pad_seq(x_mel, self.max_length)
        # x_mel = x_mel[:, :, :, np.newaxis]
        return x_feas

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        # FIXME: keras sequential model is better than keras functional api,
        # why???
        inputs = Input(shape=input_shape)
        # dropout0 = SpatialDropout2D(rate=0.1, data_format='channels_last')(inputs)
        min_size = min(input_shape[:2])
        pool_l = None
        for i in range(5):
            if i == 0:
                conv_l = Conv2D(
                    64,
                    3,
                    input_shape=input_shape,
                    padding='same',
                    data_format='channels_last')(inputs)
            else:
                conv_l = Conv2D(64, 3, padding='same')(pool_l)
            activation_l = Activation('relu')(conv_l)
            bn_l = BatchNormalization()(activation_l)
            pool_l = MaxPooling2D(pool_size=(2, 2))(bn_l)
            min_size //= 2
            if min_size < 2:
                break

        avgpool_l = GlobalAveragePooling2D(data_format='channels_last')(pool_l)
        maxpool_l = GlobalMaxPooling2D(data_format='channels_last')(pool_l)
        concat = Concatenate()([avgpool_l, maxpool_l])
        flatten = Flatten()(concat)
        bn1 = BatchNormalization()(flatten)
        dense1 = Dense(256, activation='relu')(bn1)
        bn2 = BatchNormalization()(dense1)
        dropout1 = Dropout(rate=0.5)(bn2)
        outputs = Dense(num_classes, activation='softmax')(dropout1)

        model = TFModel(inputs=inputs, outputs=outputs)
        # optimizer = tf.keras.optimizers.Adadelta()
        optimizer = tf.keras.optimizers.Adam()
        # optimizer = optimizers.SGD(lr=1e-3, decay=2e-4, momentum=0.9, clipvalue=5)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        self.is_init = True
        self._model = model

    def fit(self, train_x, train_y, validation_data_fit,
            train_loop_num, **kwargs):
        val_x, val_y = validation_data_fit
        epochs = 5
        patience = 2
        batch_size = 32
        # over_batch = len(train_x) % batch_size
        # append_idx = np.random.choice(np.arange(len(train_x)), size=batch_size-over_batch, replace=False)
        # train_x = np.concatenate([train_x, train_x[append_idx]], axis=0)
        # train_y = np.concatenate([train_y, train_y[append_idx]], axis=0)

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)]

        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=batch_size,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
