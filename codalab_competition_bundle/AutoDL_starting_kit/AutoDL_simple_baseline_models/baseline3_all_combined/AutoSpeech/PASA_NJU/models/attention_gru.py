#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/15 22:44
# @Author:  Mecthew

import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import (SpatialDropout1D, Input, Bidirectional, GlobalMaxPool1D, GlobalAvgPool1D,
                                            Concatenate,
                                            Dense, Dropout, CuDNNLSTM)
from tensorflow.python.keras.models import Model as TFModel

from CONSTANT import MAX_FRAME_NUM
from data_process import ohe2cat, extract_mfcc_parallel, get_max_length, pad_seq
from models.attention import Attention
from models.my_classifier import Classifier
from tools import log


class AttentionGru(Classifier):
    def __init__(self):
        # clear_session()
        log('init AttentionGru')
        self.max_length = None
        self._model = None
        self.is_init = False

    def preprocess_data(self, x):
        # if IS_CUT_AUDIO:
        #     x = [sample[0:MAX_AUDIO_DURATION*AUDIO_SAMPLE_RATE] for sample in x]
        # extract mfcc
        x = extract_mfcc_parallel(x, n_mfcc=96)
        if self.max_length is None:
            self.max_length = get_max_length(x)
            self.max_length = min(MAX_FRAME_NUM, self.max_length)
        x = pad_seq(x, pad_len=self.max_length)
        return x

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        inputs = Input(shape=input_shape)
        # bnorm_1 = BatchNormalization(axis=-1)(inputs)
        x = Bidirectional(CuDNNLSTM(96, name='blstm1',
                                    return_sequences=True),
                          merge_mode='concat')(inputs)
        # activation_1 = Activation('tanh')(lstm_1)
        x = SpatialDropout1D(0.1)(x)
        x = Attention(8, 16)([x, x, x])
        x1 = GlobalMaxPool1D()(x)
        x2 = GlobalAvgPool1D()(x)
        x = Concatenate(axis=-1)([x1, x2])
        x = Dense(units=128, activation='elu')(x)
        x = Dense(units=64, activation='elu')(x)
        x = Dropout(rate=0.4)(x)
        outputs = Dense(units=num_classes, activation='softmax')(x)

        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0002,
            amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, round_num, **kwargs):
        val_x, val_y = validation_data_fit
        if round_num >= 2:
            epochs = 10
        else:
            epochs = 5
        patience = 2
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience)]

        self._model.fit(train_x, ohe2cat(train_y),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_x, ohe2cat(val_y)),
                        verbose=1,  # Logs once per epoch.
                        batch_size=32,
                        shuffle=True)

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
