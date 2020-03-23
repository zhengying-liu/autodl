#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-10-15

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import (Input, Dense, Dropout, Convolution2D,
                                            MaxPooling2D, ELU, Reshape, CuDNNGRU)
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model as TFModel

from CONSTANT import MAX_FRAME_NUM, IS_CUT_AUDIO, MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import ohe2cat, get_max_length, pad_seq, extract_mfcc_parallel
from models.my_classifier import Classifier


class Crnn2dLargerModel(Classifier):
    def __init__(self):
        self.max_length = None

        self._model = None
        self.is_init = False

    def preprocess_data(self, x):
        # mel-spectrogram parameters
        SR = 16000
        N_FFT = 512
        N_MELS = 96
        HOP_LEN = 256
        DURA = 21.84  # to make it 1366 frame.
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]

        # x_mel = extract_melspectrogram_parallel(x, n_mels=128, use_power_db=True)
        x_mfcc = extract_mfcc_parallel(x, n_mfcc=96)
        if self.max_length is None:
            self.max_length = get_max_length(x_mfcc)
            self.max_length = min(MAX_FRAME_NUM, self.max_length)
        x_mfcc = pad_seq(x_mfcc, pad_len=self.max_length)
        x_mfcc = x_mfcc[:, :, :, np.newaxis]
        return x_mfcc

    def init_model(self,
                   input_shape,
                   num_classes,
                   **kwargs):
        layers = 5
        filters_size = [64, 128, 256, 512, 512]
        kernel_size = (3, 3)
        pool_size = [(2, 2), (2, 2), (2, 2), (4, 1), (4, 1)]

        freq_axis = 2
        channel_axis = 3

        channel_size = 128
        min_size = min(input_shape[:2])
        melgram_input = Input(shape=input_shape)
        # x = ZeroPadding2D(padding=(0, 37))(melgram_input)

        x = Reshape((input_shape[0], input_shape[1], 1))(melgram_input)
        x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

        # Conv block 1
        x = Convolution2D(
            filters=filters_size[0],
            kernel_size=kernel_size,
            padding='same',
            name='conv1')(x)
        x = ELU()(x)
        x = BatchNormalization(axis=channel_axis, name='bn1')(x)
        x = MaxPooling2D(
            pool_size=pool_size[0],
            strides=pool_size[0],
            name='pool1')(x)
        x = Dropout(0.1, name='dropout1')(x)

        min_size = min_size // pool_size[0][0]

        for layer in range(1, layers):
            min_size = min_size // pool_size[layer][0]
            if min_size < 1:
                break
            x = Convolution2D(
                filters=filters_size[layer],
                kernel_size=kernel_size,
                padding='same',
                name='conv{}'.format(layer + 1))(x)
            x = ELU()(x)
            x = BatchNormalization(axis=channel_axis, name='bn{}'.format(layer + 1))(x)
            x = MaxPooling2D(
                pool_size=pool_size[layer],
                strides=pool_size[layer],
                name='pool{}'.format(layer + 1))(x)
            x = Dropout(0.1, name='dropout{}'.format(layer + 1))(x)

        x = Reshape((-1, channel_size))(x)

        gru_units = 32
        if num_classes > 32:
            gru_units = int(num_classes * 1.5)
        # GRU block 1, 2, output
        x = CuDNNGRU(gru_units, return_sequences=True, name='gru1')(x)
        x = CuDNNGRU(gru_units, return_sequences=False, name='gru2')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', name='output')(x)

        model = TFModel(inputs=melgram_input, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=1e-4,
            amsgrad=True)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit,
            train_loop_num, **kwargs):
        val_x, val_y = validation_data_fit
        epochs = 10
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
