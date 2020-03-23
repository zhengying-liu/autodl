#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/5 10:35
# @Author:  Mecthew

import numpy as np
from sklearn.linear_model import logistic
from sklearn.preprocessing import StandardScaler

from CONSTANT import MAX_AUDIO_DURATION, AUDIO_SAMPLE_RATE
from data_process import (extract_melspectrogram_parallel)
from data_process import ohe2cat
from models.my_classifier import Classifier
from tools import timeit


# Consider use LR as the first model because it can reach high point at
# first loop
class LogisticRegression(Classifier):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        # clear_session()
        self.max_length = None
        self._model = None
        self.is_init = False

    def init_model(self,
                   kernel,
                   max_iter=200,
                   C=1.0,
                   **kwargs):
        self._model = logistic.LogisticRegression(
            C=C, max_iter=max_iter, solver='liblinear', multi_class='auto')
        self.is_init = True

    @timeit
    def preprocess_data(self, x):
        # cut down
        x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in x]
        # extract mfcc
        # x_mfcc = extract_mfcc_parallel(x, n_mfcc=63)
        x_mel = extract_melspectrogram_parallel(
            x, n_mels=40, use_power_db=True)
        # x_chroma_stft = extract_chroma_stft_parallel(x, n_chroma=12)
        # x_rms = extract_rms_parallel(x)
        # x_contrast = extract_spectral_contrast_parallel(x, n_bands=6)
        # x_flatness = extract_spectral_flatness_parallel(x)
        # x_polyfeatures = extract_poly_features_parallel(x, order=1)
        # x_cent = extract_spectral_centroid_parallel(x)
        # x_bw = extract_bandwidth_parallel(x)
        # x_rolloff = extract_spectral_rolloff_parallel(x)
        # x_zcr = extract_zero_crossing_rate_parallel(x)

        x_feas = []
        for i in range(len(x_mel)):
            mel = np.mean(x_mel[i], axis=0).reshape(-1)
            mel_std = np.std(x_mel[i], axis=0).reshape(-1)
            # mel = np.mean(x_mel[i], axis=0).reshape(-1)
            # mel_std = np.std(x_mel[i], axis=0).reshape(-1)
            # chroma_stft = np.mean(x_chroma_stft[i], axis=0).reshape(-1)
            # chroma_stft_std = np.std(x_chroma_stft[i], axis=0).reshape(-1)
            # rms = np.mean(x_rms[i], axis=0).reshape(-1)
            # contrast = np.mean(x_contrast[i], axis=0).reshape(-1)
            # contrast_std = np.std(x_contrast[i], axis=0).reshape(-1)
            # flatness = np.mean(x_flatness[i], axis=0).reshape(-1)
            # polyfeatures = np.mean(x_polyfeatures[i], axis=0).reshape(-1)
            # cent = np.mean(x_cent[i], axis=0).reshape(-1)
            # cent_std = np.std(x_cent[i], axis=0).reshape(-1)
            # bw = np.mean(x_bw[i], axis=0).reshape(-1)
            # bw_std = np.std(x_bw[i], axis=0).reshape(-1)
            # rolloff = np.mean(x_rolloff[i], axis=0).reshape(-1)
            # zcr = np.mean(x_zcr[i], axis=0).reshape(-1)
            x_feas.append(np.concatenate([mel, mel_std], axis=-1))
            # x_feas.append(np.concatenate([mfcc, mel, contrast, bw, cent, mfcc_std, mel_std, contrast_std, bw_std, cent_std]))
        x_feas = np.asarray(x_feas)

        scaler = StandardScaler()
        X = scaler.fit_transform(x_feas[:, :])
        # log(   'x_feas shape: {X.shape}\n'
        #        'x_feas[0]: {X[0]}')
        return X

    def fit(self, x_train, y_train, *args, **kwargs):
        self._model.fit(x_train, ohe2cat(y_train))

    def predict(self, x_test, batch_size=32):
        return self._model.predict_proba(x_test)
