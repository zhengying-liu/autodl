#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/24 15:03
# @Author:  Mecthew

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from CONSTANT import MAX_AUDIO_DURATION, IS_CUT_AUDIO, AUDIO_SAMPLE_RATE
from data_process import (extract_chroma_stft_parallel, extract_spectral_centroid_parallel, extract_mfcc_parallel,
                          extract_spectral_rolloff_parallel, extract_melspectrogram_parallel,
                          extract_zero_crossing_rate_parallel,
                          extract_bandwidth_parallel, extract_spectral_contrast_parallel,
                          extract_spectral_flatness_parallel)
from data_process import ohe2cat
from models.my_classifier import Classifier
from tools import timeit


class SvmModel(Classifier):
    def __init__(self):
        # TODO: init model, consider use CalibratedClassifierCV
        # clear_session()
        self.max_length = None
        self._model = None
        self.is_init = False

    def init_model(self,
                   kernel,
                   max_iter=800,
                   C=1.0,
                   **kwargs):
        self._model = SVC(C=C, kernel=kernel, max_iter=100)
        self.is_init = True

    @timeit
    def preprocess_data(self, x):
        if IS_CUT_AUDIO:
            x = [sample[0:MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE]
                 for sample in x]
        # extract mfcc
        x_mfcc = extract_mfcc_parallel(x, n_mfcc=20)
        x_mel = extract_melspectrogram_parallel(
            x, n_mels=20, use_power_db=True)
        x_chroma_stft = extract_chroma_stft_parallel(x, n_chroma=12)
        # x_rms = extract_rms_parallel(x)
        x_contrast = extract_spectral_contrast_parallel(x, n_bands=6)
        x_flatness = extract_spectral_flatness_parallel(x)
        # x_polyfeatures = extract_poly_features_parallel(x, order=1)
        x_cent = extract_spectral_centroid_parallel(x)
        x_bw = extract_bandwidth_parallel(x)
        x_rolloff = extract_spectral_rolloff_parallel(x)
        x_zcr = extract_zero_crossing_rate_parallel(x)

        x_feas = []
        for i in range(len(x_mfcc)):
            mfcc = np.mean(x_mfcc[i], axis=0).reshape(-1)
            mel = np.mean(x_mel[i], axis=0).reshape(-1)
            chroma_stft = np.mean(x_chroma_stft[i], axis=0).reshape(-1)
            # rms = np.mean(x_rms[i], axis=0).reshape(-1)
            contrast = np.mean(x_contrast[i], axis=0).reshape(-1)
            flatness = np.mean(x_flatness[i], axis=0).reshape(-1)
            # polyfeatures = np.mean(x_polyfeatures[i], axis=0).reshape(-1)
            cent = np.mean(x_cent[i], axis=0).reshape(-1)
            bw = np.mean(x_bw[i], axis=0).reshape(-1)
            rolloff = np.mean(x_rolloff[i], axis=0).reshape(-1)
            zcr = np.mean(x_zcr[i], axis=0).reshape(-1)
            x_feas.append(np.concatenate([mfcc, mel, chroma_stft, contrast, flatness,
                                          cent, bw, rolloff, zcr], axis=-1))
        x_feas = np.asarray(x_feas)

        scaler = StandardScaler()
        X = scaler.fit_transform(x_feas[:, :])
        return X

    def fit(self, x_train, y_train, *args, **kwargs):
        self._model.fit(x_train, ohe2cat(y_train))

    def predict(self, x_test, batch_size=32):
        return self._model.decision_function(x_test)
