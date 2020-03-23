#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
from functools import partial
from multiprocessing.pool import ThreadPool

import librosa
import numpy as np
from tensorflow.python.keras.preprocessing import sequence

from CONSTANT import NUM_MFCC, FFT_DURATION, HOP_DURATION
from tools import timeit, log

pool = ThreadPool(os.cpu_count())


def ohe2cat(label):
    return np.argmax(label, axis=1)


@timeit
def get_max_length(x, ratio=0.95):
    """
    Get the max length cover 95% data.
    """
    lens = [len(_) for _ in x]
    max_len = max(lens)
    min_len = min(lens)
    lens.sort()
    # TODO need to drop the too short data?
    specified_len = lens[int(len(lens) * ratio)]
    log("Max length: {}; Min length {}; 95 length {}".format(max_len, min_len, specified_len))
    return specified_len


def pad_seq(data, pad_len):
    return sequence.pad_sequences(data, maxlen=pad_len, dtype='float32', padding='post', truncating='post')


def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(pool.map(extract, data_with_index))

    results_with_index.sort(key=lambda x: x[1])

    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)

# mfcc
@timeit
def extract_mfcc(data, sr=16000, n_mfcc=NUM_MFCC):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d, sr=sr, n_mfcc=n_mfcc)
        r = r.transpose()
        results.append(r)

    return results


def extract_for_one_sample(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    r = extract(data, **kwargs)
    # for melspectrogram
    if use_power_db:
        r = librosa.power_to_db(r)

    r = r.transpose()
    return r, idx


@timeit
def extract_mfcc_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mfcc=NUM_MFCC):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.mfcc, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    results = extract_parallel(data, extract)

    return results


# zero crossings

@timeit
def extract_zero_crossing_rate_parallel(data):
    extract = partial(extract_for_one_sample, extract=librosa.feature.zero_crossing_rate, pad=False)
    results = extract_parallel(data, extract)

    return results


# spectral centroid

@timeit
def extract_spectral_centroid_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_centroid, sr=sr,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_melspectrogram_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.melspectrogram,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    results = extract_parallel(data, extract)

    return results


# spectral rolloff
@timeit
def extract_spectral_rolloff_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_rolloff,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)  # data+0.01?
    # sklearn.preprocessing.scale()
    return results


# chroma stft
@timeit
def extract_chroma_stft_parallel(data, sr=16000, n_fft=None, hop_length=None, n_chroma=12):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_stft, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_bandwidth_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_bandwidth,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_contrast_parallel(data, sr=16000, n_fft=None, hop_length=None, n_bands=6):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_contrast,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_spectral_flatness_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_flatness,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_tonnetz_parallel(data, sr=16000):
    extract = partial(extract_for_one_sample, extract=librosa.feature.tonnetz, sr=sr)
    results = extract_parallel(data, extract)
    return results


@timeit
def extract_chroma_cens_parallel(data, sr=16000, hop_length=None, n_chroma=12):
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_cens, sr=sr,
                      hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_rms_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.rms,
                      frame_length=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


@timeit
def extract_poly_features_parallel(data, sr=16000, n_fft=None, hop_length=None, order=1):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.poly_features,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    results = extract_parallel(data, extract)

    return results
