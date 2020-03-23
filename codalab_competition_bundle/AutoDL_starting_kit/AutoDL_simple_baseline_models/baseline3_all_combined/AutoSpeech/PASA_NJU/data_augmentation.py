#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-26
import librosa
import numpy as np


def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    # more noise reduce the value to 0.5
    noise_amp = 0.05 * np.random.uniform() * np.amax(data)
    data = data.astype('float64') + noise_amp * \
        np.random.normal(size=data.shape[0])
    return data


def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high=5) * 1000)  # default at 500
    return np.roll(data, s_range)


def stretch(data, rate=0.8):
    """
    Streching the Sound. Note that this expands the dataset slightly
    """
    # keep the same length, drop some
    data = librosa.effects.time_stretch(data, rate)[:len(data)]
    return data


def pitch(data, sr=16000):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                       sr,
                                       n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data


def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(
        low=-0.5, high=7)  # default low = 1.5, high = 3
    return data * dyn_change


def speed_npitch(data):
    """
    speed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.2 / length_change  # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(
        np.arange(
            0, len(data), speed_fac), np.arange(
            0, len(data)), data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data
