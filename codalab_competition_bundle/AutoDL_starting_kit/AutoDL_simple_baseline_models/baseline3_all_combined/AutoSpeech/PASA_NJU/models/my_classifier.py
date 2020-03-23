#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import abc


class Classifier(object):
    @abc.abstractmethod
    def init_model(self, **kwargs):
        pass

    @abc.abstractmethod
    def preprocess_data(self, x):
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, test_x):
        pass
