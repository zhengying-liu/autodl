#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from .data_manager import DataManager
from .model_manager import ModelManager
from tools import log, timeit

config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)


class Model(object):
    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_loop_num = 0
        log('Metadata: {}'.format(self.metadata))

        self.data_manager = None
        self.model_manager = None

        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

    @timeit
    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.

        :param train_dataset: tuple, (train_x, train_y)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        self.train_loop_num += 1

        if self.train_loop_num == 1:
            self.data_manager = DataManager(self.metadata, train_dataset)
            self.model_manager = ModelManager(self.metadata, self.data_manager)

        self.model_manager.fit(train_loop_num=self.train_loop_num)

        if self.train_loop_num > 500:
            self.done_training = True

    @timeit
    def test(self, test_x, remaining_time_budget=None):
        """
        :param test_x: list of vectors, input test speech raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        # extract test feature
        pred_y = self.model_manager.predict(test_x, is_final_test_x=True)

        result = pred_y
        return result


if __name__ == '__main__':
    from ingestion.dataset import AutoSpeechDataset
    D = AutoSpeechDataset(os.path.join("../sample_data/DEMO", 'DEMO.data'))
    D.read_dataset()
    m = Model(D.get_metadata())
    m.train(D.get_train())
    m.test(D.get_test())
    m.train(D.get_train())
