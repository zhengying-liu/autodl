#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import gc
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.python.keras import backend as K

from CONSTANT import CLASS_NUM, MODEL_FIRST_MAX_RUN_LOOP, FIRST_ROUND_DURATION, SECOND_ROUND_DURATION
from models import *  # import all models and model_name constant
from models.crnn2d import Crnn2dModel
from models.crnn2d_larger import Crnn2dLargerModel
from models.crnn2d_vgg import Crnn2dVggModel
from models.my_classifier import Classifier
from tools import log
from data_process import ohe2cat


def auc_metric(solution, prediction):
    if solution.sum(axis=0).min() == 0:
        return np.nan
    auc = roc_auc_score(solution, prediction, average='macro')
    return np.mean(auc * 2 - 1)


def acc_metric(solution, prediction):
    if solution.sum(axis=0).min() == 0:
        return np.nan
    acc = accuracy_score(solution, prediction)
    return acc


class ModelManager(Classifier):
    def __init__(self,
                 meta,
                 data_manager,
                 keep_num=5,
                 each_model_keep_num=3,
                 each_model_top_k=2,
                 patience=3,
                 auc_threshold=0.5,
                 *args,
                 **kwargs):
        self.metadata = meta
        self._data_manager = data_manager

        self._keep_num = keep_num
        self._each_model_keep_num = each_model_keep_num
        self._each_model_top_k = each_model_top_k
        self._patience = patience
        self._not_rise_num = 0

        self._input_shape = None
        self._model = None
        self._model_name = None
        self._last_model_name = None
        self._cur_model_run_loop = 0
        self._model_num = 0
        self._model_idx = 0
        self._round_num = 0

        self._val_set = None
        self._test_x = None

        self._use_new_train = False
        self._is_reset_model = False

        self._use_mfcc = True
        self._is_nedd_30s = False
        self._use_mel_round = None

        self._k_best_predicts = [-1] * self._keep_num
        self._k_best_auc = [-1.1] * self._keep_num
        self._each_model_best_predict = {}
        self._each_model_best_auc = {}
        self._cur_model_max_auc = -1
        self._auc_threshold = auc_threshold

        self._num_classes = self.metadata[CLASS_NUM]
        self._model_lib = {
            LR_MODEL: LogisticRegression,
            LSTM_MODEL: LstmAttention,
            CRNN_MODEL: CrnnModel,
            CRNN2D_MODEL: Crnn2dModel,
            CRNN2D_LARGER_MODEL: Crnn2dLargerModel,
            CRNN2D_VGG_MODEL: Crnn2dVggModel,
            BILSTM_MODEL: BilstmAttention,
            CNN_MODEL_2D: CnnModel2D,
            SVM_MODEL: SvmModel,
            ATTGRU: AttentionGru
        }
        self._model_sequences = [
            LR_MODEL,
            LSTM_MODEL,
            CRNN_MODEL,
            BILSTM_MODEL]
        self._max_first_model_run_loop = MODEL_FIRST_MAX_RUN_LOOP
        self._max_model_run_loop = 12

        self._models = {}

    def _get_or_create_model(self):
        # use new model and not reset model, have to initialize the model
        if not self._model.is_init:
            log('get new model {}'.format(self._model_name))
            # init model parameters
            if self._model_name == CNN_MODEL_2D:
                kwargs = {
                    'input_shape': self._input_shape[1:],
                    'num_classes': self.metadata[CLASS_NUM],
                    'max_layer_num': 10
                }
            elif self._model_name in [LSTM_MODEL, BILSTM_MODEL, CRNN_MODEL, CRNN2D_MODEL, CRNN2D_LARGER_MODEL,
                                      CRNN2D_VGG_MODEL, ATTGRU]:
                kwargs = {
                    'input_shape': self._input_shape[1:],
                    'num_classes': self.metadata[CLASS_NUM],
                }
            elif self._model_name == SVM_MODEL:
                kwargs = {
                    'kernel': 'linear',
                    'max_iter': 1000
                }
            elif self._model_name == LR_MODEL:
                kwargs = {
                    'kernel': 'liblinear',
                    'max_iter': 100
                }
            else:
                raise Exception("No such model!")
            if not self._model.is_init:
                self._model.init_model(**kwargs)
        log('This train loop use {}, last train loop use {}'.format(self._model_name, self._last_model_name))

    def _pre_select_model(self, train_loop_num):
        self._last_model_name = self._model_name

        if train_loop_num == 1 or self._model_name is None:
            self._model_name = self._model_sequences[0]
            self._each_model_best_auc[self._model_name] = [-1]
            self._each_model_best_predict[self._model_name] = [-1]
            self._use_new_train = True

        if self._not_rise_num == self._patience \
                or (self._model_num == 0 and self._cur_model_run_loop >= self._max_first_model_run_loop) \
                or (self._round_num == 0 and self._cur_model_run_loop >= self._max_model_run_loop):
            self._model_idx += 1
            if self._model_idx == len(
                    self._model_sequences) and LR_MODEL in self._model_sequences:
                # TODO be careful!
                self._model_idx = 1
                self._round_num += 1
                if self._round_num > 1:
                    self._patience = 4
                # sort model sequences by auc, desc
                if not self._data_manager.crnn_first:
                    self._model_sequences = [self._model_sequences[0]] \
                        + sorted(self._model_sequences[1:],
                                 key=lambda x: self._each_model_best_auc[x][-1], reverse=True)
                else:
                    self._model_sequences.remove(CRNN_MODEL)
                    self._model_sequences = [self._model_sequences[0]] + [CRNN_MODEL] \
                        + sorted(self._model_sequences[1:],
                                 key=lambda x: self._each_model_best_auc[x][-1], reverse=True)
                log(
                    'round {} start, model sequences {}'.format(self._round_num, self._model_sequences[self._model_idx:]))
            self._model_name = self._model_sequences[self._model_idx]
            self._model_num += 1
            self._not_rise_num = 0
            log(
                'change model from {} to {}, loop_num: {}'.format(self._last_model_name, self._model_name, self._cur_model_run_loop))

            self._use_new_train = self._model_num in [0,
                                                      1,
                                                      (2 * (len(self._model_sequences) - 1)) + 1,
                                                      (3 * (len(self._model_sequences) - 1)) + 1,
                                                      (4 * (len(self._model_sequences) - 1)) + 1]
            self._is_reset_model = (self._round_num > 1
                                    and self._model_num == self._round_num * (len(self._model_sequences) - 1) + 1)

            if self._use_new_train:
                self._test_x = None
            self._cur_model_run_loop = 0

            if self._round_num == 0 and self._cur_model_run_loop == 0:
                self._each_model_best_auc[self._model_name] = [-1]
                self._each_model_best_predict[self._model_name] = [-1]
                self._cur_model_max_auc = -1
            elif self._round_num == 1 and self._cur_model_run_loop == 0:
                self._cur_model_max_auc = self._each_model_best_auc[self._model_name][-1]
            elif self._round_num >= 2 and self._cur_model_run_loop == 0:
                self._each_model_best_auc[self._model_name] += [-1]
                self._each_model_best_predict[self._model_name] += [-1]
                self._cur_model_max_auc = -1

            if self._is_reset_model:
                log('new round {}'.format(self._round_num))
                # clear all models
                self._models.clear()
                del self._model
                self._model = None
                gc.collect()
                K.clear_session()
                # self._new_round = False

        if self._model_name != self._last_model_name or self._model is None or self._is_reset_model:
            if self._model_name in self._models:
                self._model = self._models[self._model_name]
            else:
                self._model = self._model_lib[self._model_name]()
                self._models[self._model_name] = self._model

    def _get_each_model_top_k_predicts(self):
        predicts = []
        for k, v in self._each_model_best_auc.items():
            if k == LR_MODEL:
                continue
            k_predicts = np.asarray(self._each_model_best_predict[k])
            temp = [(auc, k_predicts[i]) for i, auc in enumerate(v)
                    if auc > max(self._auc_threshold, self._k_best_auc[0] - 0.1)]
            temp.sort(key=lambda x: x[0], reverse=True)
            predicts.extend(temp[:self._each_model_top_k])

        if len(predicts) == 0:
            return [], []

        predicts = sorted(predicts, key=lambda x: x[0], reverse=True)[
            :self._each_model_keep_num]
        top_k_aucs = [predicts[i][0] for i in range(len(predicts))]
        top_k_predicts = [predicts[i][1] for i in range(len(predicts))]

        return top_k_aucs, top_k_predicts

    def _blending_ensemble(self):
        selected_k_best = [self._k_best_predicts[i]
                           for i, a in enumerate(self._k_best_auc) if a > 0.0]
        each_model_k_aucs, selected_each_model_k_best = self._get_each_model_top_k_predicts()

        if self._round_num >= 2:
            selected = selected_k_best + selected_each_model_k_best
        else:
            selected = selected_k_best

        log("model_num: {} Select k best {} predicts which have auc {}, ".format(self._model_num, self._keep_num, self._k_best_auc) +
            "each model {} best which have auc {}, ".format(self._each_model_keep_num, each_model_k_aucs) +
            "and each previous model's best predict which have auc " +
            "{} ".format(['({}:{})'.format(k, v) for k, v in self._each_model_best_auc.items()]))

        return np.mean(selected, axis=0)

    @property
    def data_manager(self):
        return self._data_manager

    def fit(self, train_loop_num=1, **kwargs):
        # select model first, inorder to use preprocess data method
        self._pre_select_model(train_loop_num)
        log('fit {} for {} times'.format(self._model_name, self._cur_model_run_loop))
        self._cur_model_run_loop += 1

        # get data
        if self._round_num == 0:
            train_x, train_y, val_x, val_y = self._data_manager.get_train_data(train_loop_num=train_loop_num,
                                                                               model_num=self._model_num,
                                                                               round_num=self._round_num,
                                                                               use_new_train=self._use_new_train,
                                                                               use_mfcc=self._use_mfcc)
            self._is_nedd_30s = self._data_manager.need_30s
            if self._is_nedd_30s:
                self._use_mel_round = 3
            else:
                self._use_mel_round = 2
        else:
            if self._round_num == self._use_mel_round:
                self._use_mfcc = False
            else:
                self._use_mfcc = True
            train_x, train_y, val_x, val_y = self._data_manager.get_train_data(train_loop_num=train_loop_num,
                                                                               model_num=self._model_num,
                                                                               round_num=self._round_num,
                                                                               use_new_train=self._use_new_train,
                                                                               use_mfcc=self._use_mfcc)

        self._val_set = (val_x, val_y)

        self._input_shape = train_x.shape
        log('train_x: {}; train_y: {};'.format(train_x.shape, train_y.shape) +
            ' val_x: {}; val_y: {};'.format(val_x.shape, val_y.shape))

        # init model really
        self._get_or_create_model()
        self._classes = np.unique(ohe2cat(train_y))
        self._model.fit(train_x, train_y, (val_x, val_y),
                        self._round_num, **kwargs)

    def predict(self, test_x, is_final_test_x=False):
        x_val, y_val = self._val_set
        auc = auc_metric(y_val, self._model.predict(x_val))
        need_predict = False
        if auc > self._cur_model_max_auc:
            log(
                'cur_max_auc {}; cur_auc {}; {} auc rise for {} times'.format(self._cur_model_max_auc, auc, self._model_name, self._cur_model_run_loop))
            self._cur_model_max_auc = auc
            if self._round_num == 0:
                self._not_rise_num = max(0, self._not_rise_num - 1)
            else:
                self._not_rise_num = 0
            if auc > self._each_model_best_auc[LR_MODEL][-1] - 0.1:
                need_predict = True
        else:
            self._not_rise_num += 1
            log(
                'cur_max_auc {}; cur_auc {}; {} auc not rise for {} times'.format(self._cur_model_max_auc, auc, self._model_name, self._not_rise_num))

        if max(self._k_best_auc[-1], self._each_model_best_auc[LR_MODEL]
               [-1] - 0.1) >= auc and not need_predict:
            log('not predict')
        else:
            log('new predict')
            if is_final_test_x:
                if self._test_x is None:
                    if self._model_num == 0:
                        self._test_x = self._data_manager.lr_preprocess(test_x)
                    elif self._round_num == 0:
                        self._test_x = self._data_manager.nn_preprocess(test_x,
                                                                        n_mfcc=96,
                                                                        max_duration=FIRST_ROUND_DURATION,
                                                                        is_mfcc=self._use_mfcc)
                    else:
                        self._test_x = self._data_manager.nn_preprocess(test_x,
                                                                        n_mfcc=128,
                                                                        max_duration=SECOND_ROUND_DURATION,
                                                                        is_mfcc=self._use_mfcc)
            if self._round_num > 1:
                _y_pred = self._model.predict(self._test_x, batch_size=32)
            else:
                _y_pred = self._model.predict(self._test_x, batch_size=32 * 8)

            # ensure y_pred has the same shape as y_test
            y_pred = np.zeros((self._test_x.shape[0], self._num_classes))
            for _id, _col in enumerate(self._classes):
                y_pred[:, _col] = _y_pred[:, _id]

            if self._k_best_auc[-1] < auc and auc > self._each_model_best_auc[LR_MODEL][-1] - 0.1:
                self._k_best_predicts[-1] = y_pred
                self._k_best_auc[-1] = auc
            if self._each_model_best_auc[self._model_name][-1] < auc:
                self._each_model_best_predict[self._model_name][-1] = y_pred
                self._each_model_best_auc[self._model_name][-1] = auc

            i = 0
            for auc, pred in sorted(
                    zip(self._k_best_auc, self._k_best_predicts), key=lambda x: x[0], reverse=True):
                self._k_best_auc[i] = auc
                self._k_best_predicts[i] = pred
                i += 1

        self._use_new_train = False
        self._is_reset_model = False

        return self._blending_ensemble()
