# Author: Zhengying LIU
# Creation date: 16 Oct 2018

import numpy as np
import pandas as pd
import scipy
import os
import sys
from pprint import pprint
sys.path.append('./ingestion_program/')
import data_io
from data_manager import DataManager
from autosklearn.classification import AutoSklearnClassifier

verbose = False

def is_sparse(obj):
  return scipy.sparse.issparse(obj)

def binary_to_multilabel(binary_label):
  return np.stack([1 - binary_label, binary_label], axis=1)

def regression_to_multilabel(regression_label, get_threshold=np.median):
  threshold = get_threshold(regression_label)
  binary_label = (regression_label > threshold)
  return binary_to_multilabel(binary_label)

def _prepare_metadata_features_and_labels(D, set_type='train'):
  data_format = D.info['format']
  task = D.info['task']
  if set_type == 'train':
    # Fetch features
    X_train = D.data['X_train']
    X_valid = D.data['X_valid']
    Y_train = D.data['Y_train']
    Y_valid = D.data['Y_valid']
    if is_sparse(X_train):
      concat = scipy.sparse.vstack
    else:
      concat = np.concatenate
    features = concat([X_train, X_valid])
    # Fetch labels
    labels = np.concatenate([Y_train, Y_valid])
  elif set_type == 'test':
    features = D.data['X_test']
    labels = D.data['Y_test']
  else:
    raise ValueError("Wrong set type, should be `train` or `test`!")
  # when the task if binary.classification or regression, transform it to multilabel
  if task == 'regression':
    labels = regression_to_multilabel(labels)
  elif task == 'binary.classification':
    labels = binary_to_multilabel(labels)
  return features, labels

if __name__ == '__main__':
  input_dir = '../../../autodl-contrib/raw_datasets/automl'
  output_dir = '../'
  for dataset_name in ['dorothea', 'adult']:
    D = DataManager(dataset_name, input_dir, replace_missing=False, verbose=verbose)
    X_test, Y_test = _prepare_metadata_features_and_labels(D, set_type='test')
    X_train, Y_train = _prepare_metadata_features_and_labels(D, set_type='train')
    print(Y_test.shape)
    time_budget = 7200
    model = AutoSklearnClassifier(time_left_for_this_task=time_budget, per_run_time_limit=time_budget//10)
    model.fit(X_train, Y_train)
    predict_path = os.path.join(output_dir, dataset_name + '.predict')
    Y_hat_test = model.predict_proba(X_test)
    print(Y_hat_test.shape)
    data_io.write(predict_path, Y_hat_test)
