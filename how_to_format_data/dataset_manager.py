# Author: LIU Zhengying
# Creation Date: 18 June 2018
"""
A class to facilitate the management of datasets for AutoDL. It can keep
track of the files of each component (training set, test set, metadata, etc)
of a dataset, and make manipulations on them such as format transformation,
train/test split, example-label separation, checking, etc.

[IMPORTANT] To use this class, one should be very careful about file names! So
try not to modify file names manually.
Of course, originally this class is only reserved for organizers or their
collaborators, since the participants will only see the result of this class.
"""

import yaml
import tensorflow as tf
import os


class DatasetManager(object):

  # Conventional YAML file to save and load dataset info
  global_filename = 'dataset_info.yaml'

  def __init__(self, dataset_dir, dataset_name=None):

    # Dict containing almost all useful info on this dataset
    self._dataset_info = {}

    if os.path.isdir(dataset_dir):
      self._dataset_dir = dataset_dir
    else:
      raise ValueError("Failed to create dataset. {} is not a directory!"\
                       .format(dataset_dir))

    path_to_yaml = os.path.join(dataset_dir, global_filename)

    if os.path.exists(path_to_yaml):
      with open(path_to_yaml, 'r') as f:
        self._dataset_info = yaml.load(f)
    else:
      if dataset_name:
        self._dataset_name = dataset_name
      else:
        # Use folder name as dataset name
        self._dataset_name = os.path.basename(dataset_dir)

  def __del__(self):
    """Serialize the current state to the YAML file <global_filename> before
    destruction.
    """
    # TODO
    pass


  def convert_AutoML_to_AutoDL(self, *arg, **kwarg):
    """Convert a dataset in AutoML format to AutoDL format.

    This facilitates the process of generating new datasets in AutoDL format,
    since there exists a big database of datasets in AutoML format.
    """
    pass

def main():
  pass

if __name__ == '__main__':
  main()
