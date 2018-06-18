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

import tensorflow as tf
import os
import json


class DatasetManager(object):
  
  global_filename = '.dataset_manager_info'
  
  def __new__(cls, dataset_dir, dataset_name=None):
    """Load from `<global_filename>` if exists
    """
    pass

  def __init__(self, dataset_dir, dataset_name=None):
    
    if os.path.isdir(dataset_dir):
      self._dataset_dir = dataset_dir
    else:
      raise ValueError("Failed to create dataset. {} is not a directory!"\
                       .format(dataset_dir))
      
    if dataset_name:
      self._dataset_name = dataset_name
    else:
      # Use folder name as dataset name
      self._dataset_name = os.path.basename(dataset_dir)  
    
    
    if tf.gfile.Exists('./.dataset_manager'):
      self

  def __del__(self):
    """Use pickle (or #TODO json) to serialize the current state before 
    destruction.
    """
    # TODO
    pass
  
  
  



def convert_AutoML_to_AutoDL(*arg, **kwarg):
  """Convert a dataset in AutoML format to AutoDL format.

  This facilitates the process of generating new datasets in AutoDL format,
  since there exists a big database of datasets in AutoML format.
  """
  pass




class Haha(object):
  def __init__(self, juhua=1, heihei=2):
    print("INIT", juhua)
    self.__yo = "juhua!"
    
  def __new__(cls, juhua=0):
    print("NEW", juhua)
    inst = super(Haha, cls).__new__(cls)
    return inst
    
  def __del__(self):
    pass
    
  def haha():
    print("hahaha")

if __name__ == '__main__':
  h = Haha(juhua=3,heihei=4)
