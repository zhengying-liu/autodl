import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt

import readers 
import models 
import utils 
from tensorflow import flags

FLAGS = flags.FLAGS






def random_mini_batches(X, Y, mini_batch_size = 64, seed = None):
  # print (X)
  # X = X.eval()
  # print (X)
  m = X.shape[1] 
  mini_batches = []
  np.random.seed(seed)
  
  # Step 1: Shuffle (X, Y)
  permutation = list(np.random.permutation(m))
  shuffled_X = X[:, permutation]
  shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

  # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
  # num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
  num_complete_minibatches = int(m/mini_batch_size)
  for k in range(0, int(num_complete_minibatches)):
    mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
    mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)

  # Handling the end case (last mini-batch < mini_batch_size)
  if m % mini_batch_size != 0:
    mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
    mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
  
  return mini_batches