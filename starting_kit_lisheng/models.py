
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt




def create_placeholders(n_x, n_y):
  """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 28 * 28 = 784)
    n_y -- scalar, number of classes (from 0 to 9, so -> 10)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
  """
  X = tf.placeholder(tf.float32, (n_x, None))
  Y = tf.placeholder(tf.float32, (n_y, None))

  return X, Y


def compute_cost(Y_hat, Y):
  """
  Computes the cost
  
  Arguments:
  Y_hat -- output of forward propagation (of shape (number of class, number of examples)
  Y -- "true" labels vector placeholder, same shape as Y_hat
  
  Returns:
  cost - Tensor of the cost function
  """
  logits = tf.transpose(Y_hat)
  labels = tf.transpose(Y)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
  return cost


class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def initialize_parameters(self, *unused_params_shape, **unused_params):
    raise NotImplementedError()

  def forward_propagation(self, *unused_model_input, **unused_params):
    raise NotImplementedError

  def predict(self, *unused_model_input, **unused_params):
    raise NotImplementedError()

class Fully3layersModel(BaseModel):
  """LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX"""
  def initialize_parameters(self, n_x=784, n_nodes_hl1=500, n_nodes_hl2=500, n_y=10):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_nodes_hl1, n_x]
                        b1 : [n_nodes_hl1, 1]
                        W2 : [n_nodes_hl2, n_nodes_hl1]
                        b2 : [n_nodes_hl2, 1]
                        W3 : [n_y, n_nodes_hl2]
                        b3 : [n_y, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    # tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [n_nodes_hl1, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [n_nodes_hl1, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W2 = tf.get_variable("W2", [n_nodes_hl2, n_nodes_hl1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [n_nodes_hl2, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W3 = tf.get_variable("W3", [n_y, n_nodes_hl2], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


  def forward_propagation(self, X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (784, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit: the prediction
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                            # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                            # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                             # Z3 = np.dot(W3,Z2) + b3
    
    return Z3 


  def predict(self, X, parameters):
    """Predict X using given parameters
    Arguments:
    X -- input dataset placeholder, of shape (784, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3".
                  
    Return:
    prediction: of shape (number of classes, number of exmaples)

    """
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    (n_x, m) = X.shape
    x = tf.placeholder("float", [784, m])
    
    z3 = self.forward_propagation(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


class Fully6layersModel(BaseModel):
  """LINEAR -> RELU -> LINEAR -> RELU -> LINEAR ->... SOFTMAX"""
  def initialize_parameters(self, n_x=784, n_nodes_hl1=800, n_nodes_hl2=800, \
    n_nodes_hl3=800, n_nodes_hl4=800, n_nodes_hl5=800, n_y=10):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_nodes_hl1, n_x]
                        b1 : [n_nodes_hl1, 1]
                        W2 : [n_nodes_hl2, n_nodes_hl1]
                        b2 : [n_nodes_hl2, 1]
                        W3 : [n_y, n_nodes_hl2]
                        b3 : [n_y, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    # tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [n_nodes_hl1, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [n_nodes_hl1, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W2 = tf.get_variable("W2", [n_nodes_hl2, n_nodes_hl1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [n_nodes_hl2, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W3 = tf.get_variable("W3", [n_nodes_hl3, n_nodes_hl2], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_nodes_hl3, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W4 = tf.get_variable("W4", [n_nodes_hl4, n_nodes_hl3], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [n_nodes_hl3, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W5 = tf.get_variable("W5", [n_nodes_hl5, n_nodes_hl4], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [n_nodes_hl5, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    W6 = tf.get_variable("W6", [n_y, n_nodes_hl5], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable("b6", [n_y, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3, 
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6}
    
    return parameters


  def forward_propagation(self, X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit: the prediction
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                            # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                            # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3) 
    A3 = tf.nn.relu(Z3)                                            # Z3 = np.dot(W3,Z2) + b3
    Z4 = tf.add(tf.matmul(W4,A3), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5,A4),b5)                                            # Z2 = np.dot(W2, a1) + b2
    A5 = tf.nn.relu(Z5)                                            # A2 = relu(Z2)
    Z6 = tf.add(tf.matmul(W6,A5),b6)      
    return Z6


  def predict(self, X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    W5 = tf.convert_to_tensor(parameters["W5"])
    b5 = tf.convert_to_tensor(parameters["b5"])
    W6 = tf.convert_to_tensor(parameters["W6"])
    b6 = tf.convert_to_tensor(parameters["b6"])
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4,
              "W5": W5,
              "b5": b5,
              "W6": W6,
              "b6": b6}
    
    (n_x, m) = X.shape
    x = tf.placeholder("float", [784, m])
    
    z6 = self.forward_propagation(x, params)
    p = tf.argmax(z6)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction




class LogisticModel(BaseModel):
  """Linear y_hat = WX + b -> softmax, no hidden layer"""
  def initialize_parameters(self, n_x=784, n_y=10):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_y, n_x]
                        b1 : [n_y, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    # tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [n_y, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [n_y, 1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    
    parameters = {"W1": W1,
                  "b1": b1}
    
    return parameters

  def forward_propagation(self, X, parameters):
    """
    Implements the forward propagation for the model: y_hat = WX + b
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z1 -- the output of the last LINEAR unit: the prediction
    the softmax part will be done by compute_cost()
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    
    return Z1


  def predict(self, X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    params = {"W1": W1,
              "b1": b1}
    (n_x, m) = X.shape
    x = tf.placeholder("float", [784, m])
    
    z1 = self.forward_propagation(x, params)
    p = tf.argmax(z1)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction




