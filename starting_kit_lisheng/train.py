import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow import flags
from tensorflow import logging

import readers
import models
import utils
import os
import pickle




FLAGS = flags.FLAGS

debug = True



def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def train(inputfile_pattern='new_sample.tfrecord', test_inputfile_pattern='new_sample.tfrecord', model_architecture='Fully3layersModel', \
          optimizer='AdamOptimizer', learning_rate = 0.0001,
          num_epochs = 500, num_training_ex=60000, num_test_ex=10000, unused_batch_size=1000, \
          minibatch_size = 128, print_cost=True, cost_dir='costs/', parameter_dir='parameters/', \
          plot_costs=True):
    """
    train the model.

    Arguments:
    inputfile -- path of input tfrecord file, will be read by readers.read_inputs()
    model_architecture -- model class name implemented in models.py, will be referenced by find_class_by_name() to build the model graph
    optimizer -- some optimizer implemented in tf.train, will be referenced by find_class_by_name() to build the optimizer object
    learning rate -- learning rate used by specified optimizer
    num_epochs -- number of epochs of the optimization loop
    num_training_ex -- tell readers how many training examples to load from inputfile
    unused_batch_size -- when the entire training set is too large, we might want to batch it, not used in current implementation
    minibatch_size -- size of a minibatch for each epoch
    print_cost -- True to print the cost every 100 epochs
    cost_dir -- where to save the training cost (for later use, for example, learning curve)
    parameter_dir -- where to save the trained parameters (for later use, for example, prediction)
    plot_costs -- True to plot training costs at the end of training

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of trainig costs per 10 epochs.
    """
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    X_train, Y_train, labels_train = readers.read_inputs(inputfile_pattern, batch_size=num_training_ex) # read only num_training_ex this training
    if debug:
        print("*"*50, type(X_train), X_train.shape)
    X_test, Y_test, labels_test = readers.read_inputs(test_inputfile_pattern, batch_size=num_test_ex)
    (n_x, m) = X_train.get_shape().as_list()    # (n_x: input size, m : number of examples in the train set)
    logging.info("Number of training examples: "+str(m))
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = [] # Keep track of training epoch costs (every 10 epochs)
    train_costs = []    # Keep track of training costs (SGD effect, every iteration)
    test_costs = [] # Keep track of test costs (every iteration)

    X, Y = models.create_placeholders(n_x, n_y)     # Create Placeholders of shape (n_x, n_y)

    model_architecture_class = find_class_by_name(model_architecture, [models])() # create the model architecture
    logging.info("Built model: %s"%model_architecture)

    # Initialize parameters
    parameters = model_architecture_class.initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Y_hat = model_architecture_class.forward_propagation(X, parameters)

    cost = models.compute_cost(Y_hat, Y) # Cost function: Add cost function to tensorflow graph

    # Backpropagation: Define the tensorflow optimizer. Customize optimizer choice.
    logging.info("Set optimizer: %s"%optimizer)
    optimizer_class = find_class_by_name(optimizer, [tf.train])
    optimizer = optimizer_class(learning_rate).minimize(cost)



    # Initialize all the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      if debug:
          print("$"*50, sess.run(X_train).shape)
          print("$"*50, sess.run(labels_train))

      # read training data
      coord = tf.train.Coordinator() # The Coordinator class helps multiple threads stop together and report exceptions to a program that waits for them to stop
      logging.info('Read training data: Created coordinator')
      threads = tf.train.start_queue_runners(sess=sess, coord=coord) # create threads that use coord
      logging.info('Read training data: Started queue runners')
      X_train, Y_train, labels_train = sess.run([X_train, Y_train, labels_train])

      logging.info('Read training data DONE')

      logging.info('Read test data:')
      X_test, Y_test, labels_test = sess.run([X_test, Y_test, labels_test])
      coord.request_stop()
      coord.join(threads)
      logging.info('Read test data DONE: Stopped coordinator')

      for epoch in range(num_epochs+1):
        # print ("+++++++++++++++++++Epoch %i++++++++++++++++++++++"%epoch)
        # epoch_cost = 0.                       # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

        # print (X_train.shape, Y_train.shape, num_minibatches)
        minibatches = utils.random_mini_batches(X_train, Y_train, minibatch_size)

        for minibatch in minibatches:
          # Select a minibatch
          (minibatch_X, minibatch_Y) = minibatch

          # IMPORTANT: The line that runs the graph on a minibatch.
          # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
          _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
          # parameters_minibatch = sess.run(parameters)
          # Y_hat_train = sess.run(model_architecture_class.forward_propagation(X_train, parameters_minibatch))
          # train_cost = sess.run(cost, feed_dict={Y_hat:Y_hat_train, Y:Y_train})
          # train_costs.append(train_cost)

          # Y_hat_test = sess.run(model_architecture_class.forward_propagation(X_test, parameters_minibatch))
          # test_cost = sess.run(cost, feed_dict={Y_hat:Y_hat_test, Y:Y_test})
          # test_costs.append(test_cost)


        # print ("+++++++++++++++++++Epoch %i++++++++++++++++++++++"%epoch)
      # Print the cost every 100 epoch

        if print_cost == True and epoch % 10 == 0:

          parameters_epoch = sess.run(parameters)
          Y_hat_train = sess.run(model_architecture_class.forward_propagation(X_train, parameters_epoch))
          train_cost = sess.run(cost, feed_dict={Y_hat:Y_hat_train, Y:Y_train})
          logging.info("Training cost after epoch %i: %f" % (epoch, train_cost))
          train_costs.append(train_cost)

          Y_hat_test = sess.run(model_architecture_class.forward_propagation(X_test, parameters_epoch))
          test_cost = sess.run(cost, feed_dict={Y_hat:Y_hat_test, Y:Y_test})
          logging.info("Test cost after epoch %i: %f" % (epoch, test_cost))
          test_costs.append(test_cost)


      if plot_costs:# plot the cost
        plt.plot(np.squeeze(train_costs), label='costs on training set')
        plt.plot(np.squeeze(test_costs), label='costs on test set')
        plt.legend()
        plt.ylabel('cost')
        plt.xlabel('epochs (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

      # save the parameters in a variable
      parameters = sess.run(parameters)
      logging.info("Parameters have been trained!")


      if not os.path.exists(cost_dir):
        os.makedirs(cost_dir)
      if not os.path.exists(parameter_dir):
        os.makedirs(parameter_dir)

      with open(os.path.join(cost_dir, '%s_%sEX.test'%(model_architecture,str(num_training_ex))), 'wb') as fp:
        pickle.dump(test_costs, fp)
      with open(os.path.join(cost_dir, '%s_%sEX.train'%(model_architecture,str(num_training_ex))), 'wb') as fp:
        pickle.dump(train_costs, fp)
      logging.info("Test Costs are saved to %s"%(cost_dir))

      with open(os.path.join(parameter_dir, '%s_%sEX'%(model_architecture,str(num_training_ex))), 'wb') as fp:
        pickle.dump(parameters, fp)
      logging.info("Trained parameters are saved to %s"%(parameter_dir))


      return parameters, train_costs, test_costs



def main():
  inputfile_pattern = FLAGS.inputfile_pattern
  test_inputfile_pattern = FLAGS.test_inputfile_pattern
  model_architecture = FLAGS.model_architecture
  optimizer = FLAGS.optimizer
  learning_rate = FLAGS.base_learning_rate
  num_epochs = FLAGS.num_epochs
  num_training_ex = FLAGS.num_training_ex
  num_test_ex = FLAGS.num_test_ex
  unused_batch_size = FLAGS.batch_size
  minibatch_size = FLAGS.minibatch_size
  print_cost = FLAGS.print_cost
  cost_dir = FLAGS.cost_dir
  parameter_dir = FLAGS.parameter_dir
  plot_costs = FLAGS.plot_costs



  parameters, train_costs, test_costs = train(inputfile_pattern, test_inputfile_pattern, model_architecture, \
          optimizer, learning_rate, num_epochs, num_training_ex, num_test_ex, unused_batch_size,
          minibatch_size, print_cost, cost_dir, parameter_dir, plot_costs)




if __name__ == '__main__':

  logging.set_verbosity(tf.logging.INFO)

  flags.DEFINE_string(
      "inputfile_pattern", "new_sample.tfrecord",
      "Pattern of inputfile")

  flags.DEFINE_string(
      "test_inputfile_pattern", "new_sample.tfrecord",
      "Pattern of test inputfile")

  flags.DEFINE_string(
      "model_architecture", "Fully3layersModel",
      "Which architecture to use. Models are implemented in models.py.")
  # Training flags.
  flags.DEFINE_integer("num_training_ex", 60000,
                       "How many training examples to load from input file.")

  flags.DEFINE_integer("num_test_ex", 10000,
                       "How many test examples to load from test input file.")

  flags.DEFINE_integer("batch_size", 1000,
                       "Used when the training set is large. Unused for the moment.") #


  flags.DEFINE_integer("minibatch_size", 128,
                       "How many examples to process per batch for training.")

  flags.DEFINE_float("base_learning_rate", 0.0001,
                     "Which learning rate to start with for the chosen optimizer.")
  # flags.DEFINE_string("optimizer", "AdamOptimizer",
  #                     "What optimizer class to use.")
  flags.DEFINE_integer("num_epochs", 500,
    "How many passes to make over the dataset for training")

  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use, must be implemented in tf.train.")

  flags.DEFINE_string("cost_dir", "costs/",
                      "Where to save costs during training")

  flags.DEFINE_string("parameter_dir", "parameters/",
                      "Where to save trained parameters")

  flags.DEFINE_bool("print_cost", "True",
                      "True to print costs every 100 iterations")

  flags.DEFINE_bool("plot_costs", "True",
                      "True to plot costs after training")


  # tf.app.run(main=main)
  main()
  # test()
