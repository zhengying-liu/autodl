"""A less trivial example of learning algorithm.

Multinomial Logistic Regression model.
"""

import tensorflow as tf
import algorithm


class SoftmaxClassifier(algorithm.Algorithm):
  """Linear y_hat = WX + b -> softmax, no hidden layer"""

  def __init__(self, metadata):
    super(SoftmaxClassifier, self).__init__(metadata)
    self.model = tf.estimator.LinearClassifier(
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0))

  def train(self, dataset):
    """Keeps only the first example of the training set."""
    dataset_iterator = dataset.make_one_shot_iterator()
    # The next lines assume that
    # (a) get_next() returns a minibatch of examples
    # (b) each minibatch is a pair (inputs, outputs)
    # (c) the outputs has the same length as the inputs
    # We get the first minibatch by get_next,
    # then the output by [1], then the first example by [0].
    with tf.Session() as sess:
      a




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
