# AutoDL starting kit (1st version Dec 2017, by Lisheng Sun-Hosoya and Duc-Hieu Tran)

This starting kit contains codes for training AutoDL datasets (MNIST only for the moment).

These codes are inspired by:
* Youtube-8m Challenge starter codes (https://github.com/google/youtube-8m/blob/master/eval.py)
* Andrew Ng's Deep Learning course (https://www.coursera.org/learn/deep-neural-network/home/welcome)

## Structure of starting kit:
* readers.py and test_readers.py: build readers (and its tester) to read our input .tfrecord files. Shapes of tensors are specified to be consistent to these files, can be changed if needed. 
* utils.py: helper functions.

* models.py: 3 models are implemented. It contains also an evaluation function (compute_cost()) which returns the softmax cross entropy. You can extend this file to have your own model. It will be called by train.py.

* train.py: where we build the model graph (using one of the models implemented in models.py) and train on some training examples (read by readers.py). We have provided some arguments to customize this training process:
	* inputfile_pattern: Pattern of inputfile, default = new_sample.tfrecord
	* model_architecture: Which architecture to use. Models are implemented in models.py. Use the model class name. default = Fully3layersModel
	* num_training_ex: How many examples to load from input file. default = 1000
	* batch_size: Used when the training set is large. Unused for the moment. default = 1000
	* minibatch_size: How many examples to process per minibatch (in SGD context) for training. default = 128
	* optimizer: What optimizer class to use, must be implemented in tf.train.
	* base_learning_rate: Which learning rate to start with for the chosen optimizer (now we use the constant learning rate, but learning rate decay can be used later.) default=0.0001
	* num_epochs: How many passes to make over the dataset for training. default=500
	* cost_dir: Where to save costs during training. default=./cost/
	* parameter_dir: Where to save trained parameters. default=./parameters/
	* print_cost: True to print costs every 100 iterations
	* plot_costs: True to plot costs after training

## Quick start:
	* python train.py (train a LINEAR->RELU->LINEAR->RELU_LINEAR->SOFTMAX fully connected model on 1000 training examples for 500 epochs)
	* python train.py --model_architecture LogisticModel --num_training_ex 10000 --num_epochs 1000
## TODO:
* Argument Exception handling: how to raise error when invalid flags are passed? For example, if I type 'model_architecture' in place of '--model_architecture'; or 'inputfile' in place of 'inputfile_pattern'


