################################################################################
# Name:         Ingestion Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  Apr 29 2019
# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir
#                            data      result     ingestion             code of participants

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.

VERSION = 'v20190504'
DESCRIPTION =\
"""This is the "ingestion program" written by the organizers. It takes the
code written by participants (with `model.py`) and one dataset as input,
run the code on the dataset and produce predictions on test set. For more
information on the code/directory structure, please see comments in this
code (ingestion.py) and the README file of the starting kit.
Previous updates:
20190504: [ZY] Check if model.py has attribute done_training and use it to
               determinate whether ingestion has ended;
               Use module-specific logger instead of logging (with root logger);
               At beginning, write start.txt with ingestion_pid and start_time;
               In the end, write end.txt with end_time and ingestion_success;
20190429: [ZY] Remove useless code block; better code layout.
20190425: [ZY] Check prediction shape.
20190424: [ZY] Use logging instead of logger; remove start.txt checking;
20190419: [ZY] Try-except clause for training process;
          always terminates successfully.
"""
# The input directory input_dir (e.g. AutoDL_sample_data/) contains one dataset
# folder (e.g. adult.data/) with the training set (train/)  and test set (test/),
# each containing an some tfrecords data with a `metadata.textproto` file of
# metadata on the dataset. So one AutoDL dataset will look like
#
#   adult.data
#   ├── test
#   │   ├── metadata.textproto
#   │   └── sample-adult-test.tfrecord
#   └── train
#       ├── metadata.textproto
#       └── sample-adult-train.tfrecord
#
# The output directory output_dir (e.g. AutoDL_sample_result_submission/)
# will receive all predictions made during the whole train/predict process
# (thus this directory is updated when a new prediction is made):
# 	adult.predict_0
# 	adult.predict_1
# 	adult.predict_2
#        ...
# after ingestion has finished, a file end.txt will be written, containing
# info on the duration ingestion used. This file is also used as a signal
# for scoring program showing that ingestion has terminated.
#
# The code directory submission_program_dir (e.g. AutoDL_sample_code_submission/)
# should contain your code submission model.py (and possibly other functions
# it depends upon).
#
# We implemented several classes:
# 1) DATA LOADING:
#    ------------
# dataset.py
# dataset.AutoDLMetadata: Read metadata in metadata.textproto
# dataset.AutoDLDataset: Read data and give tf.data.Dataset
# 2) LEARNING MACHINE:
#    ----------------
# model.py
# model.Model.train
# model.Model.test
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# UNIVERSITE PARIS SUD, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL UNIVERSITE PARIS SUD AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Zhengying Liu

# =========================== BEGIN OPTIONS ==============================

# Verbosity level of logging:
##############
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
time_budget = 7200

# Some common useful packages
from os import getcwd as pwd
from os.path import join
from sys import argv, path
import datetime
import glob
import logging
import numpy as np
import os
import sys
import time

def get_logger(verbosity_level, use_error_log=False):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  if use_error_log:
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger(verbosity_level)

def _HERE(*args):
  """Helper function for getting the current directory of this script."""
  h = os.path.dirname(os.path.realpath(__file__))
  return os.path.abspath(os.path.join(h, *args))

def write_start_file(output_dir, start_time):
  """Create start file 'start.txt' in `output_dir` with ingestion's pid and
  start time.
  """
  ingestion_pid = os.getpid()
  start_filename =  'start.txt'
  start_filepath = os.path.join(output_dir, start_filename)
  with open(start_filepath, 'w') as f:
    f.write('ingestion_pid: ' + str(ingestion_pid) + '\n')
    f.write('start_time: ' + str(start_time) + '\n')
  logger.debug("Finished writing 'start.txt' file.")

class ModelApiError(Exception):
  pass

class BadPredictionShapeError(Exception):
  pass

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__":
    # Mark starting time of ingestion
    start = time.time()
    logger.info("="*5 + " Start ingestion program. " +
                "Version: {} ".format(VERSION) + "="*5)

    #### Check whether everything went well
    ingestion_success = True

    # Default I/O directories:
    # root_dir is the parent directory of this script (ingestion.py)
    root_dir = _HERE(os.pardir)
    default_input_dir = join(root_dir, "AutoDL_sample_data")
    default_output_dir = join(root_dir, "AutoDL_sample_result_submission")
    default_program_dir = join(root_dir, "AutoDL_ingestion_program")
    default_submission_dir = join(root_dir, "AutoDL_sample_code_submission")

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
        score_dir = join(root_dir, "AutoDL_scoring_output")
    elif len(argv)==2: # the case for indicating special input_dir
        input_dir = argv[1]
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
        score_dir = join(root_dir, "AutoDL_scoring_output")
    elif len(argv)==3: # the case for indicating special input_dir and submission_dir. The case for run_local_test.py
        input_dir = argv[1]
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= argv[2]
        score_dir = join(root_dir, "AutoDL_scoring_output")
    else: # the case on CodaLab platform
        input_dir = os.path.abspath(os.path.join(argv[1], '../input_data'))
        output_dir = os.path.abspath(os.path.join(argv[1], 'res'))
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(os.path.join(argv[4], '../submission'))
        score_dir = os.path.abspath(os.path.join(argv[4], '../output'))

    logger.debug("sys.argv = " + str(sys.argv))
    logger.debug("Using input_dir: " + input_dir)
    logger.debug("Using output_dir: " + output_dir)
    logger.debug("Using program_dir: " + program_dir)
    logger.debug("Using submission_dir: " + submission_dir)

	  # Our libraries
    path.append(program_dir)
    path.append(submission_dir)
    #IG: to allow submitting the starting kit as sample submission
    path.append(submission_dir + '/AutoDL_sample_code_submission')
    import data_io
    from dataset import AutoDLDataset # THE class of AutoDL datasets

    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    if len(datanames) != 1:
      raise ValueError("Multiple (or zero) datasets found in dataset_dir={}!\n"\
                       .format(input_dir) +
                       "Please put only ONE dataset under dataset_dir.")

    basename = datanames[0]

    write_start_file(output_dir, start_time=start)

    logger.info("************************************************")
    logger.info("******** Processing dataset " + basename[:-5].capitalize() +
                 " ********")
    logger.info("************************************************")
    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    ##### Begin creating training set and test set #####
    logger.info("Reading training set and test set...")
    D_train = AutoDLDataset(os.path.join(input_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(input_dir, basename, "test"))
    ##### End creating training set and test set #####

    ## Get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    try:
      # ========= Creating a model
      from model import Model # in participants' model.py
      ##### Begin creating model #####
      logger.info("Creating model...")
      M = Model(D_train.get_metadata()) # The metadata of D_train and D_test only differ in sample_count
      ###### End creating model ######

      # Check if the model has methods `train` and `test`.
      for attr in ['train', 'test']:
        if not hasattr(M, attr):
          raise ModelApiError("Your model object doesn't have the method " +
                              "`{}`. Please implement it in model.py.")

      # Check if model.py uses new done_training API instead of marking
      # stopping by returning None
      use_done_training_api = hasattr(M, 'done_training')
      if not use_done_training_api:
        logger.warning("Your model object doesn't have an attribute " +
                       "`done_training`. But this is necessary for ingestion " +
                       "program to know whether the model has done training " +
                       "and to decide whether to proceed more training. " +
                       "Please add this attribute to your model.")

      # Keeping track of how many predictions are made
      prediction_order_number = 0

      # Start the CORE PART: train/predict process
      while(not (use_done_training_api and M.done_training)):
        remaining_time_budget = start + time_budget - time.time()
        # Train the model
        logger.info("Begin training the model...")
        M.train(D_train.get_dataset(),
                remaining_time_budget=remaining_time_budget)
        logger.info("Finished training the model.")
        remaining_time_budget = start + time_budget - time.time()
        # Make predictions using the trained model
        logger.info("Begin testing the model by making predictions " +
                     "on test set...")
        Y_pred = M.test(D_test.get_dataset(),
                        remaining_time_budget=remaining_time_budget)
        logger.info("Finished making predictions.")
        if Y_pred is None: # Stop train/predict process if Y_pred is None
          logger.info("The method model.test returned `None`. " +
                      "Stop train/predict process.")
          break
        else: # Check if the prediction has good shape
          prediction_shape = tuple(Y_pred.shape)
          if prediction_shape != correct_prediction_shape:
            raise BadPredictionShapeError(
              "Bad prediction shape! Expected {} but got {}."\
              .format(correct_prediction_shape, prediction_shape)
            )
        # Prediction files: adult.predict_0, adult.predict_1, ...
        filename_test = basename[:-5] + '.predict_' +\
          str(prediction_order_number)
        # Write predictions to output_dir
        data_io.write(os.path.join(output_dir,filename_test), Y_pred)
        prediction_order_number += 1
        logger.info("[+] {0:d} predictions made, time spent so far {1:.2f} sec"\
                     .format(prediction_order_number, time.time() - start))
        remaining_time_budget = start + time_budget - time.time()
        logger.info( "[+] Time left {0:.2f} sec".format(remaining_time_budget))
        if remaining_time_budget<=0:
          break
    except Exception as e:
      ingestion_success = False
      logger.info("Failed to run ingestion.")
      logger.error("Encountered exception:\n" + str(e), exc_info=True)

    # Finishing ingestion program
    end_time = time.time()
    overall_time_spent = end_time - start

    # Write overall_time_spent to a end.txt file
    duration_filename =  'end.txt'
    with open(os.path.join(output_dir, duration_filename), 'w') as f:
      f.write('ingestion_duration: ' + str(overall_time_spent) + '\n')
      f.write('ingestion_success: ' + str(int(ingestion_success)) + '\n')
      f.write('end_time: ' + str(end_time) + '\n')
      logger.info("Successfully write the file {}.".format(duration_filename))
      if ingestion_success:
          logger.info("[+] Done. Ingestion program successfully terminated.")
          logger.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
      else:
          logger.info("[-] Done, but encountered some errors during ingestion.")
          logger.info("[-] Overall time spent %5.2f sec " % overall_time_spent)

    # Copy all files in output_dir to score_dir
    os.system("cp -R {} {}".format(os.path.join(output_dir, '*'), score_dir))
    logger.debug("Copied all ingestion output to scoring output directory.")
