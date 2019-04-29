# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir
#                            data      result     ingestion             code of participants

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
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
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
##############
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Verbosity level of logging:
##############
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'DEBUG' # To be changed to INFO for real competition

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there may be several datasets).
# The code should keep track of time spent and NOT exceed the time limit
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 300

# Maximum number of cycles, number of samples, and estimators
#############################################################
# Your training algorithm may be fast, so you may want to limit anyways the
# number of points on your learning curve (this is on a log scale, so each
# point uses twice as many time than the previous one.)
# The original code was modified to do only a small "time probing" followed
# by one single cycle. We can now also give a maximum number of estimators
# (base learners).
max_cycle = 1
max_estimators = 1000
max_samples = float('Inf')

# I/O defaults
##############
# If true, the previous output directory is not overwritten, it changes name
save_previous_results = False

# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
import logging
import os
from os import getcwd as pwd
from os.path import join
from functools import partial
import glob

# Redirect stardant output to live results page (detailed_results.html)
# to have live output for debugging
REDIRECT_STDOUT = False

VERSION = 'v20190429'
DESCRIPTION =\
"""See code comments in ingestion.py.
Previous updates:
20190429: [ZY] Remove useless code block
20190425: [ZY] Check prediction shape.
20190424: [ZY] Use logging instead of logger; remove start.txt checking;
20190419: [ZY] Try-except clause for training process;
          always terminates successfully.
"""

logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

# Default I/O directories:
# root_dir is the parent directory of the folder "AutoDL_ingestion_program"
root_dir = os.path.abspath(os.path.join(_HERE(), os.pardir))
default_input_dir = join(root_dir, "AutoDL_sample_data")
default_output_dir = join(root_dir, "AutoDL_sample_result_submission")
default_program_dir = join(root_dir, "AutoDL_ingestion_program")
default_submission_dir = join(root_dir, "AutoDL_sample_code_submission")


# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# General purpose functions
import time
import numpy as np
overall_start = time.time()         # <== Mark starting time
import os
import sys
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

def get_time_budget(autodl_dataset):
  """Time budget for a given AutoDLDataset."""
  # TODO: get this budget from CodaLab
  return 7200

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:
    #### Check whether everything went well
    ingestion_success = True

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

    # Redirect standard output to have live debugging info (esp. on CodaLab)
    if REDIRECT_STDOUT:
      if not os.path.exists(score_dir):
        os.makedirs(score_dir)
      detailed_results_filepath = os.path.join(score_dir,
                                               'detailed_results.html')
      logging.basicConfig(filename=detailed_results_filepath)
      logging.info("""<html><head> <meta http-equiv="refresh" content="5"> </head><body><pre>""")
      logging.info("Redirecting standard output. " +
                "Please check out output at {}."\
                .format(detailed_results_filepath))

    logging.debug("sys.argv = " + str(sys.argv))
    logging.debug("Using input_dir: " + input_dir)
    logging.debug("Using output_dir: " + output_dir)
    logging.debug("Using program_dir: " + program_dir)
    logging.debug("Using submission_dir: " + submission_dir)

	  # Our libraries
    path.append(program_dir)
    path.append(submission_dir)
    #IG: to allow submitting the starting kit as sample submission
    path.append(submission_dir + '/AutoDL_sample_code_submission')
    import data_io
    from dataset import AutoDLDataset # THE class of AutoDL datasets

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        logging.info('****** Ingestion program version ' + str(VERSION) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = [] # Do not proceed with learning and testing

    if len(datanames) != 1:
      raise ValueError("Multiple (or zero) datasets found in dataset_dir={}!\n"\
                       .format(input_dir) +
                       "Please put only ONE dataset under dataset_dir.")

    basename = datanames[0]

    logging.info("========== Ingestion program version " + str(VERSION) + " ==========")
    logging.info("************************************************")
    logging.info("******** Processing dataset " + basename[:-5].capitalize() + " ********")
    logging.info("************************************************")

    logging.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    # ======== Creating a data object with data, informations about it
    logging.info("Reading training set and test set...")

    ##### Begin creating training set and test set #####
    D_train = AutoDLDataset(os.path.join(input_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(input_dir, basename, "test"))
    ##### End creating training set and test set #####

    ## Get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    # ======== Keep track of time
    time_budget = get_time_budget(D_train)        # <== HERE IS THE TIME BUDGET!

    try:
      # ========= Creating a model
      from model import Model # in participants' model.py
      logging.info("Creating model...")
      ##### Begin creating model #####
      M = Model(D_train.get_metadata()) # The metadata of D_train and D_test only differ in sample_count
      ###### End creating model ######

      # Keeping track of how many predictions are made
      prediction_order_number = 0

      # Start the CORE PART: train/predict process
      start = time.time()
      while(True):
        remaining_time_budget = start + time_budget - time.time()
        logging.info("Training the model...")
        # Train the model
        M.train(D_train.get_dataset(),
                remaining_time_budget=remaining_time_budget)
        remaining_time_budget = start + time_budget - time.time()
        # Make predictions using the trained model
        Y_pred = M.test(D_test.get_dataset(),
                        remaining_time_budget=remaining_time_budget)
        if Y_pred is None: # Stop train/predict process if Y_pred is None
          break
        else:
          prediction_shape = tuple(Y_pred.shape)
          if prediction_shape != correct_prediction_shape:
            raise ValueError("Bad prediction shape! Expected {} but got {}."\
                          .format(correct_prediction_shape, prediction_shape))
        # Prediction files: adult.predict_0, adult.predict_1, ...
        filename_test = basename[:-5] + '.predict_' +\
          str(prediction_order_number)
        # Write predictions to output_dir
        data_io.write(os.path.join(output_dir,filename_test), Y_pred)
        prediction_order_number += 1
        logging.info("[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        remaining_time_budget = start + time_budget - time.time()
        logging.info( "[+] Time left %5.2f sec" % remaining_time_budget)
        if remaining_time_budget<=0:
          break
    except Exception as e:
      ingestion_success = False
      logging.info("Failed to run ingestion.")
      logging.error("Encountered exception:\n" + str(e), exc_info=True)

    # Finishing ingestion program
    overall_time_spent = time.time() - overall_start

    # Write overall_time_spent to a duration.txt file
    duration_filename =  'duration.txt'
    with open(os.path.join(output_dir, duration_filename), 'w') as f:
      f.write('Duration: ' + str(overall_time_spent) + '\n')
      f.write('Success: ' + str(int(ingestion_success)) + '\n')
      if verbose:
          logging.info("Successfully write duration to {}.".format(duration_filename))
      if ingestion_success:
          logging.info("[+] Done")
          logging.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
      else:
          logging.info("[-] Done, but encountered some errors during ingestion")
          logging.info("[-] Overall time spent %5.2f sec " % overall_time_spent)

    # Copy all files in output_dir to score_dir
    os.system("cp -R {} {}".format(os.path.join(output_dir, '*'), score_dir))
