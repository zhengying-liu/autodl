#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir
#                            data      result     ingestion             code of participants

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# The input directory input_dir (e.g. sample_data/) contains the dataset(s), including:
#   dataname/metadata.textproto # A
#       metadata.textproto
#       sample-00000-of-00007
#       sample-00001-of-00007
#       sample-00002-of-00007
#       sample-00003-of-00007
#       sample-00004-of-00007
#       sample-00005-of-00007
#       sample-00006-of-00007
#
# The output directory output_dir (e.g. sample_result_submission/)
# will receive the predicted values (no subdirectories):
# 	dataname_test.predict
# 	dataname_valid.predict
#
# The code directory submission_program_dir (e.g. sample_code_submission/) should contain your
# code submission model.py (an possibly other functions it depends upon).
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
import os
from os import getcwd as pwd
from os.path import join

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

# Version of the sample code
version = 1

# General purpose functions
import time
import numpy as np
overall_start = time.time()         # <== Mark starting time
import os
import sys
from sys import argv, path
import datetime
the_date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:
    #### Check whether everything went well (no time exceeded)
    execution_success = True

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
        # TODO
        # Now data are with reference data, in run/input/ref
        # Eric created run/submission to store participants' model.py
        input_dir = os.path.abspath(os.path.join(argv[1], 'ref'))
        output_dir = os.path.abspath(os.path.join(argv[1], 'res'))
        submission_dir = os.path.abspath(os.path.join(argv[4], '..', 'submission'))

    if verbose: # For debugging
        print("sys.argv = ", sys.argv)
        with open(os.path.join(program_dir, 'metadata'), 'r') as f:
          print("Content of the metadata file: ")
          print(f.read())
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using program_dir: " + program_dir)
        print("Using submission_dir: " + submission_dir)
        print("In input_dir: ", os.listdir(input_dir))
        try:
          print("In output_dir: ", os.listdir(output_dir))
        except:
          print("In output_dir: ", "no output directory yet.")
        print("In program_dir: ", os.listdir(program_dir))
        print("In submission_dir: ", os.listdir(submission_dir))
        print("Ingestion datetime:", the_date)
        try:
          print("In input_dir/res: ", os.listdir(os.path.join(input_dir, 'res')))
          print("In input_dir/ref: ", os.listdir(os.path.join(input_dir, 'ref')))
          print("In run/: ", os.listdir(os.path.join(input_dir, '..')))
          print("In /: ", os.listdir('/'))
        except:
          pass

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)
    path.append (submission_dir + '/AutoDL_sample_code_submission') #IG: to allow submitting the starting kit as sample submission
    import data_io
    from data_io import vprint
    import tensorflow as tf
    from model import Model
    from dataset import AutoDLDataset

    if debug_mode >= 4: # Show library version and directory structure
        data_io.show_dir(".")

    # Move old results and create a new output directory (useful if you run locally)
    if save_previous_results:
        data_io.mvdir(output_dir, output_dir+'_'+the_date)
    data_io.mkdir(output_dir)

    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    # Overwrite the "natural" order

    #### Delete zip files and metadata file
    datanames = [x for x in datanames
      if x!='metadata' and x.endswith('.data') and not x.endswith('.zip')] # TODO: change .endswith('.data')

    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_version()
        data_io.show_io(input_dir, output_dir)
        print('\n****** Ingestion program version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')
        data_io.write_list(datanames)
        datanames = [] # Do not proceed with learning and testing

    #### MAIN LOOP OVER DATASETS:
    overall_time_budget = 0
    time_left_over = 0


    for i, basename in enumerate(datanames): # Loop over datasets (if several)

        vprint( verbose,  "\n========== Ingestion program version " + str(version) + " ==========\n")
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")

        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()

        # ======== Creating a data object with data, informations about it
        vprint( verbose,  "========= Reading and converting data ==========")
        # TODO: Read data : 2 datasets: train, test

        ##### To show to Andre #####
        D_train = AutoDLDataset(os.path.join(input_dir, basename, "train"))
        D_test = AutoDLDataset(os.path.join(input_dir, basename, "test"))
        D_train.init()
        D_test.init(batch_size=1000, repeat=False)
        ##### To show to Andre #####

        vprint( verbose,  "[+] Size of uploaded data  %5.2f bytes" % data_io.total_size(D_train))
        # TODO: modify total_size

        # ======== Keep track of time
        # TODO: different time budget for different dataset (mnist, cifar, ...)
        if debug_mode<1:
            time_budget = max_time
            #time_budget = D.info['time_budget']        # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time

        overall_time_budget = overall_time_budget + time_budget
        vprint( verbose,  "[+] Cumulated time budget (all tasks so far)  %5.2f sec" % (overall_time_budget))
        # We do not add the time left over from previous dataset: time_budget += time_left_over
        vprint( verbose,  "[+] Time budget for this task %5.2f sec" % time_budget)
        time_spent = time.time() - start
        vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        # ========= Creating a model
        vprint( verbose,  "======== Creating model ==========")

        ##### To show to Andre #####
        M = Model(D_train.get_metadata())
        ##### To show to Andre #####

        # 2 metadata files for model? Not a problem!

        # ========= Reload trained model if it exists
        vprint( verbose,  "**********************************************************")
        vprint( verbose,  "****** Attempting to reload model to avoid training ******")
        vprint( verbose,  "**********************************************************")
        you_must_train=1
        modelname = os.path.join(submission_dir,basename)
        # if os.path.isfile(modelname + '_model.pickle'):
        #     M = M.load(modelname)
        #     you_must_train=0
        #     vprint( verbose,  "[+] Model reloaded, no need to train!")

        total_pause_time = 0
        prediction_order_number = 0

        # ========= Train if needed only
        if you_must_train:
            # vprint( verbose, "======== Trained model not found, proceeding to train!")
            vprint(verbose, "======== Begin training. Use Ctrl+C to pause.")
            start = time.time()
            # TODO:
            while(True):
              try:
                # Training budget on each dataset is equal to (time budget / number of datasets)
                while(time.time() < overall_start + time_budget/len(datanames)*(i+1) + total_pause_time):
                  # Train the model
                  M.train(D_train.get_dataset())
                  # Make predictions using the most recent checkpoint
                  # Prediction files: mini.predict_0, mini.predict_1, ...
                  Y_test = M.test(D_test.get_dataset())
                  now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
                  vprint( verbose, "INFO:" + str(now)+ " ======== Saving results to: " + output_dir)
                  filename_test = basename[:-5] + '.predict_' +\
                    str(prediction_order_number)
                  # Write predictions to output_dir
                  data_io.write(os.path.join(output_dir,filename_test), Y_test)
                  prediction_order_number += 1

                # If training terminates normally then break the loop
                break

              # Ctrl+C for pause training and make predictions using present
              # model. Might be useless for the final competition, due to
              # change of logic (now we want to kill training thread completely)
              except KeyboardInterrupt:
                pause_start = time.time()
                vprint(verbose, "\n======== User interrupt. Pause training.")
                # User types 'y' to resume training, types others to quit
                vprint(verbose, "======== Resume training? (y/n)")
                input = sys.stdin.readline()[0].lower()
                if input == 'y':
                  vprint(verbose, "======== Continue training...")
                  continue
                else:
                  vprint(verbose,
                    "======== Stop training and make final predictions...")
                  break
                pause_end = time.time()
                total_pause_time += pause_end - pause_start

        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
        time_spent = time.time() - start
        time_left_over = time_budget - time_spent
        vprint( verbose,  "[+] End cycle, time left %5.2f sec" % time_left_over)
        if time_left_over<=0: break

    # Finishing ingestion program
    overall_time_spent = time.time() - overall_start
    # Write overall_time_spent to a duration.txt file
    duration_filename =  'duration.txt'
    with open(os.path.join(output_dir, duration_filename), 'w') as f:
      f.write(str(overall_time_spent))
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)
