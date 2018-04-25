#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
from sys import argv
from os import getcwd as pwd

import libscores
import my_metric
import yaml
from libscores import *

# Default I/O directories:
root_dir = pwd()
from os.path import join
default_solution_dir = join(root_dir, "AutoDL_sample_data")
default_prediction_dir = join(root_dir, "AutoDL_sample_result_submission")
default_score_dir = join(root_dir, "AutoDL_scoring_output")

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 1
verbose = True

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)


def _load_scoring_function():
    with open(_HERE('metric.txt'), 'r') as f:
        metric_name = f.readline().strip()
        try:
            score_func = getattr(libscores, metric_name)
        except:
            score_func = getattr(my_metric, metric_name)
        return metric_name, score_func

# =============================== MAIN ========================================

if __name__ == "__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default data directories if no arguments are provided
        solution_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
    elif len(argv) == 3: # The current default configuration of Codalab
        solution_dir = os.path.join(argv[1], 'ref')
        prediction_dir = os.path.join(argv[1], 'res')
        score_dir = argv[2]
    elif len(argv) == 4:
        solution_dir = argv[1]
        prediction_dir = argv[2]
        score_dir = argv[3]
    else: 
        swrite('\n*** WRONG NUMBER OF ARGUMENTS ***\n\n')
        exit(1)
        
    if verbose:
        print("Using solution_dir: " + solution_dir)
        print("Using prediction_dir: " + prediction_dir)
        print("Using score_dir: " + score_dir)

        
    # Create the output directory, if it does not already exist and open output files
    mkdir(score_dir)
    score_file = open(os.path.join(score_dir, 'scores.txt'), 'wb')
    html_file = open(os.path.join(score_dir, 'scores.html'), 'wb')

    # Get the metric
    metric_name, scoring_function = _load_scoring_function()

    # Get all the solution files from the solution directory
    solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
    
    html_file.write('<pre>'.encode('utf-8'))

    # Loop over files in solution directory and search for predictions with extension .predict having the same basename
    for i, solution_file in enumerate(solution_names):
        set_num = i + 1  # 1-indexed
        score_name = 'set%s_score' % set_num

        # Extract the dataset name from the file name
        basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.') - 1]

        if 1==1: #try:
            # Get the last prediction from the res subdirectory (must end with '.predict')
            predict_file = ls(os.path.join(prediction_dir, basename + '*.predict'))[-1]
            if (predict_file == []): raise IOError('Missing prediction file {}'.format(basename))
            predict_name = predict_file[-predict_file[::-1].index(filesep):-predict_file[::-1].index('.') - 1]
            # Read the solution and prediction values into numpy arrays
            solution = read_array(solution_file)
            prediction = read_array(predict_file)
            if (solution.shape != prediction.shape): raise ValueError(
                "Bad prediction shape {}".format(prediction.shape))

            if 1==1: #try:
                # Compute the score prescribed by the metric file 
                score = scoring_function(solution, prediction)
                str_temp = "======= Set %d" % set_num + " (" + predict_name.capitalize() + "): " + metric_name + "(" + score_name + ")=%0.12f =======\n" % score
                print(str_temp)
                html_file.write(str_temp.encode('utf-8'))
            else: #except:
                raise Exception('Error in calculation of the specific score of the task')

            if debug_mode > 0:
                scores = compute_all_scores(solution, prediction)
                write_scores(html_file, scores)

        else: #except Exception as inst:
            score = missing_score
            print(
                "======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=ERROR =======")
            html_file.write(
                "======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=ERROR =======\n")
            print
            inst

        # Write score corresponding to selected task and metric to the output file
        str_temp = score_name + ": %0.12f\n" % score
        score_file.write(str_temp.encode('utf-8'))

    # End loop for solution_file in solution_names

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        str_temp = "Duration: %0.6f\n" % metadata['elapsedTime']
        score_file.write(str_temp.encode('utf-8'))
    except:
        str_temp = "Duration: 0\n"
        score_file.write(str_temp.encode('utf-8'))

        html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode > 1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(prediction_dir, score_dir)
        show_version(scoring_version)