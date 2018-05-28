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

# Solve the Tkinter display issue of matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import time

# To compute area under learning curve
from sklearn.metrics import auc

import libscores
import my_metric
import yaml
from libscores import *

# Libraries for reconstructing the model

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

# Default I/O directories:
root_dir = os.path.abspath(os.path.join(_HERE(), os.pardir))
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

def _load_scoring_function():
    with open(_HERE('metric.txt'), 'r') as f:
        metric_name = f.readline().strip()
        try:
            score_func = getattr(libscores, metric_name)
        except:
            score_func = getattr(my_metric, metric_name)
        return metric_name, score_func


def get_prediction_files(prediction_dir, basename):
  """Return prediction files for the task <basename>.

  Examples of prediction file name: mini.predict_0, mini.predict_1
  """
  prediction_files = ls(os.path.join(prediction_dir, basename + '*.predict_*'))
  return prediction_files

def get_fig_name(basename):
  fig_name = "learning-curve-" + basename + ".png"
  return fig_name

def draw_learning_curve(solution_file, prediction_files,
                        scoring_function, output_dir, basename):
  """Draw learning curve for one task."""
  solution = read_array(solution_file) # numpy array
  scores = []
  timestamps = []
  for prediction_file in prediction_files:
    timestamp = os.path.getmtime(prediction_file)
    prediction = read_array(prediction_file) # numpy array
    if (solution.shape != prediction.shape): raise ValueError(
        "Bad prediction shape {}".format(prediction.shape))
    score = scoring_function(solution, prediction)
    scores.append(score)
    timestamps.append(timestamp)
  # Sort two lists according to timestamps
  sorted_pairs = sorted(zip(timestamps, scores))
  start = sorted_pairs[0][0]
  X = [t - start for t,_ in sorted_pairs]
  Y = [s for _,s in sorted_pairs]

  # Draw learning curve
  plt.clf()
  plt.plot(X,Y,marker="o", label="Test score")
  plt.title("Learning curve for the task " + basename)
  plt.xlabel('time/second')
  plt.ylabel('score')
  plt.legend()
  fig_name = get_fig_name(basename)
  path_to_fig = os.path.join(output_dir, fig_name)
  plt.savefig(path_to_fig)
  return X, Y

def area_under_learning_curve(X,Y):
  return auc(X,Y)

# TODO: transform this whole score.py script in an object-oriented manner
class Scorer():
  """A class for scoring one single task"""

  def ___init__(data_dir, solution_dir, prediction_dir, score_dir):
    self.birth_time = time.time()

    self.solution_dir = solution_dir
    self.prediction_dir = prediction_dir
    self.score_dir = score_dir
    self.time_budget = 300 # TODO

# =============================== MAIN ========================================

if __name__ == "__main__":
    import datetime
    the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    start = time.time()
    # TODO
    TIME_BUDGET = 300

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
        print("Scoring datetime:", the_date)



    # Create the output directory, if it does not already exist and open output files
    mkdir(score_dir)
    score_file = open(os.path.join(score_dir, 'scores.txt'), 'wb')
    html_file = open(os.path.join(score_dir, 'scores.html'), 'wb')

    # Get the metric
    metric_name, scoring_function = _load_scoring_function()
    metric_name = "Area Under Learning Curve"

    # Get all the solution files from the solution directory
    solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))

    # Automatic refresh every 5 seconds: content="5"
    tag_auto_refresh =\
      """<head> <meta http-equiv="refresh" content="5"> </head>"""

    html_file.write(tag_auto_refresh.encode('utf-8'))
    # html_file.write('<pre>'.encode('utf-8'))
    for i, solution_file in enumerate(solution_names):
        # Extract the dataset name from the file name
        basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.') - 1]
        # Add the image learning curve to html_file
        fig_name = get_fig_name(basename)
        img_tag = "<img src=" + fig_name + ">"
        html_file.write(img_tag.encode('utf-8'))
    html_file.flush()

    nb_preds = {x:0 for x in solution_names}
    scores = {x:0 for x in solution_names}

    # Moniter training processes while time budget is not attained
    while(time.time() < start + TIME_BUDGET):
      # Loop over files in solution directory and search for predictions with extension .predict having the same basename
      for i, solution_file in enumerate(solution_names):
          set_num = i + 1  # 1-indexed
          score_name = 'set%s_score' % set_num

          # Extract the dataset name from the file name
          basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.') - 1]

          time.sleep(0.5)

          # Give list of prediction files
          prediction_files = get_prediction_files(prediction_dir, basename)

          nb_preds_old = nb_preds[solution_file]
          nb_preds_new = len(prediction_files)

          if(nb_preds_new > nb_preds_old):
            print("New nb_preds:", nb_preds_new)
            # Draw the learning curve
            print("Refreshing learning curve for", basename)
            X, Y = draw_learning_curve(solution_file=solution_file,
                                prediction_files=prediction_files,
                                scoring_function=scoring_function,
                                output_dir=score_dir,
                                basename=basename)
            nb_preds[solution_file] = nb_preds_new
            scores[solution_file] = area_under_learning_curve(X,Y)


    for i, solution_file in enumerate(solution_names):
        set_num = i + 1  # 1-indexed
        score_name = 'set%s_score' % set_num

        # Extract the dataset name from the file name
        basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.') - 1]

        score = scores[solution_file]
        str_temp = "\n======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=%0.12f =======\n" % score
        print(str_temp)
        html_file.write(str_temp.encode('utf-8'))


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

    score_file.close()
    html_file.close()

    # Lots of debug stuff
    if debug_mode > 1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(prediction_dir, score_dir)
        show_version(scoring_version)
