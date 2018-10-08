#!/usr/bin/env python
import sys
import os
import os.path
import time

# Constant used for a missing score
missing_score = -0.999999

_start = time.time()

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# We have 5 datasets (tasks) in total
n_datasets = 5

# Parent phase has 1 as phase number by default

submit_dirs = []
score_names = []

for phase_number in range(2, 2 + n_datasets):
  submit_dir = os.path.join(input_dir, 'res_' + str(phase_number))
  submit_dirs.append(submit_dir)
  score_name = 'set{}_score'.format(phase_number - 1)
  score_names.append(score_name)

scores = []

for submit_dir in submit_dirs:
    if os.path.isdir(submit_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        submission_score_file = os.path.join(submit_dir, "scores.txt")
        submission_score = open(submission_score_file).readline()
        # Score is written like so: "score:<score>\nDuration:11.11"
        score_text = submission_score.split(":")[1]
        scores.append(float(score_text))
    else:
        print("{} doesn't exist. Use missing score.".format(submit_dir))
        scores.append(missing_score)

detailed_results_path = os.path.join(output_dir, "detailed_results.html")
# Write to detailed results
with open(detailed_results_path, 'a+') as detailed_results:
#     detailed_results.write('<head> <meta http-equiv="refresh" content="1"> </head>')
    detailed_results.write("Oh yeah! Now AutoDL is ready for beta testing!")

_end = time.time()
_duration = _start - _end

output_filename = os.path.join(output_dir, 'scores.txt')
with open(output_filename, 'w') as output_file:
  for i in range(n_datasets):
    score_name = score_names[i]
    score = scores[i]
    output_file.write("{}: {}\n".format(score_name, score))
  output_file.write("Duration: {:.6f}\n".format(_duration))
