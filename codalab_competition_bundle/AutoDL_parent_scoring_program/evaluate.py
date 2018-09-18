#!/usr/bin/env python
import sys
import os
import os.path
import time

_start = time.time()

input_dir = sys.argv[1]
output_dir = sys.argv[2]

n_datasets = 5 # TODO: to be changed to 5

# Parent phase has 1 as phase number by default

submit_dirs = []

for phase_number in range(2, 2 + n_datasets):
  submit_dir = os.path.join(input_dir, 'res_' + str(phase_number))
  submit_dirs.append(submit_dir)

score = 0

for submit_dir in submit_dirs:
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        submission_score_file = os.path.join(submit_dir, "scores.txt")
        submission_score = open(submission_score_file).readline()

        # Score is written like so: "score:<score>\nDuration:11.11"
        score_text = submission_score.split(":")[1]

        score += float(score_text)


detailed_results_path = os.path.join(output_dir, "detailed_results.html")
# Write to detailed results
with open(detailed_results_path, 'a+') as detailed_results:
#     detailed_results.write('<head> <meta http-equiv="refresh" content="1"> </head>')
    detailed_results.write("Oh yeah! Now it's good with real-time output and parallel tracks!")

_end = time.time()
_duration = _start - _end

output_filename = os.path.join(output_dir, 'scores.txt')
with open(output_filename, 'w') as output_file:
	output_file.write("score:{}\n".format(score))
	output_file.write("Duration:{0:.6f}\n".format(_duration))
