#!/usr/bin/env python
import sys
import os
import os.path
import time

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir_subphase_2 = os.path.join(input_dir, 'res_2')
submit_dir_subphase_3 = os.path.join(input_dir, 'res_3')

score = 0

for submit_dir in [submit_dir_subphase_2, submit_dir_subphase_3]:
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        submission_score_file = os.path.join(submit_dir, "scores.txt")
        submission_score = open(submission_score_file).readline()

        # Score is written like so: "correct:<score>"
        score_text = submission_score.split(":")[1]

        score += float(score_text)

output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'w')
output_file.write("correct:{}".format(score))
output_file.close()

# For testing realtime output, write to detailed results repeatedly
counter = 1
detailed_results_path = os.path.join(output_dir, "detailed_results.html")

# Refresh the page every second
# with open(detailed_results_path, 'a+') as detailed_results:
#     detailed_results.write('<head> <meta http-equiv="refresh" content="1"> </head>')
detailed_results.write("Oh yeah! Now it's good with real-time output and parallel tracks!")
