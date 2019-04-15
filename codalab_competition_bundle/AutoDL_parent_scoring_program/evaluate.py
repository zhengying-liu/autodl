#!/usr/bin/env python
"""For testing parent scoring program locally.

To do this, run
```
python evaluate.py test_input/ test_output/
```
"""
import sys
import os
import os.path
import time
from glob import glob
import base64
import yaml

def write_scores_html(output_dir, image_paths):
  filename = 'detailed_results.html'
  detailed_results_path = os.path.join(output_dir, filename)
  html_head = """<html><body><pre>"""
  html_end = '</pre></body></html>'
  with open(detailed_results_path, 'w') as html_file:
      # Automatic refreshing the page on file change using Live.js
      html_file.write(html_head)
      # html_file.write("Oh yeah! Now AutoDL is ready for beta testing!<br>")
      for image_path in image_paths:
        with open(image_path, "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read())
          encoded_string = encoded_string.decode('utf-8')
          s = '<img src="data:image/png;charset=utf-8;base64,%s"/>'%encoded_string
          html_file.write(s + '<br>')
      html_file.write(html_end)

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
image_paths = []

# Read result folders (submit_dirs) from metadata file
metadata_path = os.path.join(input_dir, 'metadata')
f = open(metadata_path, 'r')
metadata = f.read().split('\n')
f.close()
metadata = [x.split(':')[0] for x in metadata if x.startswith('res_')]
metadata = sorted(metadata, key=lambda x: int(x.split('_')[1])) # sort by number

if len(metadata) != n_datasets:
    raise Exception(str(os.listdir(input_dir)))

for i, l in enumerate(metadata):
    submit_dir = os.path.join(input_dir, l)
    submit_dirs.append(submit_dir)
    score_name = 'set{}_score'.format(i+1)
    score_names.append(score_name)
    learning_curve_images = glob(os.path.join(submit_dir,'learning-curve-*.png'))
    learning_curve_images = sorted(learning_curve_images) # alphabetic sort
    for image_path in learning_curve_images:
      image_paths.append(image_path)

scores = []

score_name_yaml = 'score'
for submit_dir in submit_dirs:
    if os.path.isdir(submit_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        submission_score_file = os.path.join(submit_dir, "scores.txt")
        # submission_score = open(submission_score_file).readline()
        try:
          with open(submission_score_file, 'r') as f:
            score_info = yaml.safe_load(f)
          score_text = score_info[score_name_yaml] # might be already float
          scores.append(float(score_text)) # but to be sure
        except Exception as e:
          print("Failed to load score in: {}".format(submit_dir))
          print("The submission may have failed on this task.")
          print("Set score for this task to: {}".format(missing_score))
          print("Exception encountered: ", e)
          scores.append(missing_score)
    else:
        print("{} doesn't exist. Use missing score.".format(submit_dir))
        scores.append(missing_score)

write_scores_html(output_dir, image_paths)

_end = time.time()
_duration = _start - _end

output_filename = os.path.join(output_dir, 'scores.txt')
with open(output_filename, 'w') as output_file:
  score_name = score_names[0]
  score = sum(scores) / len(scores) # mean
  output_file.write("{}: {}\n".format(score_name, score))
  output_file.write("Duration: {:.6f}\n".format(_duration))
