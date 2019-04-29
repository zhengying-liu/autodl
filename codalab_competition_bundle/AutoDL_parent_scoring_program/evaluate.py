################################################################################
# Name:         Parent Scoring Program
# Author:       Zhengying Liu, Zhen Xu, Isabelle Guyon
# Update time:  Apr 25 2019
# Usage: 		    python evaluate.py input_dir output_dir         

VERISION = "v20190426"
DESCRIPTION = '''This is the parent scoring program. It reads from \
input_dir/res_i/ all partial results from children phases, and outputs \
aggregated learning curves and scores to output_dir.'''

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
################################################################################

import os
from os.path import join
import sys
import yaml
import argparse
import base64
from shutil import copyfile
from glob import glob
import logging
logging.basicConfig(
   level=logging.DEBUG,
   format="%(asctime)s %(levelname)s %(filename)s: %(message)s",
   datefmt='%Y-%m-%d %H:%M:%S'
)

################################################################################
# USER DEFINED CONSTANTS
################################################################################


# Number of children phases/datasets (as defined in competition bundle)
DEFAULT_NUM_DATASET = 5
current_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SCORE = join(current_path, 'default_scores.txt')
DEFAULT_CURVE = join(current_path, 'default_curve.png')

print (current_path)
print (DEFAULT_SCORE)
print (DEFAULT_CURVE)

################################################################################
# FUNCTIONS
################################################################################

def validate_full_res(args):
  """
    Check if we have DEFAULT_NUM_DATASET results in the args.input_dir.
    Replace by defaulta values otherwise.
  """
  for i in range(DEFAULT_NUM_DATASET):
  	# Check whether res_i/ exists
    check_path = join(args.input_dir, "res_"+str(i+2))
    logging.info("Checking " + str(check_path))
    if not os.path.exists(check_path):
      # Replace both learning curve and score by default:
      logging.warning(str(check_path) + 
                    " does not exist. Default values will be used.")
      # Create this folder and copy default values
      os.mkdir(check_path)
      copyfile(DEFAULT_SCORE, join(check_path,"scores.txt"))
      copyfile(DEFAULT_CURVE, join(check_path,"learning-curve-default.png"))
    else:
    # Replace either learning curve or score by default, depending...
      if not os.path.exists(join(check_path,"scores.txt")):
        logging.warning("Score file" +
                       " does not exist. Default values will be used.")
        copyfile(DEFAULT_SCORE, join(check_path,"scores.txt"))
      is_curve_exist = False
      for f in os.listdir(check_path):
        if f[-4:] == ".png":
          is_curve_exist = True
          break
      if not is_curve_exist:
        logging.warning("Learning curve" +
                       " does not exist. Default values will be used.")
        copyfile(DEFAULT_CURVE, join(check_path,"learning-curve-default.png"))
  return

def read_score(args):
  """
    Fetch scores from scores.txt
  """
  # TODO: should not be hard coded: figure out which phase you are in.
  score_ls = []
  for i in range(DEFAULT_NUM_DATASET):
    score_dir = args.input_dir + "/res_"+str(i+2)
    score_file = join(score_dir, "scores.txt")
    try:
      with open(score_file, 'r') as f:
        score_info = yaml.safe_load(f)
      score_ls.append(float(score_info['score']))
    except Exception as e:
      logging.exception("Failed to load score in: {}".format(score_dir))
      logging.exception(e)
  return score_ls

def read_curve(args):
  """
    Fetch learning curve from learning-curve-*.png
  """
  curve_ls = []
  try:
    for i in range(DEFAULT_NUM_DATASET):
      curve_dir = join(args.input_dir, 'res_'+str(i+2))
      _img = glob(os.path.join(curve_dir,'learning-curve-*.png'))
      curve_ls.append(_img[0])
  except Exception as e:
    logging.exception("Failed to read curves.")
    logging.exception(e)
  return curve_ls

def write_score(score_ls, args):
  """
    Write scores to master phase scores.txt, as setj_score, where j = 1 to DEFAULT_NUM_DATASET
  """
  output_file = join(args.output_dir, 'scores.txt')
  try:
    with open(output_file, 'w') as f:
      for i in range(DEFAULT_NUM_DATASET):
        score_name = 'set{}_score'.format(i+1)
        score = score_ls[i]
        f.write("{}: {}\n".format(score_name, score))
  except Exception as e:
    logging.exception("Failed to write to" + output_file)
    logging.exception(e)
  return

def write_curve(curve_ls, args):
  """
    Write learning curves concatenated
  """
  filename = 'detailed_results.html'
  detailed_results_path = join(args.output_dir, filename)
  html_head = '<html><body><pre>'
  html_end = '</pre></body></html>'
  try:
    with open(detailed_results_path, 'w') as html_file:
      html_file.write(html_head)
      for id,image_path in enumerate(curve_ls):
        with open(image_path, "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read())
          encoded_string = encoded_string.decode('utf-8')
          t = '<font size="7">Dataset ' + str(id+1) + '.</font>'
          html_file.write(t + '<br>')
          s = '<img src="data:image/png;charset=utf-8;base64,%s"/>'%encoded_string
          html_file.write(s + '<br>')
      html_file.write(html_end)
  except Exception as e:
    logging.exception("Failed to write to" + detailed_results_path)
    logging.exception(e)
  return
  
  
################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
  try:
    # Logging version information and description
    logging.info('#' * 80)
    logging.info("Version: " + VERISION)
    logging.info(DESCRIPTION)
    logging.info('#' * 80)

  	# Get input and output dir from input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./test_input', 
                        help='where input results are stored')
    parser.add_argument('--output_dir', type=str, default='./test_output', 
                        help='where to store aggregated outputs')
    args = parser.parse_args()
    logging.debug("Parsed args are: " + str(args))
    logging.debug("-" * 80)

    # for DEBUG only
    print ("Copying input folder....")
    os.system("cp -R {} {}".format(join(args.input_dir, '*'), args.output_dir))

    if not os.path.exists(args.input_dir):
      logging.error("No input folder! Exit!")
      sys.exit()
    if not os.path.exists(args.output_dir):
      os.mkdir(args.output_dir)

	  # List the contents of the input directory (should be a bunch of resi/ subdirectories)
    input_ls = sorted(os.listdir(args.input_dir))
    logging.debug("Input dir contains: " + str(input_ls))

    # Check if we have correct results in input_dir/resi/ and copy default values otherwise
    validate_full_res(args)
    logging.info("[+] Results validation done.")
    logging.debug("-" * 80)
    logging.debug("Start aggregation...")

    # Read all scores from input_dir/resi/ subdirectories
    score_ls = read_score(args)
    logging.info("[+] Score reading done.")
    logging.debug("Score list: " + str(score_ls))

    # Aggregate all scores and write to output    
    write_score(score_ls, args)
    logging.info("[+] Score writing done.")

    # Read all learning curves
    curve_ls = read_curve(args)
    logging.info("[+] Learning curve reading done.")
    logging.debug("Curve list: " + str(curve_ls))

    # Aggregate all learning curves and write to output
    write_curve(curve_ls, args)
    logging.info("[+] Learning curve writing done.")

    logging.info("[+] Parent scoring program finished!")
    
  except Exception as e:
    logging.exception("Unexpected exception raised! Check parent scoring program!")
    logging.exception(e)