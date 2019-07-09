# Author: Zhengying LIU
# Date: 5 May 2019
# Description: This bash script tests the parent scoring program in
# AutoDL_parent_scoring_program using test_input/ as input. This folder
# test_input/ contains real outputs scratched from CodaLab platform.

set -e

# Parent directory of the directory containing this script
REPO_DIR=$(dirname $(dirname $(realpath $0)))
BUNDLE_DIR=$REPO_DIR'/codalab_competition_bundle/'
SCORING_PROGRAM_DIR=$BUNDLE_DIR'AutoDL_parent_scoring_program/'
SCORING_PROGRAM_SCRIPT=$SCORING_PROGRAM_DIR'evaluate.py'
INPUT_DIR=$SCORING_PROGRAM_DIR'test_input/'
OUTPUT_DIR=$SCORING_PROGRAM_DIR'test_output/'

python $SCORING_PROGRAM_SCRIPT --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR
