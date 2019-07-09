# Author: Zhengying LIU
# Date: 21 Mar 2019
# Description: This bash script tests the 3 baseline methods on 2 different
#   datasets included in the AutoDL challenge starting kit. The 3 baseline
#   methods are `all_zero`, `linear` and `3dcnn`, and the 2 datasets are
#   `Monkeys` and `miniciao`. This will make 3*2=6 tests.

set -e

# Parent directory of the directory containing this script
REPO_DIR=$(dirname $(dirname $(realpath $0)))
STARTING_KIT_DIR=$REPO_DIR'/codalab_competition_bundle/AutoDL_starting_kit/'
SCRIPT=$STARTING_KIT_DIR'run_local_test.py'
SAMPLE_DATA_DIR=$STARTING_KIT_DIR'AutoDL_sample_data/'
DATASET_DIR_1=$SAMPLE_DATA_DIR'miniciao'
DATASET_DIR_2=$SAMPLE_DATA_DIR'Monkeys'
CODE_DIR_1=$STARTING_KIT_DIR'AutoDL_simple_baseline_models/all_zero/'
CODE_DIR_2=$STARTING_KIT_DIR'AutoDL_simple_baseline_models/linear/'
CODE_DIR_3=$STARTING_KIT_DIR'AutoDL_simple_baseline_models/3dcnn/'

python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_1
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_1
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_2
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_2
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_3
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_3
