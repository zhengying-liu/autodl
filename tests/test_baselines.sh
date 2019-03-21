# Author: Zhengying LIU
# Date: 21 Mar 2019
# Description: This bash script tests the 2 baseline methods on 2 different
#   datasets included in the AutoDL challenge starting kit. The 2 baseline
#   methods are `all_zero` and `linear`, and the 2 datasets are `Monkeys`
#   and `miniciao`. This will make 2*2=4 tests.

STARTING_KIT_DIR='codalab_competition_bundle/AutoDL_starting_kit/'
SCRIPT=$STARTING_KIT_DIR'run_local_test.py'
DATASET_DIR_1=$STARTING_KIT_DIR'AutoDL_sample_data'
DATASET_DIR_2=$STARTING_KIT_DIR'AutoDL_another_sample_data'
CODE_DIR_1=$STARTING_KIT_DIR'AutoDL_simple_baseline_models/all_zero/'
CODE_DIR_2=$STARTING_KIT_DIR'AutoDL_simple_baseline_models/linear/'

python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_1
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_1
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_2
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_2
