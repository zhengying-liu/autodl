# Author: Zhengying LIU
# Date: 5 May 2019
# Description: This bash script tests 5 bad submissions on 2 different
#   datasets included in the AutoDL challenge starting kit. The 5 bad
#   submission methods are:
#     `all_zero_bad_prediction_shape`,
#     `exceeds_execution_time_limit`,
#     `exception_occurred_in_model`,
#     `exception_occurred_in_model_with_proba_0.5`,
#     `no_train_or_test`
#   The 2 datasets are `Monkeys` and `miniciao`. This will make 5*2=10 tests.

set -e

# Parent directory of the directory containing this script
REPO_DIR=$(dirname $(dirname $(realpath $0)))
STARTING_KIT_DIR=$REPO_DIR'/codalab_competition_bundle/AutoDL_starting_kit/'
SCRIPT=$STARTING_KIT_DIR'run_local_test.py'
SAMPLE_DATA_DIR=$STARTING_KIT_DIR'AutoDL_sample_data/'
DATASET_DIR_1=$SAMPLE_DATA_DIR'miniciao'
DATASET_DIR_2=$SAMPLE_DATA_DIR'Monkeys'
BAD_BASELINES_DIR=$REPO_DIR'/tests/'
CODE_DIR_1=$BAD_BASELINES_DIR'all_zero_bad_prediction_shape/'
CODE_DIR_2=$BAD_BASELINES_DIR'exceeds_execution_time_limit/'
CODE_DIR_3=$BAD_BASELINES_DIR'exception_occurred_in_model/'
CODE_DIR_4=$BAD_BASELINES_DIR'exception_occurred_in_model_with_proba_0.5/'
CODE_DIR_5=$BAD_BASELINES_DIR'no_train_or_test/'

python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_1
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_1
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_2 -time_budget=30
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_2 -time_budget=30
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_3
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_3
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_4 -time_budget=30
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_4 -time_budget=30
python $SCRIPT -dataset_dir=$DATASET_DIR_1 -code_dir=$CODE_DIR_5
python $SCRIPT -dataset_dir=$DATASET_DIR_2 -code_dir=$CODE_DIR_5
