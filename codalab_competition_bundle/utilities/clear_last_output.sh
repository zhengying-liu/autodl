#!/bin/bash
# This script clears possible output of last local execution

ROOT_DIR=$(pwd)
STARTING_KIT_DIR=$ROOT_DIR/../AutoDL_starting_kit/
cd $STARTING_KIT_DIR
rm -rf AutoDL_scoring_output
rm -rf AutoDL_sample_result_submission
ls | grep checkpoints_* | xargs rm -rf
cd $ROOT_DIR
