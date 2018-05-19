#!/bin/bash
# This script is going to create a zip file by
#
# 1) creating a tmp/ directory
#
# 2) copying in tmp/ the following files/directories:
#
# competition.yaml
# *.jpg
# *.html
# assets/
#
# 3) zipping and moving to tmp/ the contents or the following dirs
# and naming them xxx.zip
#
# xxx=
# AutoDL_input_data_1/
# AutoDL_reference_data/
# AutoDL_starting_kit/
#
# 4) zipping and moving to tmp/ the contents or the following dirs
# AutoDL_starting_kit/xxx
# xxx=
# AutoDL_ingestion_program
# AutoDL_scoring_program
#
# and naming them xxx.zip

DATE=`date "+%Y-%m-%d-%H-%M-%S"`
DIR='tmp/'
mkdir $DIR
cp '../'*.jpg $DIR
cp '../'*.html $DIR
cp '../'*.yaml $DIR
cp -r '../assets' $DIR
cd .. # codalab_competition_bundle/

# Begin zipping each data
for filename in $(find . -name 'AutoDL_*' | grep -v '.zip'); do
  cd $filename;
  echo $filename;
  zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../utilities/"$DIR$filename .;
  cd ..; # codalab_competition_bundle/
done

# Zipping ingestion and scoring program
cd AutoDL_starting_kit/
filename="AutoDL_ingestion_program"
cd $filename;
echo $filename;
echo $(pwd)
zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../../utilities/"$DIR$filename .;
cd ..; # AutoDL_starting_kit/
filename="AutoDL_scoring_program"
cd $filename;
echo $(pwd)
zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../../utilities/"$DIR$filename .;
cd ..; # AutoDL_starting_kit/

# Zip all to make a competition bundle
cd "../utilities/"$DIR
zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../AutoDL_competition_bundle_"$DATE .;
