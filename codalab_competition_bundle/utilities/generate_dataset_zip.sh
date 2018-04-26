# This script creates zip files for each dataset facilitating
# the creation of datasets and the editing of competition on CodaLab

ROOT_DIR=$(pwd)
INPUT_DIR=$ROOT_DIR/../AutoDL_input_data_1/ # change this line to ...AutoDL_input_data_2 for phase 2
OUTPUT_DIR=$ROOT_DIR/datasets
mkdir $OUTPUT_DIR
cd $INPUT_DIR
for dataset in */; do
  cd $dataset;
  echo "Zipping $dataset"
  zip -r --exclude=*__pycache__* --exclude=*.DS_Store* $OUTPUT_DIR/${dataset%%/*}".zip" .;
  cd ..;
done
