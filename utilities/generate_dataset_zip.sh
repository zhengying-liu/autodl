INPUT_DIR="AutoDL_input_data_1/"
cd $INPUT_DIR
for dataset in */; do
  cd $dataset;
  echo "Zipping $dataset"
  zip -r --exclude=*__pycache__* --exclude=*.DS_Store* ../${dataset%%/*} .;
  cd ..;
done
