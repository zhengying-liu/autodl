# for filename in AutoDL_*; do
#   cd $filename;
#   echo $filename;
#   DATE=`date +%Y_%m_%d`;
#   zip -r --exclude=*__pycache__* --exclude=*.DS_Store* "../$filename-$DATE" .;
#   cd ..;
# done

INPUT_DIR="AutoDL_input_data_1/"
cd $INPUT_DIR
for dataset in */; do
  cd $dataset;
  echo "Zipping $dataset"
  zip -r --exclude=*__pycache__* --exclude=*.DS_Store* ../${dataset%%/*} .;
  cd ..;
done
