for filename in AutoDL_*; do
  cd $filename;
  echo $filename;
  DATE=`date +%Y_%m_%d`;
  zip -r --exclude=*__pycache__* --exclude=*.DS_Store* "../$filename-$DATE" .;
  cd ..;
done
