for filename in $(find . -name 'AutoDL_*' | grep -v '.zip'); do
  cd $filename;
  echo $filename;
  DATE=`date +%Y_%m_%d`;
  zip -o -r --exclude=*__pycache__* --exclude=*.DS_Store* "../$filename-$DATE" .;
  cd ..;
done
