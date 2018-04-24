for filename in AutoDL_*; do
  cd $filename;
  echo $filename;
  zip -r --exclude=*__pycache__* --exclude=*.DS_Store* ../$filename .;
  cd ..;
done
