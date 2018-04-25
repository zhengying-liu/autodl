This script is going to create a zip file by 

1) creating a tmp/ directory

2) copying in tmp/ the following files/directories:

competition.yaml
*.jpg
*.html
assets/

3) zipping and moving to tmp/ the contents or the following dirs
and naming them xxx.zip

xxx=
AutoDL_input_data_1/
AutoDL_reference_data/
AutoDL_starting_kit/

4) zipping and moving to tmp/ the contents or the following dirs
AutoDL_starting_kit/xxx
xxx=
AutoDL_ingestion_program
AutoDL_scoring_program

and naming them xxx.zip