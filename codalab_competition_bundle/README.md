This is an example AutoDL competition 

Prerequisites:
Install Anaconda Python 2.7

Usage:
Zip the contents and submit to https://competitions.codalab.org/competitions/create

Test:

(1) Submit:
AutoDL_sample_submission4ingestion.zip
as sample submission to the competition you just created.

(2) If you rather NOT use an ingestion program: go to the editor and select out the ingestion program from "Ingestion program organizer dataset:" (you will use --------, i.e. the default ingestion program). Then you can sumbmit AutoDL_sample_code_submission.zip

(3) To test the proper functioning of the sample code on your local computer:

- unzip all AutoDL*.zip
- run ingestion program:
python AutoDL_ingestion_program/ingestion.py AutoDL_input_data_1 AutoDL_sample_result_submission AutoDL_ingestion_program AutoDL_sample_submission4ingestion
- run the scoring program:
python AutoDL_scoring_program/score.py AutoDL_reference_data_1 AutoDL_sample_result_submission AutoDL_scoring_output

You can try both AutoDL_input_data_1 / AutoDL_reference_data_1 (first phase) and AutoDL_input_data_2 / AutoDL_reference_data_2 (second phase)