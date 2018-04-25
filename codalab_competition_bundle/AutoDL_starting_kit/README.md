==================== This is an example AutoDL starting kit ====================

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
UNIVERSITE PARIS SUD, INRIA, CHALEARN, AND/OR OTHER ORGANIZERS 
OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES.

===== Usage:
Zip the contents of AutoDL_sample_code_submission (without the directory structure)

	zip mysubmission.zip AutoDL_sample_code_submission/*

and submit to Codalab competition "Participate>Submit/View results".


===== Local development and testing:

To make your own submission, modify AutoDL_sample_code_submission. You can then 
test it in the exact same environment as the Codalab environment using docker.

If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:

	docker run -it -u root -v $(pwd):/app/codalab evariste/autodl:dockerfile bash

You will then be able to run the ingestion program (to produce predictions) and the
scoring program (to evaluate your predictions) on toy sample data.
1) Ingestion program (using default directories):
	python AutoDL_ingestion_program/ingestion.py
	 
Eventually, substitute AutoDL_sample_data with other public data. The full call is:
	python AutoDL_ingestion_program/ingestion.py AutoDL_sample_data AutoDL_sample_result_submission AutoDL_ingestion_program AutoDL_sample_submission

2) Scoring program (using default directories):
	python AutoDL_scoring_program/score.py

The full call is:
	python AutoDL_scoring_program/score.py AutoDL_sample_data AutoDL_sample_result_submission AutoDL_scoring_output
