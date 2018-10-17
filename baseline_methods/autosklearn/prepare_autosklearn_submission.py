dataname = 'credit' # change the data name!!
## Uncomment the right line in the following:
from autosklearn.classification import AutoSklearnClassifier as Model
# from autosklearn.regression import AutoSklearnRegressor as Model

model_dir = 'sample_code_submission/'
problem_dir = 'ingestion_program/'
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir);
import numpy as np

datadir = '../public_data'
from data_manager import DataManager
D = DataManager(dataname, datadir, replace_missing=True)

X_train = D.data['X_train']
Y_train = D.data['Y_train']

model = Model(time_left_for_this_task=1200) # Change the time budget!!!!
model.fit(X_train, Y_train)

Y_hat_valid = model.predict(D.data['X_valid'])
Y_hat_test = model.predict(D.data['X_test'])

result_name = 'sample_result_submission/' + dataname
from data_io import write
write(result_name + '_valid.predict', Y_hat_valid)
write(result_name + '_test.predict', Y_hat_test)

from subprocess import call
call(["zip", "-rj", "autosklearn", "sample_result_submission/"])
