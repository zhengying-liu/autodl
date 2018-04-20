# AutoDL
A data challenge in Automatic Deep Learning (AutoDL), co-organized with Google.


The three directories correspond to:
1. `codalab_competition_bundle`: the current competition bundle used on [CodaLab](http://35.193.242.121/competitions/8);
2. `starting_kit_andre`: starting kit code given by André, slightly modified by Zhengying to make it work. This starting kit is the model that the current competition is based on. Thus we can find most codes in `codalab_competition_bundle` too;
3. `starting_kit_lisheng`: starting kit code given by Lisheng in December 2017. Listed here just for archive.

To-do:
1. Replace the fake dataset `mnist1`, ..., `mnist5` by real datasets, e.g. `mnist1` by real MNIST, with train and test (done);
2. Create a file `mnist_test.solution` containing real labels on test set and put it in the directory `AutoDL_reference_data/`;
3. Write a real estimator (neural network) as baseline model in the file `AutoDL_sample_code_submission/model.py`
4. (to be added...)

To be discussed at Zurich:
1. Integrate test dataset to each data set? or train/test independently? Personnaly, I think it's more natural to have train and test in a single a dataset but to consider them separately as 2 datasets (which is what we have now);
2. Prediction for one single line? or for a matrix? For now, we added a method `Model.test()` to make prediction on the whole test set;
3. Evaluation by batch? We need a solution for faster evaluation
4. (to be added...)
