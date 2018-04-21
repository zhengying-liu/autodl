# AutoDL
A data challenge in Automatic Deep Learning (AutoDL), co-organized with Google.


The directories correspond to:
- `codalab_competition_bundle`: the current competition bundle used on [CodaLab](http://35.193.242.121/competitions/8);
- `docker`: the Dockerfile used to generate the Docker image for the challenge;
- `starting_kit_andre`: starting kit code given by André, slightly modified by Zhengying to make it work. This starting kit is the model that the current competition is based on. Thus we can find most codes in `codalab_competition_bundle` too;
- `starting_kit_lisheng`: starting kit code given by Lisheng in December 2017. Listed here just for archive.

### Remarks:
- Zhengying wrote the script `convert_mnist_to_tfrecords.py` (in the directory `starting_kit_andre/code_zhengying`) containing a snippet of code that generates TFRecords (SequenceExample proto) from original MNIST datasets. Experts in Protocol Buffers (e.g. @André) can check if Zhengying has used the good approach.

### To-do:
1. Replace the fake datasets `mnist1`, ..., `mnist5` by real datasets, e.g. `mnist1` by real MNIST, with train and test (done);
2. Create a file `mnist_test.solution` containing real labels on test set and put it in the directory `AutoDL_reference_data/`;
3. Write a real estimator (neural network) as baseline model in the file `AutoDL_sample_code_submission/model.py`;
4. Improve codes in general by adding comments and turn them more user friendly;
5. Add Checkpoints feature to save and restore models;
6. (to be added...)

### To be discussed at Zurich:
1. Integrate test dataset to each data set? or train/test independently? Personnaly, I think it's more natural to have train and test in a single a dataset but to consider them separately as 2 datasets (which is what we have now);
2. Prediction for one single line? or for a matrix? For now, we added a method `Model.test()` to make prediction on the whole test set;
3. Evaluation by batch? We need a solution for faster evaluation
4. Use tf.estimator.Estimator instead of Model()? 
5. Sparse to dense? More adaptive handling for sparse data?
6. AutoDLDataset -> (features, labels) where both features and labels are tensors. But TensorFlow convention seems to have features as a dict;
7. (to be added...)

### Usefuls links:
- Current version of competition on [CodaLab](http://35.193.242.121/competitions/8)
- Info on [Protocol Buffers](https://developers.google.com/protocol-buffers/)
- Definition of [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto
