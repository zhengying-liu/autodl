# AutoDL
A data challenge in Automatic Deep Learning (AutoDL), co-organized with Google.


The directories correspond to:
- `codalab_competition_bundle`: the current competition bundle used on [CodaLab](http://35.193.242.121/competitions/8);
- `docker`: the Dockerfile used to generate the Docker image for the challenge;
- `how_to_format_data`: a tutorial on how to format participants' own data in this challenge's format, namely the standard TensorFlow TFRecords format;
- `starting_kit_andre`: starting kit code given by André, slightly modified by Zhengying to make it work. This starting kit is the model that the current competition is based on. Thus we can find most codes in `codalab_competition_bundle` too;
- `starting_kit_lisheng`: starting kit code given by Lisheng in December 2017. Listed here just for archive.

### Remarks:
- Zhengying wrote the script `convert_mnist_to_tfrecords.py` (in the directory `starting_kit_andre/code_zhengying`) containing a snippet of code that generates TFRecords (SequenceExample proto) from original MNIST datasets. Experts in Protocol Buffers (e.g. @André) can check if Zhengying has used the good approach.

### To-do:
1. **(Done)** Replace the fake datasets `mnist1`, ..., `mnist5` by real datasets, e.g. `mnist1` by real MNIST, with train and test ;
2. **(Done)** Create a file `mnist_test.solution` containing real labels on test set and put it in the directory `AutoDL_reference_data/`;
3. **(Done)** Write a real estimator (neural network) as baseline model in the file `AutoDL_sample_code_submission/model.py`;
4. Improve codes in general by adding comments and turn them more user friendly. And possibly rewrite score.py (and maybe ingestion.py) in a more object-oriented manner;
5. **(Done)** Add Checkpoints feature to save and restore models;
6. **(Done)** Add feature of drawing learning curves;
7. Modify the API that will be provided to participants: 
  - Checkpoint feature for the participants so that they can pause and resume their training;
  - Candidate model: during the whole training process, one candidate model should be available at any moment to make predictions.Typically, a `save()` method should be called by the participant during training to update such candidate model and a `load()` method should be implemented by each participant such that the challenge backend can call it and make predictions *in parallel* with the training;
8. Finish "How to format data" tutorial;
9. Test *parallel ingestion/score* feature;
10. Test *several independent tracks* feautre (one dataset in each track, but in only 1 phase);
11. (to be added...)


### Usefuls links:
- Current version of competition on [CodaLab](http://35.193.242.121/competitions/8)
- Info on [Protocol Buffers](https://developers.google.com/protocol-buffers/)
- Definition of [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto
