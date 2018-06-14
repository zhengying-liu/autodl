How to Format Data
========

This is a tutorial made for participants/organizers of AutoDL challenge on how to format data into the standard data format adopted by this challenge.

We strongly encourage participants to format their own data and **share with the community**, enriching the challenge and making it even more challenging, fruitful and more concretely, solving directly the problems within the participants' hands (almost free).

## What is the data format adopted by this challenge?

The data used in this challenge are in the standard TensorFlow format: [TFRecords](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data). According to the official documentation of TensorFlow,
> the TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data.

As binary files can have follow different format, TFRecords can be obtained following different protocols, as in [Protocol Buffers](https://developers.google.com/protocol-buffers/). 


## Readings (IMPORTANT)
In order to understand what TFRecords are and how to work with them, we recommend to read:
- A [basic introduction](https://developers.google.com/protocol-buffers/docs/pythontutorial) on **Protocol Buffers** for Python programmers;
- After reading above introduction, you can find the definition of two important `proto`'s (short for Protocol Buffers) in the source code of TensorFlow:
  - [Feature](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/feature.proto) proto;
  - [Example](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto, in which we find the extremely important definition of **SequenceExample** proto that we'll use in this challenge.
