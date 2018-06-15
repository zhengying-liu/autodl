How to Format Data
========

This is a tutorial made for participants/organizers of AutoDL challenge on how to format data into the standard data format adopted by this challenge.

We strongly encourage enterprises and research laboratories to format their own data and contribute to this challenge, augmenting the database of datasets and making the challenge's judge on participants' algorithm more concrete, robust and convincing. Alsot in return, contributors of data can benefit from a direct machine learning solution for their own problems, after a competitive challenge of the state of the art. 

## What is the data format used in this challenge?

The data used in this challenge are in the standard TensorFlow format: [TFRecord](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data). According to the [official documentation]((https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)) of TensorFlow,
> the TFRecord file format is a simple [record-oriented](https://en.wikipedia.org/wiki/Record-oriented_filesystem) binary format that many TensorFlow applications use for training data.

Just as binary files can have follow different encoding and decoding formula, TFRecords can be obtained following different protocols, defined by Google's [Protocol Buffers](https://developers.google.com/protocol-buffers/). In this challenge, we use the **SequenceExample** protocol buffers defined [here](https://www.tensorflow.org/code/tensorflow/core/example/example.proto). **SequenceExample** proto allows us to format all types of data, i.e. both sequential and non-sequential, into a uniform format.

## An example of MNIST dataset formatted in SequenceExample proto
Here is one record (one image) in MNIST dataset formatted as a SequenceExample:
<pre><code>
context {
  feature {
    key: "id"
    value {
      int64_list {
        value: 0
      }
    }
  }
  feature {
    key: "label_index"
    value {
      int64_list {
        value: 7
      }
    }
  }
  feature {
    key: "label_score"
    value {
      float_list {
        value: 1.0
      }
    }
  }
}
feature_lists {
  feature_list {
    key: "0_dense_input"
    value {
      feature {
        float_list {
          value: 0.0
          value: 0.0
          value: 0.0
          value: 0.0
          value: 0.0
          value: 0.0
          value: 255.0
          value: 141.0
          value: 0.0
          [<em>...More pixel-wise numerical values</em>]
        }
      }
    }
  }
}
</code></pre>

We see that each sequence 

## Readings (IMPORTANT)
In order to understand what TFRecords are and how to work with them, we strongly recommend to read the following references:
- A [basic introduction](https://developers.google.com/protocol-buffers/docs/pythontutorial) on **Protocol Buffers** for Python programmers;
- After reading above introduction, you can find the definition of two important `proto`'s (short for Protocol Buffers) in the source code of TensorFlow:
  - [Feature](https://www.tensorflow.org/code/tensorflow/core/example/feature.proto) proto;
  - [Example](https://www.tensorflow.org/code/tensorflow/core/example/example.proto) proto, in which we find the extremely important definition of **SequenceExample** proto that we'll use in this challenge.
- The [Consuming TFRecord data](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data) section of TensorFlow's official documentation;
- Other blog articles on this topics, for example [this article](https://planspace.org/20170323-tfrecords_for_humans/).
After all these readings, you should already have a basic understanding of TFRecord format and SequenceExample. We review some important points in the following section. All the information can be found in above references.

## Review on TFRecord
[TODO]
