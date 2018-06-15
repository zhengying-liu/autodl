How to Format Data
========

This is a tutorial made for participants/organizers of AutoDL challenge on how to format data into the standard data format adopted by this challenge.

We strongly encourage enterprises and research laboratories to format their own data and contribute to this challenge, augmenting the database of datasets and making the challenge's judge on participants' algorithm more concrete, robust and convincing. Alsot in return, contributors of data can benefit from a direct machine learning solution for their own problems, after a competitive challenge of the state of the art. 

## What is the data format adopted by this challenge?

The data used in this challenge are in the standard TensorFlow format: [TFRecord](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data). According to the official documentation of TensorFlow,
> the TFRecord file format is a simple [record-oriented](https://en.wikipedia.org/wiki/Record-oriented_filesystem) binary format that many TensorFlow applications use for training data.

Just as binary files can have follow different encoding and decoding methods, TFRecords can be obtained following different protocols, defined by Google's [Protocol Buffers](https://developers.google.com/protocol-buffers/).

[TODO]: Add an example here, of type:
<pre><code>
context: {
  feature: {
    key  : "id"
    value: {
      bytes_list: {
        value: [<em>Video id. Can be translated to YouTube ID (link).</em>]
      }
    }
  }
  feature: {
    key  : "labels"
      value: {
        int64_list: {
          value: [1, 522, 11, 172] # The meaning of the labels can be found here.
        }
      }
    }
}

feature_lists: {
  feature_list: {
    key  : "rgb"
    value: {
      feature: {
        bytes_list: {
          value: [<em>1024 8bit quantized features</em>]
        }
      }
      feature: {
        bytes_list: {
          value: [<em>1024 8bit quantized features</em>]
        }
      }
      ... # Repeated for every second of the video, up to 300
  }
  feature_list: {
    key  : "audio"
    value: {
      feature: {
        bytes_list: {
          value: [<em>128 8bit quantized features</em>]
        }
      }
      feature: {
        bytes_list: {
          value: [<em>128 8bit quantized features</em>]
        }
      }
    }
    ... # Repeated for every second of the video, up to 300
  }

}
</code></pre>


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
