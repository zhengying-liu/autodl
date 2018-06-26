# A Sample Dataset in AutoDL

This directory shows an example dataset used in AutoDL. In this example, the dataset (`MNIST`) has 4 components:
- metadata (`metadata.textproto`)
- training data (`mnist-train-*`)
- test data examples (`mnist-test-examples-*`)
- test data labels (`mnist-test-labels-*`)

All above 4 components are formatted into TFRecords, following [SequenceExample proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto#L292).
