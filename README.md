# AutoDL
A data challenge in Automatic Deep Learning (AutoDL), co-organized with Google.


The directories correspond to:
- `codalab_competition_bundle`: the current competition bundle used on [CodaLab](http://35.193.242.121/competitions/8);
- `docker`: the Dockerfile used to generate the Docker image for the challenge;
- `how_to_format_data`: a tutorial on how to format participants' own data in this challenge's format, namely the standard TensorFlow TFRecords format.

### How to contribute
[See instructions](https://github.com/zhengying-liu/autodl/blob/master/CONTRIBUTING.md)

### To Prepare a Competition Bundle and Create a Copy of AutoDL competition
Please run
```
git clone https://github.com/zhengying-liu/autodl.git
cd autodl/codalab_competition_bundle/utilities/
./make_competition_bundle.sh
```
then you'll see a zip file created in the directory `utilities/`. Upload it to a
CodaLab server in the tag 'Create Competition' and bang!

### Usefuls links:
- Current version of competition on [CodaLab](http://35.193.242.121/competitions/8)
- Info on [Protocol Buffers](https://developers.google.com/protocol-buffers/)
- Definition of [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto
