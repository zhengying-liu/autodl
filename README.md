# AutoDL
A data challenge in Automated Deep Learning (AutoDL), co-organized with Google.

[![Build Status](https://travis-ci.org/zhengying-liu/autodl.svg?branch=master)](https://travis-ci.org/zhengying-liu/autodl)

### How to contribute
[See instructions](https://github.com/zhengying-liu/autodl/blob/master/CONTRIBUTING.md)

### To Prepare a Competition Bundle and Create a Copy of AutoDL competition
Please run:
```
git clone https://github.com/zhengying-liu/autodl.git
cd autodl/codalab_competition_bundle/utilities/
./make_competition_bundle.sh
```
then you'll see a zip file created in the directory `utilities/`. Upload it to a
CodaLab server in the tag 'Create Competition' and bang!

### Usefuls links:
- Current version of competition on [CodaLab](https://autodl.lri.fr/)
- Info on [Protocol Buffers](https://developers.google.com/protocol-buffers/)
- Definition of [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto
