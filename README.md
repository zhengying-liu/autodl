# AutoDL
This repo contains the competition bundle (including sample data, starter code), a data formatting tutorial and other useful files of AutoDL challenge a new data science challenge aiming at pushing the state of the art in the field of Automatic Deep Learning (and more generally, Automatic Machine Learning). 

The directories correspond to:
- `codalab_competition_bundle`: the current competition bundle used on [CodaLab](http://35.193.242.121/competitions/8). It contains all the information to create a copy of the current version of the competition. For more information on how to create a competition on CodaLab, please refer to [this page](https://github.com/codalab/codalab-competitions/wiki#2-organizers). Important changes are to be made and the final version will be very different;
- `docker`: the Dockerfile used to generate the Docker image for the challenge;
- `how_to_format_data`: a tutorial on how to format participants' own data in this challenge's format, namely the AutoDL format (involving standard TensorFlow Record (TFRecord) format).

### How to participate?
The challenge is in full preparation and more details will be announced later.

### How to contribute data?
We encourage enterprises and research laboratories to format their own data and contribute to this challenge. In return, they can benefit from a direct machine learning solution for their own problems, after a competitive challenge of the state of the art. To contribute data, please follow the instructions in the GitHub repo [`autodl-contrib`](https://github.com/zhengying-liu/autodl-contrib) designed for this purpose.

### Usefuls links:
- Competitions on [CodaLab](https://competitions.codalab.org/competitions/)
- [Wiki of CodaLab Competitions](https://github.com/codalab/codalab-competitions/wiki)
- Info on [Protocol Buffers](https://developers.google.com/protocol-buffers/) and definition of [SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/core/example/example.proto) proto
