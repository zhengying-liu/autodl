FROM tensorflow/tensorflow:latest-gpu-py3

# Install useful Python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Additional packages demanded by participants
RUN pip install tensorflow_hub

WORKDIR /app/codalab
