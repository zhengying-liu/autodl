FROM tensorflow/tensorflow:latest-gpu-py3

# List of common Python packages installed in the Docker image used for this
# challenge. Participants are welcome to suggest other popular Python packages
# to be installed. If necessary, we'll update the Docker image to satisfy the
# need for most participants.
# In the case where you want to use less common packages, you can simply put
# all these packages in the the same folder of your submission (together with
# `model.py`) and the CodaLab platform should be able to find them.
RUN pip install numpy
RUN pip install pandas
RUN pip install jupyter
RUN pip install seaborn
RUN pip install scipy
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install pyyaml
RUN pip install psutil
RUN pip install h5py
RUN pip install keras
# CUDA 10 support for PyTorch
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision

# Additional packages demanded by participants
RUN pip install tensorflow_hub

WORKDIR /app/codalab
