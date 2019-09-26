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
RUN pip install playsound
RUN pip install librosa
# CUDA 10 support for PyTorch
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision

# Packages to be activated: Following packages are demanded by one of the
# participants. If another participant asks to install one of these packages,
# we'll uncomment corresponding line and rebuild this image.
RUN pip install tensorflow_hub
# RUN pip install fastai
# RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

WORKDIR /app/codalab
