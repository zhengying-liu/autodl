FROM tensorflow/tensorflow:latest-devel-py3

RUN apt-get update && apt-get install -y npm netcat nodejs-legacy

RUN pip install --upgrade pip==9.0.3  # higher version of pip will not support e.g. pip.get_installed_distributions()

COPY common.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app/codalab
