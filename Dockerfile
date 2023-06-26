# syntax = docker/dockerfile:1.2
# Provision a server via: cap ec2 provision --name RobotVoice --region eu-central-1 --type m5.xlarge
FROM ubuntu:22.04 as wheels
EXPOSE 5000

LABEL Description="RobotVoice docker image" Version="1.0"

WORKDIR /

RUN apt-get update && apt-get -f -y install espeak python3-pip unzip git wget gunicorn

RUN git clone https://github.com/polvanrijn/RobotVoice
WORKDIR /RobotVoice


RUN python3.10 -m pip install --upgrade pip

# Setup vits
RUN git clone https://github.com/jaywalnut310/vits
WORKDIR /RobotVoice/vits

RUN pip3.10 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



# Download the VCTK checkpoint
RUN pip3.10 install gdown
RUN gdown 11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru

RUN mv ../setup_vits.py setup.py
RUN pip3.10 install -e .

RUN mv monotonic_align ../monotonic_align
WORKDIR /RobotVoice/monotonic_align
RUN pip3.10 install Cython
RUN python3.10 setup.py build_ext --inplace
WORKDIR /RobotVoice

RUN mkdir "VSTs"
WORKDIR /RobotVoice/VSTs
RUN wget https://tal-software.com/downloads/plugins/Tal-Vocoder-2_64_linux-2.2.1.zip
RUN unzip Tal-Vocoder-2_64_linux-2.2.1.zip
RUN mv TAL-Vocoder-2/* .
RUN rm Tal-Vocoder-2_64_linux-2.2.1.zip

WORKDIR /RobotVoice

COPY requirements.txt /RobotVoice/
RUN pip3.10 install -r requirements.txt

COPY . /RobotVoice/


# Run the server on deployment
#gunicorn server:app --worker-tmp-dir /dev/shm --workers=6 -b :5000 -t 600 --max-requests 30

# Run the server on local
#python server.py
