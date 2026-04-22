FROM nvcr.io/nvidia/pytorch:24.11-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y libpmem-dev libpmemblk-dev libpmemlog-dev libpmemobj-dev libpmempool-dev libpmempool-dev
RUN apt-get install -y libpci-dev
RUN apt-get install -y git vim curl unzip

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.12-dev
RUN echo "alias python=python3.12" >> $HOME/.bashrc
RUN echo "alias python3=python3.12" >> $HOME/.bashrc
RUN . $HOME/.bashrc
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.12 get-pip.py

RUN python3.12 -m pip install accelerate
RUN python3.12 -m pip install wget nltk
RUN python3.12 -m pip install 'git+https://github.com/NVIDIA/dllogger'
RUN python3.12 -m pip install pynvml sacremoses evaluate
RUN python3.12 -m pip install deepspeed==0.12.6
RUN python3.12 -m pip install matplotlib
RUN python3.12 -m pip install lamb
RUN python3.12 -m pip install boto3

WORKDIR /root
