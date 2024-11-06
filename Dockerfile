FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y libpmem-dev librpmem-dev libpmemblk-dev libpmemlog-dev libpmemobj-dev libpmempool-dev libpmempool-dev
RUN apt-get install -y libpci-dev
RUN apt-get install -y git vim curl unzip

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9-dev
RUN apt install -y python3.9-distutils
RUN echo "alias python=python3.9" >> $HOME/.bashrc
RUN echo "alias python3=python3.9" >> $HOME/.bashrc
RUN . $HOME/.bashrc
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN python3.9 -m pip install numpy==1.26.2
RUN python3.9 -m pip install accelerate==0.20.3
RUN python3.9 -m pip install torchvision==0.16.2
RUN python3.9 -m pip install wget nltk
RUN python3.9 -m pip install 'git+https://github.com/NVIDIA/dllogger'
RUN python3.9 -m pip install pynvml sacremoses evaluate
RUN python3.9 -m pip install deepspeed==0.12.6
RUN python3.9 -m pip install matplotlib
RUN python3.9 -m pip install lamb
RUN python3.9 -m pip install boto3

RUN cd ~
RUN git clone https://github.com/NVIDIA/apex && cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && python3.9 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--global-option=--cpp_ext" --config-settings "--global-option=--cuda_ext" ./
RUN cd ~

# gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

WORKDIR /root
