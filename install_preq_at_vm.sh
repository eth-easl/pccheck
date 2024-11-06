#!/bin/bash

# Basics
sudo apt-get update
sudo apt-get install -y build-essential

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get -y update
sudo apt-get -y install cuda-12-1

# Python
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.9-dev
echo "alias python=python3.9" >> $HOME/.bashrc
echo "alias python3=python3.9" >> $HOME/.bashrc
source $HOME/.bashrc
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py
sudo ln -s /usr/bin/python3.9 /usr/bin/python

sudo apt-get install -y libpmem-dev librpmem-dev libpmemblk-dev libpmemlog-dev libpmemobj-dev libpmempool-dev libpmempool-dev
sudo apt-get install -y libpci-dev

# Docker

sudo apt-get -y update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get -y update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get -y update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# sudo groupadd docker
# sudo usermod -aG docker $USER
# newgrp docker

python3.9 -m pip install -r requirements.txt

cd ~
git clone https://github.com/NVIDIA/apex && cd apex && git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82 && python3.9 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--global-option=--cpp_ext" --config-settings "--global-option=--cuda_ext" ./
cd ~

# deepspeed
sudo apt-get install -y pdsh ninja-build
