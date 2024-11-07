# Hardware Dependencies

We have used a2-highgpu-1g VMs from Google Cloud, with 1 TB pd-ssd attached.
The VM specifications are:

* A100-40GB GPUs
* 24 vCPUs per machine
* 85 GB of DRAM
* 1 TB pd-ssd disk

# Software Dependencies

* GCC-9.4
* Python 3.9
* NVIDIA Driver 530.30.02
* CUDA 12.1
* NVIDIA Apex
* Torch 2.1
* Deepspeed 0.12.6
* Torchvision 0.14.1

# Datasets

* ImageNet
* SQUAD
* WikiText

# Models

We use the following models:

* VGG-16 from Torchvision
* BERT, Transformers-XL from [NVIDIA DLE](https://github.com/NVIDIA/DeepLearningExamples/tree/master)
* OPT and BLOOM from HuggingFace

# Setup

1. Create a VM with an A100 and Ubuntu 20.04. We have created an image with all packages installed. The image is [image-pccheck-ae-1](https://console.cloud.google.com/compute/imagesDetail/projects/pccheck-asplos25-ae/global/images/image-pccheck-ae-1?authuser=2&project=pccheck-asplos25-ae). For example, the following command creates a VM with the name *test-pccheck*.

```bash

 gcloud compute instances create test-pccheck --machine-type=a2-highgpu-1g --zone=us-west1-b --boot-disk-size 1000GB  --maintenance-policy TERMINATE --restart-on-failure --boot-disk-type pd-ssd --image image-pccheck-ae-1 --project pccheck-asplos25-ae.

```

2. Ssh to the VM, clone the pccheck repo.

```bash

gcloud compute ssh test-pccheck
git clone https://github.com/eth-easl/pccheck

```

3. Install PCcheck and the rest baselines.

```bash

cd pccheck
bash install.sh

```

Alternatively, we have provided the necessary packages to be installed in the [install_preq_at_vm.sh](../install_preq_at_vm.sh) script. This script has been tested only on Ubuntu 20.04! Our *image-pccheck-ae-1* image has been created by acquiring a VM with the *ubuntu-2004-focal-v20240830* image, and running the [install_preq_at_vm.sh](../install_preq_at_vm.sh) script.

# Run a simple example

To test installation, and run a simple example, you can run:  `bash test_simple.sh`

This runs a few training iterations of the VGG-16 model, using PCcheck to checkpoint every 50 iterations.

# Reproducing paper results

First, run `bash setup_models_and_datasets.sh`. This properly sets up models and datasets for evaluation.

We provided scripts to automate running experiments, collecting measurements and plotting.
After running each script, the csv files containing the results and the images are copied back in this directory.

To reduce evaluation time and costs, we focus on key figures from the paper.

## Figure 8 (~4 hours)
Step 1 generates Fig 8b.

1. Do `cd evaluation/throughput && bash get_throughput_single_node.sh`. This will generate csv files and plots for the Transformer model.

## Figure 9 (~ 10 min)
(NOTE: needs to be done after Figure 8, as it reuses information from Figure 8's results).

Do `cd evaluation/throughput && bash get_goodput.sh`. This will generate csv files and plots for each for 9b.

## Figure 11 (~ 35 min)

Do `cd evaluation/sensitivity_analysis && python run_microbenchmarks.py`. This will generate a `fig11.pdf` file.

## Figure 12 (~ 1 hour)

Do `cd evaluation/sensitivity_analysis && python run_pccheck_async.py`. This will generate a `fig12.pdf` file.