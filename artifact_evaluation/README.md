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

# Installation

1. Create a VM with an A100 and Ubuntu 20.04. We used a2-highgpu-1g VMs from GCP.

2. Ssh to the VM, clone the repo and cd

3. Run `bash install_preq_at_vm.sh` to install the necessary basics at the VM. This step installs gcc, NVIDIA Drivers, CUDA, and Python

4. Reboot the VM, and ssh again

Alternatively, we have created a GCP image (image-pccheck-ae-0), with all packages required up to this step already installed.

5. `cd pccheck/ && bash install.sh`

# Run a simple example

To test installation, and run a simple example, you can run:  `bash test_simple.sh`

This runs a few training iterations of the VGG-16 model, using PCcheck to checkpoint every 50 iterations.

# Reproducing paper results

First, run `bash setup_models_and_datasets.sh`

We provided scripts to automate running experiments, collecting measurements and plotting.
After running each script, the csv files containing the results and the images are copied back in this directory.

## Figure 8
We recommend step 1, which generates Fig 8b, 8c, 8d from the paper.

1. Single-node experiments: Do `cd evaluation/throughput && bash get_throughput_single_node.sh`. This will generate csv files and plots for each model.

2. [OPTIONAL!] Multi-node experiments: For multi-node experiments, these steps must be followed (shown for the 2-VM example of Fig 8e):
    * Create two VMs, vm1 and vm2. The VMs should be on the same cloud zone. We name IP1 the **INTERNAL_IP** of vm1, and IP2 the **INTERNAL_IP** of vm2.
    * SSH to vm1 (`gcloud compute ssh vm1`)
    * # IMPORTANT! Make sure vm1 has ssh-access to vm2.
      To do so, from within vm1:
        * `gcloud compute init`: to authenticate
        * `gcloud compute ssh vm2`: to allow ssh to vm2 and generate ssh keys. Then `exit` to get back to vm1's shell. This step should have generated keys under `~/.ssh`
        * Create an ~/.ssh/config file and put two entries, one for vm1, and one for vm2. It should look as follows:

        ```
        Host IP1
        HostName IP1
        User "your_username"
        IdentityFile ~/.ssh/google_compute_engine

        Host IP2
        HostName IP2
        User "your_username"
        IdentityFile ~/.ssh/google_compute_engine

        ```
        * Make sure `ssh IP1` and `ssh IP2` work from within vm1 without issues.
    * Create *hostfile*. This is used by Deepspeed to launch distributed workers. It should look like:
        ```
        IP1 slots=1
        IP2 slots=2
        ```
    * Run bash `get_throughput_multi_node.sh IP1 IP2`. This file:
        * Creates a `hostfile` (used by DeepSpeed to start execution at the two nodes) and a `~/.deepspeed_env` file (containing necessary env variables).
        * Runs all baselines for OPT-2.7B, generates csv files and plots for each model.


## Figure 9

Do `cd evaluation/throughput && bash get_goodput.sh`. This will generate csv files and plots for each model for figures 9b, 9c, 9d.

## Figure 11

Do `cd evaluation/sensitivity_analysis && python run_microbenchmarks.py`. This will generate a `fig11.pdf` file

## Figure 12

Do `cd evaluation/sensitivity_analysis && python run_pccheck_async.py`. This will generate a `fig12.pdf` file

## Figure 13

Do `cd evaluation/sensitivity_analysis && python run_pccheck_threads.py`. This will generate a `fig13.pdf` file
