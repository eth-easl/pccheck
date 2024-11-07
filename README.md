# PCcheck

PCcheck is a concurrent checkpoint mechanism for ML training. It is based on our ASPLOS'25 paper: "PCcheck: Persistent Concurrent Checkpointing for ML".

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Hardware Requirement](#hardware-requirement)
- [Hardware Configuration used in the paper](#hardware-configuration-used-in-the-paper)
- [Installation](#installation)
- [Example](#example)
- [ASPLOS'25 Artifact evaluation](#asplos25-artifact-evaluation)
- [Paper](#paper)

## Introduction

PCcheck is mechanism that allows frequent checkpointing for ML training workloads. The key idea behind PCcheck is to allow multiple concurrent checkpoints in parallel, thus allowing training to make progress and not stalling waiting for checkpoints to persist.

PCcheck optimizes copying and persisting a checkpoint by employing chunking and pipelining GPU-CPU copies and CPU-storage copies, and multiple threads for persisting to storage.

PCcheck optimizes the number of in-flight concurrent checkpoints, chunk size, and number of threads based on the training workload and system characteristics (e.g. storage bandwidth).

## Project Structure

The repo is structured as follows:

```
> tree .
├── checkpoint_eval
|   ├── checkfreq                  # Integration code for CheckFreq
|   ├── deepspeed                  # Necessary modifications for Deepseed
|   ├── gemini                     # Our implementation of Gemini
|   ├── gpm                        # Integration code for GPM
|   ├── models                     # Code and scripts for the models used in our evaluation
|   ├── pccheck                    # PCcheck implementation
├── artifact_evaluation            # Scripts and instructions for the ASPLOS'25 Artifact Evaluation
|   ├── evaluation                 # Scripts for reproducing key figures from the paper's evaluation sect
|   |   ├── sensitivity analysis   # Scripts for Figures 11, 12
|   |   ├── throughput             # Scripts for Figures 8, 9

```

## Hardware Configurations used in the paper

We used a2-highgpu-1g VMs from Google Cloud Platform. Each VM has an A100-40GB GPU attached, 1TB
of pd-ssd, 12 vCPUs, and 85 GB of DRAM.

## Installation

We used VMs with Ubuntu 20.04 (ubuntu-2004-focal-v20240830 from GCP).
We then used the [install_preq_at_vm.sh](install_preq_at_vm.sh) to install all required packages.
Finally, we run the [install.sh](install.sh) to build and install PCcheck and the rest baselines.

## Example

After installing, you can run [test_simple.sh](artifact_evaluation/test_simple.sh) to check everything is in place. This script trains a VGG16 model checkpointing every 50 iterations.

## ASPLOS'25 Artifact evaluation

We provide instructions for evaluating key results from our paper under the [artifact_evaluation](artifact_evaluation) directory.

## Paper

If you use PCcheck, please cite our paper:
```bibtex
@inproceedings {asplos25pccheck,
  author = {Strati Foteini and Friedman Michal and Klimovic Ana},
  title = {PCcheck: Persistent Concurrent Checkpointing for ML},
  booktitle = {},
  year = {2025},
  isbn = {},
  address = {},
  pages = {},
  url = {},
  doi = {},
  publisher = {Association for Computing Machinery},
}
```