""" Basically most vision models. """

import os
from platform import node
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse
import ctypes

from checkpoint_eval.gpm.gpm_manager import GPMCheckpoint
from checkpoint_eval.pccheck_utils import initialize,get_total_size


# preprocessing sources:
# https://github.com/pytorch/examples/blob/main/imagenet/main.py

parser = argparse.ArgumentParser(description='PyTorch ImageNet/CIFAR Training or inference using torchvision models')

parser.add_argument('--arch', default='resnet18', type=str, help='torchvision model')
parser.add_argument('--batchsize', default=128, type=int, help='batch size for training')
parser.add_argument('--train_dir', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar', type=str, help='type of dataset (cifar or imagenet)')
parser.add_argument('--cfreq', default=10, type=int, help='Checkpoint Frequency')
parser.add_argument('--bench_total_steps', default=1000, type=int, help='Number of steps to train for')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}-{it}.chk', help='checkpoint file format - for CheckFreq')
parser.add_argument("--path-to-pmem", default="", type=str, help='if specified, the checkpoint will be written to pmem, at this path')

def train():

    print(f"Process with pid {os.getpid()}, args is {args}")

    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = models.__dict__[args.arch](num_classes=1000 if args.dataset=='imagenet' else 10)
    model = model.to(local_rank) # to GPU

    optimizer =  torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)
    metric_fn = torch.nn.CrossEntropyLoss().to(0)

    print("Configure dataset")

    train_dir = args.train_dir

    # for checkpoint

    steps_since_checkp = 0
    checkpoints = 0
    warmup = 3

    total_size = get_total_size(model, [optimizer])
    print(f"total size is {total_size*4}")

    chk = GPMCheckpoint(
            f"checkpoint_gpm.chk",
            model=model,
            optimizer=optimizer,
            datasize = int(total_size*4) # use float
    )

    # start
    for i in range(1): # assume one epoch
        print("Start epoch: ", i)

        model.train()

        start = time.time()
        start_iter = time.time()

        batch_idx = 0
        batch = (torch.rand([args.batchsize, 3, 224, 224]), torch.ones([args.batchsize],dtype=torch.long))

        start_train_time = time.time()

        while batch_idx < args.bench_total_steps:

            print(f"Iter {batch_idx}")
            optimizer.zero_grad()
            data, target = batch[0].to(local_rank), batch[1].to(local_rank)

            if args.arch=='inception_v3':
                output, _ = model(data)
            else:
                output = model(data)

            loss = metric_fn(output, target)
            loss.backward()

            optimizer.step()

            if ((batch_idx == warmup) or ((args.cfreq > 0) and steps_since_checkp == args.cfreq-1)):
                if args.cfreq > 0:
                    chk._checkpoint(iter_chk=batch_idx)
                    steps_since_checkp = 0
                    checkpoints += 1
                if batch_idx == warmup:
                    print(f"Start clock!")
                    start_train_time = time.time()
            else:
                steps_since_checkp += 1

            batch_idx += 1

        end_train_time = time.time()
        total_train_time = end_train_time - start_train_time
        chk._finish()
        print(f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, Number of iterations: {batch_idx}, Number of checkpoints: {checkpoints}")
        print(f"EXECUTION TIME: {total_train_time} sec")
        print(f"THROUGHPUT IS {(args.bench_total_steps-warmup)/total_train_time}")

if __name__ == "__main__":
    args = parser.parse_args()
    train()
