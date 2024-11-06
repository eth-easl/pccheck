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
from checkpoint_eval.pccheck.chk_monitor import Chk_monitor
from checkpoint_eval.pccheck_utils import initialize, get_total_size, set_storage

# preprocessing sources:
# https://github.com/pytorch/examples/blob/main/imagenet/main.py

parser = argparse.ArgumentParser(
    description="PyTorch ImageNet/CIFAR Training or inference using torchvision models"
)

parser.add_argument("--arch", default="resnet18", type=str, help="torchvision model")
parser.add_argument(
    "--batchsize", default=128, type=int, help="batch size for training"
)
parser.add_argument("--train_dir", default="", type=str, help="path to dataset")
parser.add_argument(
    "--dataset", default="cifar", type=str, help="type of dataset (cifar or imagenet)"
)
parser.add_argument("--cfreq", default=0, type=int, help="Checkpoint Frequency")
parser.add_argument(
    "--max-async",
    default=1,
    type=int,
    help="Number of Python processes responsible for checkpointing",
)
parser.add_argument(
    "--num-threads", default=1, type=int, help="Number of CPU threads writing at NVM"
)
parser.add_argument(
    "--psize", default=1, type=int, help="Number of chunks for pipeline"
)
parser.add_argument(
    "--c_lib_path",
    default="",
    type=str,
    required=True,
    help="path to the libtest.so library",
)
parser.add_argument(
    "--bench_total_steps", default=1000, type=int, help="Number of steps to train for"
)

args = parser.parse_args()


def train():

    print(f"Process with pid {os.getpid()}, args is {args}")

    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = models.__dict__[args.arch](
        num_classes=1000 if args.dataset == "imagenet" else 10
    )
    model = model.to(local_rank)  # to GPU

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)
    metric_fn = torch.nn.CrossEntropyLoss().to(0)

    model.train()

    train_dir = args.train_dir

    # for checkpoint
    mp.set_start_method("spawn", force=True)
    gpu_ar, total_size = initialize(model, [optimizer])

    start_idx = 0
    # assume gpu_ar is big 1D GPU tensor
    set_storage(model, [optimizer], gpu_ar)
    torch.cuda.empty_cache()
    if args.cfreq > 0:
        chk_monitor = Chk_monitor(
            args.c_lib_path,
            total_size,
            args.num_threads,
            args.max_async,
            True,
            gpu_ar=gpu_ar,
            bsize=total_size,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            memory_saving=True,
        )
    else:
        chk_monitor = None

    batch_idx = 0
    steps_since_checkp = 0
    checkpoints = 0
    warmup = 3

    start_train_time = time.time()

    batch_idx = 0
    batch = (
        torch.ones([args.batchsize, 3, 224, 224]),
        torch.ones([args.batchsize], dtype=torch.long),
    )

    start_iter = time.time()
    while batch_idx < args.bench_total_steps:

        # NOTE: this increases mem footprint
        optimizer.zero_grad()
        data, target = batch[0].to(local_rank), batch[1].to(local_rank)

        if args.arch == "inception_v3":
            output, _ = model(data)
        else:
            output = model(data)

        loss = metric_fn(output, target)
        loss.backward()

        if chk_monitor:
            while chk_monitor.gpu_copy_in_progress():
                continue

        optimizer.step()

        # FOR CHECKING
        # grads = []
        # for group in optimizer.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             grads.append(p.grad[0][0])
        #             break

        if (batch_idx == warmup) or (
            (args.cfreq > 0) and steps_since_checkp == args.cfreq - 1
        ):
            if args.cfreq > 0:
                chk_monitor.save()
                steps_since_checkp = 0
                checkpoints += 1
            if batch_idx == warmup:
                print(f"Start clock!")
                start_train_time = time.time()
        else:
            steps_since_checkp += 1

        batch_idx += 1
        print(f"Step {batch_idx} took {time.time()-start_iter}")

        start_iter = time.time()

    end_train_time = time.time()
    total_train_time = end_train_time - start_train_time
    if chk_monitor:
        chk_monitor.kill_checkpoint()

    print(
        f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, Number of iterations: {batch_idx}, Number of checkpoints: {checkpoints}"
    )
    print(f"EXECUTION TIME: {total_train_time} sec")
    print(f"THROUGHPUT IS {(args.bench_total_steps-warmup)/total_train_time}")


if __name__ == "__main__":
    args = parser.parse_args()
    os.sched_setaffinity(0, {0})
    train()
