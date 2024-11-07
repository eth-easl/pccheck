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
import torch.distributed as dist

import ctypes
from checkpoint_eval.gemini.chk_manager import GeminiCheckpoint, save_checkpoint, create_receiver
from checkpoint_eval.pccheck_utils import initialize, set_storage

parser = argparse.ArgumentParser(description="CheckFreq microbenchmark")
parser.add_argument(
    "--size", default=1, type=int, help="size of the object to checkpoint (in MB)"
)
parser.add_argument("--iterations", default=10, type=int, help="iterations to simulate")
parser.add_argument("--rank", default=10, type=int, help="rank")
parser.add_argument("--world_size", default=10, type=int, help="world_size")
parser.add_argument('--master_ip', type=str, default='10.138.0.3', help='internal IP of VM running rank 0')
parser.add_argument('--master_port', type=str, default='1234', help='port of VM running rank 0')


def run(args):
    num_floats = args.size * 1000000 / 4
    print(f"allocate tensor of {int(num_floats)} floats")

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            tensor = torch.ones(int(num_floats), dtype=torch.float32)
            self.a = torch.nn.Parameter(tensor)

    model = TestModel()
    model.cuda()
    chunk_size_mb = min(32, args.size)
    gpu_ar, total_size = initialize(
        model, [], do_opt_step=False)
    print(f"----------------- total size is {total_size}")
    set_storage(model, [], gpu_ar)

    os.environ['MASTER_ADDR'] = args.master_ip
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    warmup = 3
    checkpoint_time_list = []
    for it in range(args.iterations):
        time.sleep(2)
        print(f"-------------------------- Start ITER {it}")

        dist.barrier()
        start_time = time.time()

        # emulate gemini
        if (rank==0):
            dist.send(gpu_ar, 1)
        else:
            dist.recv(gpu_ar, 0)
        torch.cuda.synchronize()

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)

        print(f"----------------- OUT CHECKPOINT {it} TOOK {duration} ms")

    print(f"AVERAGE Checkpoint time is {np.average(checkpoint_time_list)} ms")
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.sched_setaffinity(0, {0})
    args = parser.parse_args()
    run(args)
