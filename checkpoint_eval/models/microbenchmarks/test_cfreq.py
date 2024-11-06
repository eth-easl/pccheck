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

from checkpoint_eval.checkfreq.chk_manager import CFCheckpoint
from checkpoint_eval.checkfreq.utils import save_checkpoint

parser = argparse.ArgumentParser(description="CheckFreq microbenchmark")
parser.add_argument(
    "--size", default=1, type=int, help="size of the object to checkpoint (in MB)"
)
parser.add_argument("--iterations", default=10, type=int, help="iterations to simulate")


def run(args):
    num_floats = args.size * 1000000 / 4
    print(f"allocate tensor of {int(num_floats)} floats")
    tensor = torch.ones(int(num_floats), dtype=torch.float32)
    tensor = tensor.cuda()
    tensor.share_memory_()
    print(tensor)

    model_dict = {"dummy_layer": tensor}
    chk = CFCheckpoint(model=model_dict)

    active_snapshot = Value("i", 0)
    lock = Lock()
    in_progress_snapshot = Value("i", 0)
    profile_snap = Value("i", 0)
    mp_manager = Manager()
    last_chk_it = Value("i", -1)
    change = Value("i", 0)
    filepath = mp_manager.Value(ctypes.c_wchar_p, "")  # self.mp_manager.dict()
    additional_snapshot = mp_manager.dict()

    warmup = 3
    checkpoint_time_list = []
    for it in range(args.iterations):
        time.sleep(2)

        start_time = time.time()
        save_checkpoint(
            "./checkpoint-{epoch}-{it}.chk",
            "",
            filepath,
            additional_snapshot,
            chk,
            active_snapshot,
            in_progress_snapshot,
            lock,
            0,
            it,
            last_chk_it,
            change,
            profile_snap,
            sync=False,
        )

        while in_progress_snapshot.value == 1:
            continue

        while change.value == 1:
            continue

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)

        print(f"CHECKPOINT {it} TOOK {duration} ms")

    if chk.chk_process is not None:
        chk.chk_process.kill()
        chk.chk_process.join()

    print(f"AVERAGE Checkpoint time is {np.average(checkpoint_time_list)} ms")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parser.parse_args()
    run(args)
