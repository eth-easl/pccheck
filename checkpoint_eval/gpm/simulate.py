import torch
import numpy as np
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
import torch.multiprocessing as mp

import argparse
import gpm_manager
import ctypes
import time


parser = argparse.ArgumentParser(description='Simulator')

parser.add_argument('--size', default=1, type=int,
                    help='size of the object to checkpoint (in MB)')
parser.add_argument('--sleep-fb', default=10, type=float,
                    help='simulate time for forward/backward (in ms)')
parser.add_argument('--sleep-opt', default=10, type=float,
                    help='simulate time for opt (in ms)')
parser.add_argument('--cfreq', default=10, type=int,
                    help='Checkpoint Frequency')
parser.add_argument("--iterations", default=100, type=int,
                    help='iterations to simulate')

args = parser.parse_args()


def simulate():

    # create model

    num_floats = args.size*1000000/4

    print(f"allocate tensor of {int(num_floats)} floats")
    tensor = torch.ones(int(num_floats), dtype=torch.float32)
    tensor = tensor.cuda()
    tensor.share_memory_()

    model_dict = {'dummy_layer': tensor}

    chk = gpm_manager.GPMCheckpoint(
        f"checkpoint_gpm",
        model=model_dict['dummy_layer'],
        datasize=int(num_floats * 4)
    )
    print(chk)
    last_chk_it = Value('i', -1)

    batch_idx = 0
    steps_since_checkp = 0
    checkpoints = 0

    start_train_time = time.time()

    while batch_idx < args.iterations:

        print(f"Iter {batch_idx}")

        # simulate forward-backward
        time.sleep(args.sleep_fb*0.001)

        # simulate optimization step
        time.sleep(args.sleep_opt*0.001)

        if ((args.cfreq > 0) and steps_since_checkp == args.cfreq-1):
            chk._checkpoint(iter_chk=last_chk_it)
            steps_since_checkp = 0
            checkpoints += 1
        else:
            steps_since_checkp += 1

        batch_idx += 1

    end_train_time = time.time()
    total_train_time = end_train_time - start_train_time
    print(
        f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, Number of iterations: {batch_idx}, Number of checkpoints: {checkpoints}")


if __name__ == "__main__":
    args = parser.parse_args()
    mp.set_start_method("spawn")
    simulate()
