import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process

# Sender and Receiver are at the same group

class GeminiCheckpoint(object):

    def __init__(self, gpu_buffer, buffer_size, chunk_size_mb):
        self.spawned = False
        self.gpu_buffer = gpu_buffer
        self.buffer_size = buffer_size

        self.chunk_size_floats = int(chunk_size_mb*1e6/4)
        self.num_chunks = int(buffer_size/self.chunk_size_floats)

    def serialize_and_send(self, recv_rank):

        for i in range(self.num_chunks):
            dist.send(self.gpu_buffer[i*self.chunk_size_floats:(i+1)*self.chunk_size_floats], recv_rank)

        torch.cuda.synchronize()

    def send_checkpoint(self, base_rank, base_world_size, in_progress_snapshot, lock, change):

        os.environ['MASTER_ADDR'] = os.environ['GEMINI_MASTER_ADDR']
        os.environ['MASTER_PORT'] = os.environ['GEMINI_MASTER_PORT']
        print(f"At SENDER: {os.environ['GEMINI_MASTER_ADDR']}, {os.environ['GEMINI_MASTER_PORT']}")

        dist.init_process_group(
            backend='nccl',
            world_size=base_world_size*2,
            rank = base_rank*2
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"--------------------------- I am sender, world_size is {world_size}, rank is {rank}", flush=True)

        recv_rank = (rank + 3) % world_size

        # first, send total size
        sizet = torch.ones(1, dtype=torch.int64).cuda()
        sizet[0] = self.buffer_size
        dist.send(sizet, recv_rank)
        torch.cuda.synchronize()

        while True:

            with lock:
                if change.value == 0:
                    continue

            print(f"---- Send a new checkpoint, {self.num_chunks} chunks in total, chunk_size is {self.chunk_size_floats}", flush=True)
            self.serialize_and_send(recv_rank)
            print(f"---- Checkpoint SENT!", flush=True)

            with lock:
                in_progress_snapshot.value = 0
                change.value = 0


def create_receiver(chunk_size_mb, base_rank, base_world_size, fp16, lock, start_receiving):

    os.environ['MASTER_ADDR'] = os.environ['GEMINI_MASTER_ADDR']
    os.environ['MASTER_PORT'] = os.environ['GEMINI_MASTER_PORT']

    print(f"At RECEIVER: {os.environ['GEMINI_MASTER_ADDR']}, {os.environ['GEMINI_MASTER_PORT']}")

    dist.init_process_group(
        backend='nccl',
        world_size=base_world_size*2,
        rank = base_rank*2 + 1
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    send_rank = (rank-3) % world_size
    print(f"Receiver with rank {rank}, receive from {send_rank}", flush=True)

    # first, get total size
    sizet = torch.ones(1, dtype=torch.int64).cuda()
    dist.recv(sizet, send_rank)
    torch.cuda.synchronize()
    total_size = sizet[0]

    chunk_size_floats = int(chunk_size_mb*1e6/4)
    num_chunks = int(total_size/chunk_size_floats)

    print(f"Receiver with rank {rank}, will receive size {total_size}, num_chunks is {num_chunks}, chunk_size_floats is {chunk_size_floats}", flush=True)

    gpu_buffer = torch.ones(chunk_size_floats, dtype=torch.float32).cuda()
    cpu_buffer = torch.empty(
        total_size, dtype=torch.float32, pin_memory=True, device='cpu')

    while True:
        with lock:
            if start_receiving.value == 0:
                continue

        for i in range(num_chunks):
            dist.recv(gpu_buffer, send_rank)
            cpu_buffer[i*chunk_size_floats:(i+1)*chunk_size_floats].copy_(gpu_buffer)

        torch.cuda.synchronize()

        print(f"ALL RECEIVED", flush=True)

        with lock:
            start_receiving.value = 0


def save_checkpoint(chk, base_rank, base_world_size, in_progress_snapshot, lock, change):

    if not chk.spawned:
        print("------------- START A NEW GEMINI PROCESS!! ------------")

        chk.chk_process = \
            Process(target=chk.send_checkpoint, args=[base_rank, base_world_size, in_progress_snapshot, lock, change])
        chk.chk_process.start()
        chk.spawned = True
        print("-------------- GEMINI PROCESS STARTED!! ----------")

    if chk.chk_process is not None:
        while change.value == 1:
            # this means a checkpoint is on progress (wait for process doing the checkpoint to set variable to 0)
            continue

    # Once complete, initiate the next checkpoint synchronously
    with lock:
        in_progress_snapshot.value = 1
        change.value = 1