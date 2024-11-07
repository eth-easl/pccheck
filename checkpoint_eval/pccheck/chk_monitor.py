import torch
import time
from checkpoint_eval.pccheck.chk_checkpoint_pipeline import Checkpoint
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, Barrier


class Chk_monitor:

    def __init__(
        self,
        c_lib_path,
        total_size,
        num_threads,
        max_async,
        gpu_copy,
        gpu_ar,
        ratio=2.0,
        is_sync=False,
        bsize=None,
        memory_saving=False,
        is_distributed=False,
        rank=0,
        world_size=1,
        **kwargs,
    ):

        # only 1 background process
        basic_path = "pccheck_checkpoint.chk"
        self.lock = Lock()
        self.cp_in_progress = Value("i", 0)
        self.start = Value("i", 0)
        self.stop = Value("i", 0)
        self.barrier = Barrier(2)

        self.checkpoint_dict = {}

        for name, ref in kwargs.items():
            self.checkpoint_dict[name] = ref

        print(f"BSIZE IS {bsize}")

        chk = Checkpoint(
            total_size,
            num_threads,
            basic_path,
            c_lib_path,
            max_async,
            ratio=ratio,
            gpu_ar=gpu_ar,
            bsize=bsize,
            memory_saving=memory_saving,
            is_distributed=is_distributed,
            rank=rank,
            world_size=world_size
        )

        self.chk_process = Process(
            target=chk.start_chk,
            args=[
                self.barrier,
                self.lock,
                self.checkpoint_dict,
                self.cp_in_progress,
                self.start,
                self.stop,
                gpu_copy,
                is_sync,
            ],
        )
        self.chk_process.start()
        self.barrier.wait()
        # print("Chk process started! PID is: ", self.chk_process.pid)

    def gpu_copy_in_progress(self):

        # return True if at least one of the background processes is copying
        with self.lock:
            if self.cp_in_progress.value == 1:
                return True

        return False

    def checkpoint_in_progress(self):
        with self.lock:
            if self.start.value == 1:
                return True

        return False

    def save(self):
        print(f"******************** CALL SAVE ********************")

        while True:
            with self.lock:
                if self.start.value == 0:
                    break

        with self.lock:
            self.cp_in_progress.value = 1
            self.start.value = 1

    def kill_checkpoint(self):

        with self.lock:
            self.stop.value = 1

        self.chk_process.join()
