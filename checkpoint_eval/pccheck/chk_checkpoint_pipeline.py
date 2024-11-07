import torch
import time
from ctypes import *
from math import ceil
import numpy as np
from threading import Thread, Lock, Barrier, get_native_id
import os


class Writer(object):

    # constructor
    def __init__(self, fname, lib_path, max_async, bsize, num_cpu_batches, is_distributed, rank, world_size):
        # attribute
        self.lib = cdll.LoadLibrary(lib_path)
        print(bsize, num_cpu_batches, is_distributed)
        self.writer_obj = self.lib.writer(
            fname, max_async, c_size_t(bsize), num_cpu_batches, is_distributed, rank, world_size
        )

    def write_array(self, x, sz, num_threads):
        ct_arr = np.ctypeslib.as_ctypes(x)
        self.lib.savenvm(self.writer_obj, ct_arr, c_ulong(sz), num_threads)

    def finish(self):
        self.lib.finish()

    def register(self):
        self.lib.registerCheck.restype = c_int
        return self.lib.registerCheck(self.writer_obj)

    def get_cpu_address(self):
        self.lib.take_cpu_address.restype = c_void_p
        tid = get_native_id() % 128  # from MAX_THREADS In FAAArrayQueueAdd
        return self.lib.take_cpu_address(self.writer_obj, tid)

    def savenvm_new(
        self,
        arr,
        total_size,
        num_threads,
        checkp_info,
        batch_num,
        batch_size,
        last_batch,
    ):
        ct_arr = np.ctypeslib.as_ctypes(arr)
        batch_size = int(batch_size)

        self.lib.savenvm_new.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_int,
            c_void_p,
            c_int,
            c_size_t,
            c_bool,
        ]
        # print(f"num_threads {num_threads}, checkp_info {checkp_info}, batch_num {batch_num}, batch_size {batch_size}, last_batch {last_batch}")
        tid = get_native_id() % 128  # from MAX_THREADS In FAAArrayQueueAdd
        self.lib.savenvm_new(
            self.writer_obj,
            tid,
            ct_arr,
            total_size,
            num_threads,
            checkp_info,
            batch_num,
            c_size_t(batch_size),
            last_batch,
        )


class Checkpoint:

    def __init__(
        self,
        total_size,
        num_threads,
        filename,
        lib_path,
        max_async,
        ratio=2.0,
        gpu_ar=None,
        bsize=None,
        memory_saving=False,
        is_distributed=False,
        rank=0,
        world_size=1
    ):

        self.total_size = total_size
        self.filename = filename
        self.num_threads = num_threads
        self.max_async = max_async
        self.lib_path = lib_path
        self.memory_saving = memory_saving
        self.ratio = ratio
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size

        self.gpu_ar = gpu_ar
        if self.gpu_ar is not None:
            self.gpu_ar_given = True
        else:
            self.gpu_ar_given = False

        if bsize is None:
            self.bsize = total_size
        else:
            self.bsize = bsize

        print(f"At checkpoint, bsize is {self.bsize}")

        self.chk_times = []

    def write_array(self, x, sz, num_threads):
        print("Write checkpoint of size ", sz)
        start = time.time()
        self.writer.write_array(x, sz, num_threads)
        chk_time = time.time() - start

        self.timing_lock.acquire()
        self.timing_list.append(chk_time)
        self.timing_lock.release()
        print("Writing checkpoint to NVM took: ", chk_time)

    def write_batch(
        self,
        prev_barrier,
        next_barrier,
        cpu_ar,
        start_idx,
        end_idx,
        num_threads,
        total_size,
        batch_num,
        batch_size,
        is_last_batch,
        ret,
        lock,
        cp_in_progress,
    ):
        if prev_barrier is not None:
            prev_barrier.wait()

        gpu_copy_start = time.time()
        # copy
        gpu_ar_new = self.gpu_ar[start_idx:end_idx]

        if cpu_ar is not None:
            cpu_ar_new = cpu_ar[start_idx:end_idx]
            cpu_ar_np = cpu_ar_new.numpy()
        else:
            # TODO: check sizes, check tensor-to-numpy, pinning
            cpu_ar_ptr = self.writer.get_cpu_address()
            cpu_ar_ptr = cast(cpu_ar_ptr, POINTER(c_float))
            cpu_ar_np = np.ctypeslib.as_array(cpu_ar_ptr, shape=(int(batch_size),))
            cpu_ar_new = torch.from_numpy(cpu_ar_np)

        cpu_ar_new.copy_(gpu_ar_new)
        gpu_copy_end = time.time()
        print(f"GPU copy took {(gpu_copy_end-gpu_copy_start)*1000} ms")

        if next_barrier is not None:
            next_barrier.wait()

        # copy for this checkpoint has finished
        if is_last_batch:
            with lock:
                cp_in_progress.value = 0

        cpu_copy_start = time.time()
        # save nvm
        self.writer.savenvm_new(
            cpu_ar_np,
            total_size,
            num_threads,
            ret,
            batch_num,
            batch_size,
            is_last_batch,
        )

        cpu_copy_end = time.time()
        print(f"CPU copy took {(cpu_copy_end-cpu_copy_start)*1000} ms")

    def write_pipelined(self, cpu_ar, num_threads, sz, bsize, lock, cp_in_progress):

        num_batches = int(sz / bsize)
        chk_ret = self.writer.register()
        print(f"************** New checkpoint at position {chk_ret}")
        start = time.time()

        barriers = [Barrier(2) for _ in range(num_batches)]
        threads = []

        for i in range(num_batches):

            if i == 0:
                prev_barrier = None
            else:
                prev_barrier = barriers[i - 1]

            if i == num_batches - 1:
                next_barrier = None
            else:
                next_barrier = barriers[i]

            start_idx = int(i * bsize)
            end_idx = min(sz, int((i + 1) * bsize))

            threads.append(
                Thread(
                    target=self.write_batch,
                    args=(
                        prev_barrier,
                        next_barrier,
                        cpu_ar,
                        start_idx,
                        end_idx,
                        num_threads,
                        sz,
                        i + 1,
                        bsize,
                        i == num_batches - 1,
                        chk_ret,
                        lock,
                        cp_in_progress,
                    ),
                )
            )

        for i in range(num_batches):
            threads[i].start()

        for i in range(num_batches):
            threads[i].join()

        total_time = time.time() - start
        self.chk_times.append(total_time)
        print(
            f"---------------------- [PERF] Single Checkpoint time is {time.time()-start} sec, average is {np.median(self.chk_times)}"
        )

    def convert_1d(self, gpu_ar, checkpoint_dict, idx):
        for name, obj in checkpoint_dict.items():
            if torch.is_tensor(obj):
                t = obj.flatten().to(torch.float32)
                sz = t.shape[0]
                gpu_ar[idx : idx + sz] = t  # this will create a memory copy
                idx += sz
            elif isinstance(obj, dict):
                idx = self.convert_1d(gpu_ar, obj, idx)
            elif type(obj) == int or type(obj) == float:
                gpu_ar[idx] = float(obj)
                idx += 1
        return idx

    def start_chk(
        self,
        barrier,
        lock,
        checkpoint_dict,
        cp_in_progress,
        start1,
        stop,
        gpu_copy=True,
        is_sync=False,
    ):

        os.sched_setaffinity(0, {1})

        # GOAL: same mem footprint as GPM/CheckFreq
        print(f"total_size: {self.total_size}, bsize: {self.bsize}, ratio: {self.ratio}")
        total_mem_batches = int(self.ratio * self.total_size / self.bsize)
        self.writer = Writer(
            self.filename.encode(),
            self.lib_path,
            self.max_async,
            int(self.bsize),
            total_mem_batches,
            self.is_distributed,
            self.rank,
            self.world_size
        )

        self.timing_lock = Lock()
        self.timing_list = []

        self.threads = [None for _ in range(self.max_async)]
        self.cpu_ar_list = []

        if not self.gpu_ar_given:
            self.gpu_ar = torch.ones(self.total_size, dtype=torch.float32)
            self.gpu_ar = self.gpu_ar.cuda()

        if not self.memory_saving:
            for i in range(self.max_async):
                self.cpu_ar_list.append(
                    torch.empty(
                        self.total_size,
                        dtype=torch.float32,
                        pin_memory=True,
                        device="cpu",
                    )
                )

        else:
            print("SAVE MEM!")

        torch.cuda.empty_cache()
        barrier.wait()

        while True:
            with lock:
                if stop.value == 1:
                    break

            with lock:
                if start1.value == 0:
                    continue

            # snapshot here
            stime = time.time()

            # copy all in the same structure
            if not self.gpu_ar_given:
                flattened_size = self.convert_1d(self.gpu_ar, checkpoint_dict, 0)
                print(flattened_size, self.total_size)
                assert flattened_size == self.total_size

            if is_sync:
                self.write_pipelined(
                    self.cpu_ar_list[i] if not self.memory_saving else None,
                    self.num_threads,
                    self.total_size,
                    self.bsize,
                    lock,
                    cp_in_progress,
                )
            else:
                while True:
                    tid = -1
                    for i in range(self.max_async):
                        if (self.threads[i] is None) or (
                            not self.threads[i].is_alive()
                        ):
                            p = Thread(
                                target=self.write_pipelined,
                                args=(
                                    (
                                        self.cpu_ar_list[i]
                                        if not self.memory_saving
                                        else None
                                    ),
                                    self.num_threads,
                                    self.total_size,
                                    self.bsize,
                                    lock,
                                    cp_in_progress,
                                ),
                            )
                            p.start()
                            self.threads[i] = p
                            tid = i
                            print(f"Save checkpoint with process {tid}")
                            break
                    if tid >= 0:
                        break

            with lock:
                start1.value = 0

        for x in self.threads:
            # if x.is_alive():
            if x is not None and x.is_alive():
                x.join()

        print("---- exit")
