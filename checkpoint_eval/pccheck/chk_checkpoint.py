import torch
import time
from ctypes import *
import numpy as np
from threading import Thread, Lock


class Writer(object):

    # constructor
    def __init__(self, fname, lib_path):
        # attribute
        self.lib = cdll.LoadLibrary(lib_path)
        self.writer_obj = self.lib.writer(fname)

    def write_array(self, x, sz, num_threads):
        ct_arr = np.ctypeslib.as_ctypes(x)
        self.lib.savenvm(self.writer_obj, ct_arr, c_ulong(sz), num_threads)

    def finish(self):
        self.lib.finish()


class Checkpoint:

    def __init__(self, total_size, num_threads, filename, lib_path, max_async):

        self.total_size = total_size
        self.filename = filename
        self.num_threads = num_threads
        self.max_async = max_async
        self.lib_path = lib_path

    def write_array(self, x, sz, num_threads):
        print("Write checkpoint of size ", sz)
        start = time.time()
        self.writer.write_array(x, sz, num_threads)
        chk_time = time.time() - start

        self.timing_lock.acquire()
        self.timing_list.append(chk_time)
        self.timing_lock.release()
        print("Writing checkpoint to NVM took: ", chk_time)

    def start_chk(
        self,
        lock,
        model_state,
        opt_state,
        total_state,
        cp_in_progress1,
        start1,
        gpu_copy=True,
    ):

        print("Inside start_chk")
        self.writer = Writer(self.filename.encode(), self.lib_path)

        self.timing_lock = Lock()
        self.timing_list = []

        self.threads = [None] * self.max_async
        self.cpu_ar_list = []

        if gpu_copy:
            gpu_ar = torch.ones(self.total_size, dtype=torch.float32)
            gpu_ar = gpu_ar.cuda()
        else:
            print(f"------------ Avoid multiple copies ")
            gpu_ar = total_state

        for i in range(self.max_async):
            self.cpu_ar_list.append(
                torch.empty(self.total_size, dtype=torch.float32, pin_memory=True)
            )

        print("about to enter while loop")
        while True:

            with lock:
                if start1.value == 0:
                    continue

            # print("--------------- PROCESS: ", proc_id, " FROM CHECKPOINT: ", model_state['fc.bias'])
            # snapshot here
            stime = time.time()

            print("--------------------------------------- Time for new checkpoint!")
            if gpu_copy:
                start = time.time()
                idx = 0
                for name, ref in model_state.items():
                    # print(name, ref.dtype)
                    if torch.is_tensor(ref):
                        sz = ref.numel()
                        # print("model ref device: ", ref.device)
                        gpu_ar[idx : idx + sz] = torch.flatten(ref)
                        idx += sz
                    else:
                        gpu_ar[idx] = float(ref)
                        idx += 1

                for name, ref in opt_state["state"].items():
                    for k, r in ref.items():
                        if torch.is_tensor(r):
                            # print(r.device)
                            # print("optimizer r device: ", r.device)
                            sz = r.numel()
                            gpu_ar[idx : idx + sz] = torch.flatten(r)
                            idx += sz
                        else:
                            gpu_ar[idx] = float(r)
                            idx += 1

                print("Filling up large GPU array took: ", time.time() - start)

                with lock:
                    cp_in_progress1.value = 0

            # initiate checkpoint
            while True:
                tid = -1
                for i in range(self.max_async):
                    if (self.threads[i] is None) or (not self.threads[i].is_alive()):
                        start = time.time()
                        self.cpu_ar_list[i].copy_(gpu_ar)
                        print("------------- GPU to CPU took ", time.time() - start)

                        if not gpu_copy:
                            # in this case, we should wait for the copy to be made to the CPU before changing anything
                            with lock:
                                cp_in_progress1.value = 0

                        # 3. transform to numpy
                        start = time.time()
                        cpu_np = self.cpu_ar_list[i].numpy()
                        print("--------------- Np took: ", time.time() - start)

                        print(f"------------- SNAPSHOT took: {time.time()-stime} sec")
                        print(self.total_size)
                        p = Thread(
                            target=self.write_array,
                            args=[cpu_np, self.total_size, self.num_threads],
                        )
                        p.start()
                        self.threads[i] = p
                        tid = i
                        print(f"tid is: {tid}")
                        break
                if tid >= 0:
                    break

            # exit stall when a new thread is created
            with lock:
                start1.value = 0

            # show current median checkpoint time
            self.timing_lock.acquire()
            if len(self.timing_list) > 0:
                print(
                    f"Median checkpoint time so far: {np.median(np.asarray(self.timing_list))}"
                )
            self.timing_lock.release()
