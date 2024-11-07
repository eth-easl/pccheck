import torch
import time
import sys

def _to_gpu(ele, snapshot=None):
    if snapshot is None:
        snapshot = {}

    if isinstance(ele, torch.Tensor):  # tensor
        snapshot = ele.cuda()
    elif isinstance(ele, dict):
        snapshot = {}
        for k, v in ele.items():
            snapshot[k] = None
            snapshot[k] = _to_gpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_gpu(v, snapshot[idx])
    else:
        return ele
    return snapshot


def get_load_time(input_file):
    # get time to bring the checkpoint back to GPU mem

    # warmup - gpu

    for i in range(10):
        t1 = torch.ones(int(1e8))
        gpu_t1 = t1.cuda()
        torch.cuda.synchronize()

    # only once - 'cold start' is what we care about
    for i in range(1):
        start_time = time.time()
        data = torch.load(input_file)
        gpu_data = _to_gpu(data)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Time is {end_time-start_time} sec")

if __name__ == "__main__":
    get_load_time(sys.argv[1])