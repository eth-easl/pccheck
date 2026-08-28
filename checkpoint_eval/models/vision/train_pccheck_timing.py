""" Copy of train_pccheck.py that records WHEN checkpoints become durable.

Same training loop and same checkpointing path -- the only additions are
timestamps, so throughput numbers stay comparable.

Two timestamps per checkpoint, taken in two different processes:

  * t_request  -- this process, the instant the trainer called save()
  * t_persist  -- the checkpoint worker process, the instant the checkpoint
                  became recoverable on the SSD

The second cannot be observed from here: checkpointing runs in a spawned
worker. It comes from write_pipelined() in chk_checkpoint_pipeline.py, which
already timed each checkpoint; setting PCCHECK_PERSIST_LOG makes it write
those timings out with absolute timestamps. The two logs are joined on idx
(registration order) at the end.

Usage:

    python3 train_pccheck_timing.py --arch resnet18 --cfreq 10 \
        --bench_total_steps 200 --c_lib_path .../libtest_ssd.so \
        --timing_out /tmp/chk_timeline.csv

Output columns:

    idx, step, t_request, t_persist, queue_ms, persist_ms, total_ms, since_prev_s

`since_prev_s` is the answer to "how often": wall-clock seconds between one
durable checkpoint and the next.
"""

import os
from platform import node
import sys
import csv
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
import argparse

import ctypes
from checkpoint_eval.pccheck.chk_monitor import Chk_monitor
from checkpoint_eval.pccheck_utils import initialize, get_total_size, set_storage

parser = argparse.ArgumentParser(
    description="PyTorch ImageNet/CIFAR Training with checkpoint persist timing"
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
parser.add_argument(
    "--timing_out",
    default="/tmp/chk_timeline.csv",
    type=str,
    help="where to write the merged per-checkpoint timeline",
)

args = parser.parse_args()

# Raw worker-side log. Removed up front so a previous run's rows do not get
# appended to; write_pipelined() opens it in append mode.
PERSIST_LOG = args.timing_out + ".persist"


def merge_and_report(requests, persist_log, out_path):
    """Join trainer-side requests with worker-side persists, on idx.

    Kept as a join rather than a single log because the two halves live in
    different processes, and because with --max-async > 1 checkpoints may
    complete out of order while always being registered in order.
    """
    persists = {}
    if os.path.exists(persist_log):
        with open(persist_log) as f:
            for r in csv.DictReader(f):
                persists[int(r["idx"])] = r

    rows, prev = [], None
    for idx, step, t_request in requests:
        p = persists.get(idx)
        if p is None:
            # Requested but never completed -- e.g. the run ended first, or the
            # checkpoint was superseded before it published.
            rows.append({"idx": idx, "step": step,
                         "t_request": f"{t_request:.6f}", "t_persist": "",
                         "queue_ms": "", "persist_ms": "", "total_ms": "",
                         "since_prev_s": "", "note": "not persisted"})
            continue

        t_start, t_persist = float(p["t_start"]), float(p["t_persist"])
        rows.append({
            "idx": idx,
            "step": step,
            "t_request": f"{t_request:.6f}",
            "t_persist": f"{t_persist:.6f}",
            "queue_ms": f"{(t_start - t_request) * 1000:.3f}",
            "persist_ms": p["dur_ms"],
            "total_ms": f"{(t_persist - t_request) * 1000:.3f}",
            "since_prev_s": f"{t_persist - prev:.3f}" if prev else "",
            "note": "",
        })
        prev = t_persist

    fields = ["idx", "step", "t_request", "t_persist", "queue_ms",
              "persist_ms", "total_ms", "since_prev_s", "note"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if r["t_persist"]]
    print(f"\n=== CHECKPOINT PERSIST TIMELINE -> {out_path} ===")
    print(f"{'idx':>4} {'step':>6} {'persist_ms':>11} {'total_ms':>10} {'since_prev_s':>13}")
    for r in rows:
        print(f"{r['idx']:>4} {r['step']:>6} {r['persist_ms'] or '-':>11} "
              f"{r['total_ms'] or '-':>10} {r['since_prev_s'] or '-':>13}"
              f"  {r['note']}")

    if len(done) >= 2:
        gaps = [float(r["since_prev_s"]) for r in done if r["since_prev_s"]]
        durs = [float(r["persist_ms"]) for r in done]
        print(f"\n{len(done)} checkpoints persisted, "
              f"{len(rows) - len(done)} requested but not persisted")
        print(f"interval between persists: min {min(gaps):.3f}s  "
              f"median {np.median(gaps):.3f}s  max {max(gaps):.3f}s  "
              f"=> {1.0 / np.median(gaps):.3f} checkpoints/sec")
        print(f"persist duration:          min {min(durs):.1f}ms  "
              f"median {np.median(durs):.1f}ms  max {max(durs):.1f}ms")
    elif done:
        print(f"\nonly {len(done)} checkpoint persisted; "
              f"need at least 2 to report an interval")
    else:
        print("\nno checkpoints persisted")


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
    requests = []  # (idx, step, t_request), joined with the worker's log later

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

        if (batch_idx == warmup) or (
            (args.cfreq > 0) and steps_since_checkp == args.cfreq - 1
        ):
            if args.cfreq > 0:
                # save() returns once the worker has been signalled; the
                # checkpoint is NOT durable yet. Its persist time comes from
                # the worker's log.
                requests.append((checkpoints, batch_idx, time.time()))
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
        # join()s the worker, so every in-flight checkpoint has finished (and
        # logged) by the time this returns.
        chk_monitor.kill_checkpoint()

    print(
        f"-- BENCHMARK ENDED: Total time: {total_train_time} sec, Number of iterations: {batch_idx}, Number of checkpoints: {checkpoints}"
    )
    print(f"EXECUTION TIME: {total_train_time} sec")
    print(f"THROUGHPUT IS {(args.bench_total_steps-warmup)/total_train_time}")

    if args.cfreq > 0:
        merge_and_report(requests, PERSIST_LOG, args.timing_out)


if __name__ == "__main__":
    args = parser.parse_args()
    PERSIST_LOG = args.timing_out + ".persist"
    if os.path.exists(PERSIST_LOG):
        os.remove(PERSIST_LOG)
    # Inherited by the spawned checkpoint worker, which is what actually
    # writes this file.
    os.environ["PCCHECK_PERSIST_LOG"] = PERSIST_LOG

    os.sched_setaffinity(0, {0})
    train()
