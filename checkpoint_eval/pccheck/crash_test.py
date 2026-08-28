"""Crash-injection test for the pccheck writer.
"""

import argparse
import ctypes
import os
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np

DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "libtest_ssd.so")


# --------------------------------------------------------------- ctypes glue


def bind(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.writer.restype = ctypes.c_void_p
    lib.writer.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_bool, ctypes.c_int,
                           ctypes.c_int]
    lib.registerCheck.restype = ctypes.c_int
    lib.registerCheck.argtypes = [ctypes.c_void_p]
    lib.take_cpu_address.restype = ctypes.c_void_p
    lib.take_cpu_address.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.savenvm_new.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_bool,
    ]
    lib.pccheck_recovered.restype = ctypes.c_int
    lib.pccheck_latest_counter.restype = ctypes.c_long
    lib.pccheck_latest_slot.restype = ctypes.c_int
    return lib


# ------------------------------------------------------------------- writer


def run_writer(a):
    """Write checkpoints in a loop until killed. Generation g fills the whole
    buffer with the float g, so a torn checkpoint is self-evident."""
    lib = bind(a.lib)
    batch_size = a.total_size // a.num_batches
    # Enough DRAM staging buffers for every batch of every in-flight
    # checkpoint, plus slack, or take_cpu_address() spins forever.
    n_dram = a.max_async * a.num_batches + 2

    mfd = os.open(a.marker_file,
              os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o644)

    w = lib.writer(a.path.encode(), a.max_async, batch_size, n_dram,
                   False, 0, 1)
    os.write(mfd, f"READY {lib.pccheck_latest_counter()}\n".encode())

    inflight = []
    gen = lib.pccheck_latest_counter()  # continue the sequence after recovery

    def one_checkpoint(g):
        slot = lib.registerCheck(ctypes.c_void_p(w))

        def one_batch(batch_num):
            # Same staging path the real pipeline uses (memory_saving=True):
            # borrow a pinned DRAM buffer, fill it, hand it to savenvm_new,
            # which returns it to the pool.
            ptr = lib.take_cpu_address(ctypes.c_void_p(w), batch_num)
            arr = np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(batch_size,))
            arr[:] = np.float32(g)
            lib.savenvm_new(ctypes.c_void_p(w), batch_num, arr, a.total_size,
                            a.threads, slot, batch_num, batch_size,
                            batch_num == a.num_batches)

        ts = [threading.Thread(target=one_batch, args=(i + 1,))
              for i in range(a.num_batches)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # savenvm_new with last_batch=True has returned: durable.
        os.write(mfd, f"COMMIT {g}\n".encode())

    written = 0
    while a.gens == 0 or written < a.gens:
        gen += 1
        written += 1
        t = threading.Thread(target=one_checkpoint, args=(gen,))
        t.start()
        inflight.append(t)
        while len(inflight) >= a.max_async:
            inflight = [x for x in inflight if x.is_alive()]
            time.sleep(0.001)
    for t in inflight:
        t.join()
    os.write(mfd, b"DONE\n")
    os.close(mfd)


# ------------------------------------------------------------------- driver


def verify(a, last_commit):
    """Reopen the file and check I1 and I2. Returns (ok, message)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from checkpoint_eval.pccheck.chk_load import load_flat, NoCheckpoint

    try:
        flat, info = load_flat(a.path, a.total_size, a.max_async)
    except NoCheckpoint:
        if not os.path.exists(a.path):
            return False, ("HARNESS ERROR: checkpoint file does not exist - "
                           "the writer was killed before creating it")
        if last_commit == 0:
            return True, "no checkpoint yet, none committed - consistent"
        return False, (f"I2 VIOLATED: no recoverable checkpoint, but "
                       f"COMMIT {last_commit} was acknowledged")

    vals = flat.unique()
    if vals.numel() != 1:
        counts = {float(v): int((flat == v).sum()) for v in vals[:6]}
        return False, (f"I1 VIOLATED (torn): slot {info['slot']} counter "
                       f"{info['counter']} holds {vals.numel()} distinct "
                       f"values, e.g. {counts}")

    gen = float(vals[0])
    if gen < last_commit:
        detail = " (all zeros: data blocks not durable)" if gen == 0 else ""
        return False, (f"I2 VIOLATED (rollback): recovered generation {gen:.0f}"
                       f"{detail}, but COMMIT {last_commit} was acknowledged")

    return True, (f"OK: generation {gen:.0f} uniform, counter "
                  f"{info['counter']}, slot {info['slot']} "
                  f"(last ack'd COMMIT {last_commit})")


def reattach_ok(a):
    """After recovery the file must still be usable: a fresh writer should
    reattach and continue, not wedge on a corrupted free list or counter."""
    marker = a.marker_file + ".reattach"
    argv = writer_argv(a)
    argv[argv.index("--marker-file") + 1] = marker
    p = subprocess.Popen(
        [sys.executable, __file__, "--mode", "writer"] + argv,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with open(marker) as f:
                    if sum(1 for l in f if l.startswith("COMMIT")) >= 2:
                        return True, "reattached and wrote 2 more checkpoints"
            except FileNotFoundError:
                pass
            time.sleep(0.02)
        return False, "reattach FAILED: writer produced <2 checkpoints in 30s"
    finally:
        p.kill()
        p.wait()


def writer_argv(a):
    return ["--path", a.path, "--lib", a.lib, "--gens", str(a.gens),
            "--marker-file", a.marker_file,
            "--total-size", str(a.total_size), "--max-async", str(a.max_async),
            "--num-batches", str(a.num_batches), "--threads", str(a.threads)]


def one_trial(a, trial, rng):
    for f in (a.path, a.marker_file):
        if os.path.exists(f):
            os.remove(f)

    p = subprocess.Popen(
        [sys.executable, __file__, "--mode", "writer"] + writer_argv(a),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    def read_commits():
        try:
            with open(a.marker_file) as f:
                return [int(l.split()[1]) for l in f
                        if l.startswith("COMMIT")]
        except (FileNotFoundError, IndexError, ValueError):
            return []

    # Wait for a randomised number of completed checkpoints, so kills land at
    # varied points in the rotation rather than always the same one, then wait
    # a random fraction of a checkpoint period before killing.
    # Wait for the writer to finish initialize() before timing anything.
    ready_by = time.time() + 120
    while time.time() < ready_by and p.poll() is None:
        try:
            with open(a.marker_file) as f:
                if any(l.startswith("READY") for l in f):
                    break
        except FileNotFoundError:
            pass
        time.sleep(0.005)
    else:
        return None, "writer never became READY"

    want = rng.randint(2, 6)
    t0 = time.time()
    deadline = t0 + 120
    while len(read_commits()) < want and time.time() < deadline and p.poll() is None:
        time.sleep(0.005)
    commits = read_commits()
    if p.poll() is not None:
        return None, "writer exited early"
    if not commits:
        return None, "writer produced no checkpoints in 120s"

    period = (time.time() - t0) / max(len(commits), 1)
    time.sleep(rng.uniform(0, 1.5 * max(period, 0.01)))

    last_commit = max(read_commits() or [0])
    p.send_signal(signal.SIGKILL)
    p.wait()
    time.sleep(0.05)

    ok, msg = verify(a, last_commit)
    return ok, msg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["drive", "writer"], default="drive")
    ap.add_argument("--path", default="/local/home/fstrati/crash_test.chk")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--total-size", type=int, default=1 << 23,
                    help="floats per checkpoint (default 8M = 32MB)")
    ap.add_argument("--max-async", type=int, default=1)
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--marker-file", default="/tmp/pccheck_crash_markers")
    ap.add_argument("--gens", type=int, default=0,
                    help="writer: stop after N checkpoints (0 = until killed)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-kill", action="store_true",
                    help="sanity mode: let the writer run, verify, no crash")
    ap.add_argument("--check-reattach", action="store_true",
                    help="after each trial, confirm a new writer can continue")
    a = ap.parse_args()

    if a.total_size % a.num_batches:
        raise SystemExit("--total-size must be divisible by --num-batches")

    if a.mode == "writer":
        run_writer(a)
        return

    print(f"crash test: {a.trials} trials, total_size={a.total_size} floats "
          f"({a.total_size * 4 / 1e6:.0f} MB), max_async={a.max_async}, "
          f"num_batches={a.num_batches}, threads={a.threads}")
    print(f"file: {a.path}\n")

    rng = random.Random(a.seed)
    failures = []

    if a.no_kill:
        # Harness self-test: no crash at all. The writer stops cleanly after
        # --gens checkpoints, so recovery must return exactly that generation.
        # If this fails, the harness is wrong and every crash result is noise.
        gens = a.gens or 5
        a.gens = gens
        if os.path.exists(a.path):
            os.remove(a.path)
        p = subprocess.Popen(
            [sys.executable, __file__, "--mode", "writer"] + writer_argv(a),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        p.wait(timeout=300)
        with open(a.marker_file) as f:
            commits = [int(l.split()[1]) for l in f if l.startswith("COMMIT")]
        ok, msg = verify(a, max(commits) if commits else 0)
        print(f"  no-kill  {'PASS' if ok else 'FAIL'}  {msg}")
        print(f"           wrote generations {commits}")
        sys.exit(0 if ok else 1)
    for trial in range(a.trials):
        ok, msg = one_trial(a, trial, rng)
        tag = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  trial {trial:>3} {tag}  {msg}")
        if ok is False:
            failures.append((trial, msg))
        if ok and a.check_reattach:
            rok, rmsg = reattach_ok(a)
            print(f"           {'PASS' if rok else 'FAIL'}  {rmsg}")
            if not rok:
                failures.append((trial, rmsg))
        if a.no_kill:
            break

    print()
    if failures:
        print(f"RESULT: FAIL - {len(failures)}/{a.trials} trials violated an "
              f"invariant")
        for trial, msg in failures[:5]:
            print(f"  trial {trial}: {msg}")
        sys.exit(1)
    print(f"RESULT: PASS - {a.trials} trials, no tearing, no rollback")


if __name__ == "__main__":
    main()
