"""Read the latest pccheck checkpoint back into a model/optimizer.

The on-disk layout is fully determined by (max_async, total_size), so this
reads the file directly with np.memmap rather than going through the C++
writer. Two reasons that matters:

  * writer()/initialize() has side effects -- it allocates the DRAM staging
    area, re-points the checkpoint head, and (before the O_TRUNC fix) would
    destroy the file outright. Loading must not require any of that.
  * a pure-Python reader is an independent implementation of the layout, so
    it cross-checks the C++ loader in step 3 instead of sharing its bugs.

Layout (byte offsets from the start of the file), mirroring main_ssd_memory.cpp:

    0                     checkpoint*        (a raw VA -- unusable across runs)
    1 * SLOT_BYTES        PEER_CHECK_ADDR
    2 * SLOT_BYTES        init checkpoint metadata          (slot -1)
    (i+3) * SLOT_BYTES    checkpoint metadata for slot i    (i = 0..max_async)
    (max_async+3) * SLOT_BYTES = PR_ADDR_DATA, start of the data region

and within the data region, slot i's payload starts at

    i*total_size + (i*pad_bytes)//4 + DATA_SKIP_FLOATS      (in floats)

where pad_bytes reproduces the C++ 4KB alignment expression exactly, and
DATA_SKIP_FLOATS is the +4096 that savenvmNew applies to every write.
"""

import os
import struct

import numpy as np
import torch

# PR_ADDR is an int*, so OFFSET_SIZE (4096) is in ints -> 16384 bytes per slot.
SLOT_BYTES = 4096 * 4
# Metadata slots consumed before the data region: head pointer, peer pointer,
# init metadata, then one per usable slot (0..max_async). Must match
# PR_ADDR_DATA in main_ssd_memory.cpp.
META_SLOTS_BEFORE_DATA = 4
# The payload starts exactly at the slot's computed address. The old +4096
# float skip existed only to keep writes off the last metadata slot, which
# used to overlap the data region; both were removed together.
DATA_SKIP_FLOATS = 0
# struct checkpoint { long area; long counter; } __attribute__((aligned(64)))
_META = struct.Struct("<qq")


class NoCheckpoint(Exception):
    """The file exists but holds no completed checkpoint."""


def _meta_offset(slot):
    """Byte offset of slot's metadata. Slot -1 is the init metadata."""
    return (slot + 3) * SLOT_BYTES


def _data_offset(slot, total_size, max_async):
    """Byte offset of slot's payload."""
    pad_bytes = 4096 - (total_size * 4) % 4096  # same expression as the C++
    start_floats = slot * total_size + (slot * pad_bytes) // 4
    return ((max_async + META_SLOTS_BEFORE_DATA) * SLOT_BYTES
            + (start_floats + DATA_SKIP_FLOATS) * 4)


def find_latest(path, max_async):
    """Return (slot, counter) of the newest checkpoint in the file.

    Same rule initialize() uses on the recovery path: the highest counter
    wins, and counter == 0 everywhere means nothing was ever written
    (registerCheck starts at 1; unwritten sparse regions read as zero).
    """
    if not os.path.exists(path):
        raise NoCheckpoint(f"{path} does not exist")

    best_slot, best_counter = None, 0
    with open(path, "rb") as f:
        for slot in range(-1, max_async + 1):
            f.seek(_meta_offset(slot))
            raw = f.read(_META.size)
            if len(raw) < _META.size:
                break
            area, counter = _META.unpack(raw)
            if counter > best_counter:
                best_slot, best_counter = slot, counter

    if best_slot is None:
        raise NoCheckpoint(f"{path} holds no completed checkpoint")
    return best_slot, best_counter


def load_flat(path, total_size, max_async, slot=None):
    """Read a checkpoint payload as a 1-D float32 CPU tensor.

    Returns (flat, info). Pass slot to read a specific one; by default the
    newest is used.
    """
    counter = None
    if slot is None:
        slot, counter = find_latest(path, max_async)

    offset = _data_offset(slot, total_size, max_async)
    need = offset + total_size * 4
    actual = os.path.getsize(path)
    if actual < need:
        raise NoCheckpoint(
            f"{path} is {actual} bytes, need {need} to read slot {slot} "
            f"of {total_size} floats -- wrong total_size or max_async?"
        )

    mm = np.memmap(path, dtype=np.float32, mode="r",
                   offset=offset, shape=(total_size,))
    # np.array(copy=True): detach from the mapping entirely. A view would keep
    # the file mapped and hand back a read-only tensor that the caller cannot
    # copy_ into, and would race with any concurrent writer.
    flat = torch.from_numpy(np.array(mm, dtype=np.float32, copy=True))
    del mm

    return flat, {"slot": slot, "counter": counter,
                  "total_size": total_size, "byte_offset": offset}


def _bind(lib_path):
    import ctypes

    lib = ctypes.CDLL(lib_path)
    lib.pccheck_find_latest.restype = ctypes.c_int
    lib.pccheck_find_latest.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_long)]
    lib.pccheck_load_latest.restype = ctypes.c_long
    lib.pccheck_load_latest.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_int),
    ]
    return lib


def load_flat_c(lib_path, path, total_size, max_async):
    """Same as load_flat(), but via the C loader in main_ssd_memory.cpp.

    Kept alongside the Python reader rather than replacing it: two independent
    implementations of the layout catch offset mistakes that a single one
    would simply agree with. See cross_check().
    """
    import ctypes

    lib = _bind(lib_path)
    buf = np.empty(total_size, dtype=np.float32)
    slot = ctypes.c_int(0)
    counter = lib.pccheck_load_latest(path.encode(), max_async,
                                      total_size, buf, ctypes.byref(slot))
    if counter == 0:
        raise NoCheckpoint(f"{path} holds no completed checkpoint")
    if counter < 0:
        raise IOError(f"C loader failed on {path} (see stderr)")

    return torch.from_numpy(buf), {
        "slot": slot.value, "counter": counter, "total_size": total_size,
        "byte_offset": _data_offset(slot.value, total_size, max_async),
    }


def cross_check(lib_path, path, total_size, max_async):
    """Assert the Python and C loaders agree. Returns their shared result."""
    py_flat, py_info = load_flat(path, total_size, max_async)
    c_flat, c_info = load_flat_c(lib_path, path, total_size, max_async)

    if py_info["slot"] != c_info["slot"] or py_info["counter"] != c_info["counter"]:
        raise AssertionError(f"loaders disagree on which checkpoint: "
                             f"python={py_info} c={c_info}")
    if not torch.equal(py_flat, c_flat):
        bad = (py_flat != c_flat).nonzero().flatten()
        raise AssertionError(
            f"loaders disagree on {bad.numel()} of {total_size} floats, "
            f"first at index {int(bad[0])}"
        )
    return c_flat, c_info


def restore_from_flat(model, optimizer_list, flat, strict=True):
    """Scatter a flat checkpoint back into model, optimizer and buffers.

    Pairs with set_storage_updated(): it walks iter_checkpoint_state(), the
    same order that laid the buffer out, so parameters, optimizer state and BN
    buffers all land where they were written. Sharing one iterator between the
    writer and the reader is what keeps the two from drifting apart.

    A checkpoint written under the ORIGINAL set_storage() (parameters then
    gradients) cannot be read by this function -- the regions do not line up.

    Copies in place (copy_), so tensors aliased into a flat GPU buffer stay
    aliased. Note that when set_storage() has been applied, every destination
    *is* a view of gpu_ar, and this is equivalent to gpu_ar.copy_(flat); it is
    written as a scatter so it also works on an unaliased model.
    """
    from checkpoint_eval.pccheck_utils import iter_checkpoint_state

    flat = flat.view(-1)
    idx = 0
    restored = {"param": 0, "opt": 0, "buffer": 0}

    with torch.no_grad():
        for name, ref in iter_checkpoint_state(model, optimizer_list):
            n = ref.numel()
            if idx + n > flat.numel():
                raise ValueError(
                    f"checkpoint too small: need {idx + n} floats by "
                    f"'{name}', have {flat.numel()}. Was it written with a "
                    f"different model or a different layout?"
                )
            src = flat[idx:idx + n].view(ref.size()).to(ref.device)
            ref.copy_(src)  # copy_ casts if the buffer dtype is not float32
            kind = ("param" if name.startswith("param.")
                    else "buffer" if name.startswith("buffer.") else "opt")
            restored[kind] += 1
            idx += n

    report = {
        "restored_params": restored["param"],
        "restored_optimizer": restored["opt"],
        "restored_buffers": restored["buffer"],
        "restored_floats": idx,
        "buffer_floats": flat.numel(),
    }

    if strict and idx != flat.numel():
        report["warning"] = (
            f"consumed {idx} of {flat.numel()} floats; the checkpoint does "
            f"not match this model's layout."
        )

    return report


def load_latest(path, model, optimizer_list, total_size, max_async=1):
    """Convenience wrapper: find newest checkpoint, read it, scatter it."""
    flat, info = load_flat(path, total_size, max_async)
    report = restore_from_flat(model, optimizer_list, flat)
    report.update(info)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a pccheck checkpoint file")
    ap.add_argument("path")
    ap.add_argument("--max-async", type=int, default=1)
    ap.add_argument("--total-size", type=int, default=None,
                    help="if given, also read and summarise the payload")
    ap.add_argument("--c-lib", default=None,
                    help="path to libtest_ssd.so; cross-checks the C loader "
                         "against the Python one (needs --total-size)")
    a = ap.parse_args()

    print(f"file: {a.path} ({os.path.getsize(a.path)} bytes)")
    with open(a.path, "rb") as f:
        for slot in range(-1, a.max_async + 1):
            f.seek(_meta_offset(slot))
            area, counter = _META.unpack(f.read(_META.size))
            tag = "init" if slot == -1 else f"slot {slot}"
            print(f"  {tag:8s} @ {_meta_offset(slot):>10d}  "
                  f"area={area:<4d} counter={counter}")

    slot, counter = find_latest(a.path, a.max_async)
    print(f"latest: slot {slot}, counter {counter}")

    if a.total_size:
        if a.c_lib:
            flat, info = cross_check(a.c_lib, a.path, a.total_size, a.max_async)
            print("cross-check: python and C loaders agree exactly")
        else:
            flat, info = load_flat(a.path, a.total_size, a.max_async)
        finite = torch.isfinite(flat)
        print(f"payload @ byte {info['byte_offset']}: {flat.numel()} floats, "
              f"finite={int(finite.sum())}/{flat.numel()}, "
              f"nonzero={int((flat != 0).sum())}, "
              f"min={flat[finite].min():.6g}, max={flat[finite].max():.6g}, "
              f"mean={flat[finite].mean():.6g}")
