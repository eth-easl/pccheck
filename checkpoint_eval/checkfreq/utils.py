import torch
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock


def save_checkpoint(
    checkpoint_format,
    path_to_pmem,
    filepath,
    additional_snapshot,
    chk,
    active_snapshot,
    in_progress_snapshot,
    lock,
    epoch,
    it,
    last_chk_it,
    change,
    profile_snap,
    sync=False,
):

    filepath.value = checkpoint_format.format(epoch=epoch, it=0)
    if path_to_pmem != "":
        filepath.value = f"{path_to_pmem}/{filepath.value}"

    additional_snapshot["epoch"] = epoch
    additional_snapshot["iter"] = it
    # print(f"Call {chk.chk_process} at epoch {epoch} and iteration {it}")

    if not chk.spawned:
        print("------------- START A NEW PROCESS!! ------------")
        keywords = {
            "snapshot_ready": False,
            "profile_snap": profile_snap,
            "background": True,
            "iter_chk": last_chk_it,
            "overwrite": True,
        }
        chk.chk_process = Process(
            target=chk._serialize_and_persist,
            args=[
                filepath,
                active_snapshot,
                in_progress_snapshot,
                lock,
                change,
                additional_snapshot,
            ],
            kwargs=keywords,
        )
        chk.chk_process.start()
        chk.spawned = True
        print("-------------- PROCESS STARTED!! ----------")

    if chk.chk_process is not None:
        while change.value == 1:
            # this means a checkpoint is on progress (wait for process doing the checkpoint to set variable to 0)
            continue

    # Once complete, initiate the next checkpoint synchronously
    with lock:
        in_progress_snapshot.value = 1
        change.value = 1


def make_shm(obj):

    if obj is None:
        return

    if torch.is_tensor(obj):
        obj.share_memory_()

    elif isinstance(obj, dict):
        for name, ref in obj.items():
            make_shm(ref)

    elif isinstance(obj, list):
        for x in obj:
            make_shm(x)
    else:
        return
