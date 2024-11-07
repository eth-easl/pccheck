import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time
import numpy as np

# Checkpointing and restoring, inspired from CheckFreq


class CFCheckpoint(object):

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.tracking_map = OrderedDict()
        self.spawned = False
        self.chk_process = None

        for name, ref in kwargs.items():
            # if hasattr(ref, 'state_dict'):
            self.tracking_map[name] = ref
            # else:
            #       self.logger.info("Skipping object `{}` in CF Checkpointing. \
            #       No state_dict() method exposed".format(name))

        self.num_tracking = len(self.tracking_map.keys())

        if self.num_tracking == 0:
            raise ValueError("Nothing to track")

    def __getstate__(self):
        d = self.__dict__.copy()
        if "logger" in d:
            d["logger"] = d["logger"].name
        return d

    def __setstate__(self, d):
        if "logger" in d:
            d["logger"] = logging.getLogger(d["logger"])
        self.__dict__.update(d)

    def _snapshot(self, active_snapshot, additional_state=None):
        if active_snapshot == 1:
            # if self.latest_snapshot is not None:
            self.logger.info("Snapshot is not None. Checkpoint underway")
            return False

        self.latest_snapshot = OrderedDict()
        start = time.time()

        # print('MODEL KEYS: ', self.tracking_map['model'].keys())
        # print('OPTIMIZER KEYS: ', self.tracking_map['optimizer'].keys())

        # Snapshot the state of tractable items
        for name, ref in self.tracking_map.items():
            if name not in self.latest_snapshot:
                if hasattr(ref, "state_dict"):
                    self.latest_snapshot[name] = _to_cpu(ref.state_dict())
                else:
                    self.latest_snapshot[name] = {}
                    for n, r in ref.items():
                        self.latest_snapshot[name][n] = _to_cpu(r)
                        # print(n, r._grad.shape)
                    # self.latest_snapshot[name] = copy.deepcopy(ref)
            else:
                self.logger.info("Repeated entry for {}".format(name))
                return False

        if additional_state:
            self.latest_snapshot.update(additional_state)

        return True

    def _serialize_and_persist(
        self,
        filepath,
        active_snapshot,
        in_progress_snapshot,
        lock,
        change,
        additional_state=None,
        background=False,
        snapshot_ready=False,
        profile_snap=None,
        iter_chk=None,
        epoch_chk=None,
        overwrite=True,
    ):

        time_meas = []

        while True:

            if background:
                with lock:
                    if change.value == 0:
                        # print("---------- I am stuck!")
                        continue

            start_total = time.time()

            # print("------------------------------------------ kwargs: ", background, iter_chk, overwrite)
            if not snapshot_ready:
                # self.logger.info("[{}] START SNAPSHOT".format(time.time()))
                start = time.time()
                success = self._snapshot(
                    active_snapshot.value, additional_state=additional_state
                )
                end1 = time.time()
                print("SNAPSHOT TOOK: ", end1 - start)

                if success:
                    with lock:
                        in_progress_snapshot.value = 0
                        active_snapshot.value = 1
                else:
                    change.value = 0
                    self.logger.error("Cannot persist. Empty snapshot")
                    return
            else:
                with lock:
                    if active_snapshot.value == 0:
                        change.value = 0
                        self.logger.error("Cannot persist. Empty snapshot")
                        return

            snapshot = self.latest_snapshot
            if background and profile_snap.value == 1:
                snapshot = {}
                with lock:
                    active_snapshot.value = 0
                    change.value = 0
                    continue  # not sure about that

            start = time.time()
            torch.save(snapshot, filepath.value)

            # Clear the snapshot.
            with lock:
                active_snapshot.value = 0

            # Ensure its persisted
            f = open(filepath.value, "a+")
            os.fsync(f.fileno())
            f.close()

            update_stats(filepath.value, overwrite=False, iter_chk=iter_chk)
            # print("[{}] END ASYNC".format(time.time()))

            if not background:
                print(
                    " *** ------------------------------------ TEMPORARY, exit now -----------------------------------"
                )
                return

            with lock:
                snapshot = {}
                change.value = 0

            print(f"CHECKPOINT TOOK {time.time()-start}")

            time_meas.append(time.time() - start_total)
            print(f"TOTAL TIME IS: {time.time()-start_total} **************")

            print(f"Current checkpoint time median is: {np.median(time_meas)}")


def update_stats(filepath, overwrite=True, iter_chk=None):

    dirpath = os.path.dirname(filepath)
    fname = os.path.splitext(os.path.basename(filepath))[0]
    # print(dirpath, fname)
    tokens = fname.split("-")
    idx = tokens[1]
    cur_iter = int(tokens[2])
    base_name = tokens[0]
    # print(tokens)
    if overwrite:
        del_filepath = (
            dirpath + "/" + tokens[0] + "-" + idx + "-" + str(iter_chk.value) + ".chk"
        )
        print("to delete: ", del_filepath)
        if os.path.exists(del_filepath):

            os.remove(del_filepath)
    iter_chk.value = cur_iter


def _to_cpu(ele, snapshot=None):
    if snapshot is None:
        snapshot = {}
    if hasattr(ele, "cpu"):  # tensor
        snapshot = ele.cpu()
    elif isinstance(ele, dict):
        snapshot = {}
        for k, v in ele.items():
            snapshot[k] = None
            snapshot[k] = _to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_cpu(v, snapshot[idx])
    else:
        return ele
    return snapshot
