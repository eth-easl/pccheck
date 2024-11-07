import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time
import numpy as np
import ctypes

# Checkpointing and restoring, inspired from CheckFreq


class GPMCheckpoint(object):

    def __init__(
            self,
            filepath,
            **kwargs):
        self.logger = logging.getLogger(__name__)
        self.tracking_map = OrderedDict()

        for name, ref in kwargs.items():
            self.tracking_map[name] = ref

        self.num_tracking = len(self.tracking_map.keys())
        self.filepath = filepath
        if self.num_tracking == 0:
            raise ValueError("Nothing to track")

        home_dir = os.path.expanduser("~")
        self.lib = ctypes.cdll.LoadLibrary(
            f'{home_dir}/pccheck/checkpoint_eval/gpm/libtest.so')

        total_size = ctypes.c_ulong(self.tracking_map['datasize'])

        tensor_ptrs = []
        self._get_ptr(self.tracking_map, tensor_ptrs)
        VoidPArray = ctypes.c_void_p * len(tensor_ptrs)
        tensor_ptrs_ar = VoidPArray(*tensor_ptrs)

        tensor_sizes = []
        self._get_sizes(self.tracking_map, tensor_sizes)
        SizetArray = ctypes.c_size_t * len(tensor_sizes)
        tensor_sizes_ar = SizetArray(*tensor_sizes)

        print(f"totalsize is {total_size}, {len(tensor_sizes)} tensors")

        # TODO: probably some conversion here
        self.lib.init(
            self.filepath.encode(),
            total_size,
            tensor_ptrs_ar,
            tensor_sizes_ar,
            len(tensor_sizes)
        )

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def _get_ptr(self, input, l):
        if hasattr(input, 'data_ptr'):
            l.append(ctypes.c_void_p(input.data_ptr()))
        elif hasattr(input, 'state_dict'):
            self._get_ptr(input.state_dict(), l)
        elif isinstance(input, dict):
            for _, item in input.items():
                self._get_ptr(item, l)

    def _get_sizes(self, input, l):
        if hasattr(input, 'numel'):
            if input.dtype == torch.float32 or input.dtype == torch.int32:
                l.append(input.numel()*4)
            elif input.dtype == torch.float16:
                l.append(input.numel()*2)
            elif input.dtype == torch.int64:
                l.append(input.numel()*8)
        elif hasattr(input, 'state_dict'):
            self._get_sizes(input.state_dict(), l)
        elif isinstance(input, dict):
            for _, item in input.items():
                self._get_sizes(item, l)

    def _checkpoint(
            self,
            iter_chk=None,
            epoch_chk=None):

        time_meas = []
        start = time.time()
        self.lib.save_gpm()
        time_meas.append(time.time()-start)
        print(f"CHECKPOINT TOOK {time.time()-start}")

        print(f"Current checkpoint time median is: {np.median(time_meas)}")

    def _finish(self):
        self.lib.finish()
