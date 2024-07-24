# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods
"""Distributed Context"""
import raf._ffi.distributed as ffi
from raf._core.core_utils import register_node
from raf._ffi.distributed import _make
from raf._lib import Object


@register_node("raf.distributed.DistContext")
class DistContext(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_make.DistContext)

    @property
    def enable_data_parallel(self):
        return self.enable_data_parallel_

    @enable_data_parallel.setter
    def enable_data_parallel(self, value):
        self.enable_data_parallel_ = value
        ffi.EnableDataParallel(value)

    @property
    def overlap_comm_forward(self):
        return self.overlap_comm_forward_

    @overlap_comm_forward.setter
    def overlap_comm_forward(self, value):
        self.overlap_comm_forward_ = value
        ffi.OverlapCommForward(value)

    @property
    def size(self):
        return self.size_

    @size.setter
    def size(self, value):
        self.size_ = value
        ffi.SetGlobalSize(value)

    @property
    def rank(self):
        return self.rank_

    @rank.setter
    def rank(self, value):
        self.rank_ = value
        ffi.SetGlobalRank(value)

    def set_local_rank_for_tuning(self, value):
        ffi.SetLocalRankForTuning(value)

    @property
    def zero_opt_level(self):
        return self.zero_opt_level_

    @zero_opt_level.setter
    def zero_opt_level(self, value):
        self.zero_opt_level_ = value
        ffi.ZeroOpt(value)

    @property
    def force_sync_after_comm(self):
        return self.force_sync_after_comm_

    @force_sync_after_comm.setter
    def force_sync_after_comm(self, value):
        self.force_sync_after_comm_ = value
        ffi.ForceSyncAfterComm(value)

    @property
    def auto_dp_profiling_start_iter(self):
        return self.auto_dp_profiling_start_iter_

    @auto_dp_profiling_start_iter.setter
    def auto_dp_profiling_start_iter(self, value):
        self.auto_dp_profiling_start_iter_ = value
        ffi.AutoDPProfilingStartIter(value)

    @property
    def auto_dp_profiling_end_iter(self):
        return self.auto_dp_profiling_end_iter_

    @auto_dp_profiling_end_iter.setter
    def auto_dp_profiling_end_iter(self, value):
        self.auto_dp_profiling_end_iter_ = value
        ffi.AutoDPProfilingEndIter(value)

    def barrier(self):
        ffi.Barrier()


def get_context():
    return ffi.Global()
