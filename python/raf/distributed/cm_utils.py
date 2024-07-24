# pylint: disable=protected-access, invalid-name
"""Util functions related to distributed trainings"""
from collections import defaultdict
import os
import signal
from typing import Tuple
import ctypes
import numpy as np
from sklearn.linear_model import LinearRegression

from tvm.runtime.object import Object
from raf._core.core_utils import register_node
import raf._ffi.distributed as _ffi
from raf._lib import tvm

INT64_MAX=9223372036854775806

@register_node("raf.distributed.CommCostModelParams")
class CommCostModelParams(Object):
    __fields__ = [("overhead", ctypes.c_uint64), ("throughput", ctypes.c_uint64)]
    def __init__(self, overhead, throughput):
        self.__init_handle_by_constructor__(_ffi.CommCostModelParams, overhead, throughput)

@register_node("raf.distributed.DynamicScheduleParams")
class DynamicScheduleParams(Object):
    __fields__ = [("lambda_comp", ctypes.c_double), ("lambda_comm", ctypes.c_double), ("gamma", ctypes.c_double), ("theta_comp", ctypes.c_double), ("theta_comm", ctypes.c_double), ("beta", ctypes.c_double)]
    def __init__(self, lamb_comp, lamb_comm, gamma, theta_comp, theta_comm, beta):
        self.__init_handle_by_constructor__(_ffi.DynamicScheduleParams, lamb_comp, lamb_comm, gamma, theta_comp, theta_comm, beta)

@tvm._ffi.register_func("raf.distributed.fit_allreduce_cost_model")
def fit_allreduce_cost_model(comm_times_ptr, comm_sizes_ptr, size) -> Tuple[float, float]:
    if "DEBUG_CM_FIXED_PARAMS" in os.environ:
        params = os.environ.get("DEBUG_CM_FIXED_PARAMS")
        latency, throughput = [float(x) for x in params.split(";")]
        return CommCostModelParams(int(latency), int(throughput))

    comm_times = ctypes.cast(comm_times_ptr, ctypes.POINTER(ctypes.c_double))
    comm_sizes = ctypes.cast(comm_sizes_ptr, ctypes.POINTER(ctypes.c_double))
    py_comm_times = []
    py_comm_sizes = []
    deduplicated_sizes = set()
    for i in range(size):
        py_comm_times.append(comm_times[i])
        py_comm_sizes.append(comm_sizes[i])
        deduplicated_sizes.add(comm_sizes[i])
    if len(deduplicated_sizes) == 1:
        mean_time = np.mean(py_comm_times)
        size = py_comm_sizes[0]
        if size < 64:
            print("[CostModel] WARNING: All profiled comm op have the same size. Assuming zero throughput.")
            throughput = 1
            overhead = int(mean_time)
        else:
            print("[CostModel] WARNING: All profiled comm op have the same size. Assuming zero overhead.")
            overhead = 0
            throughput = int(float(size) / float(mean_time))
            if throughput == 0:
                throughput = 1
        return CommCostModelParams(overhead, throughput)
    inv_sq_comm_times = [1.0/(x*x) for x in py_comm_times]
    size_time = defaultdict(list)
    for i in range(len(py_comm_sizes)):
        size_time[py_comm_sizes[i]].append(py_comm_times[i])
    # reg = LinearRegression().fit(np.array(py_comm_sizes).reshape(-1, 1), py_comm_times, sample_weight=inv_sq_comm_times)
    reg = LinearRegression().fit(np.array(py_comm_sizes).reshape(-1, 1), py_comm_times)
    if reg.intercept_ < 0:
        reg = LinearRegression(fit_intercept=False, positive=True).fit(np.array(py_comm_sizes).reshape(-1, 1), py_comm_times)
        reg.intercept_ = 0
    throughput =  (1 / reg.coef_)[0]
    if throughput == float("inf"):
        throughput = int(INT64_MAX)
    else:
        throughput = int(throughput)
    return CommCostModelParams(int(reg.intercept_), throughput)
