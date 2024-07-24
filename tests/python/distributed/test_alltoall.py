# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
"""Test all-to-all communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test all-to-all ops, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_alltoall.py`
(in ci/tash_python_unittest.sh)
"""
import sys
import pytest
import numpy as np

import raf
from raf import distributed as dist
from raf.testing import check, get_dist_info, skip_dist_test, run_vm_model

dctx = dist.get_context()
SKIP_REASON = "Distribution is not enabled or #rank is not expected"

def test_alltoall():
    print("Testing all-to-all with a single tensor as input.")

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            x = raf._all_to_all([x])
            return x

    if raf.build.with_nccl() < 20700:
        print("alltoall is not supported in NCCL < 2.10")
        return

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(total_rank, 4), dtype="float32") * (rank + 1)
    x = raf.array(x, device=device)
    if rank == 0:
        print(f"{rank} - X: ", x)
    model.to(device=device)
    y = model(x)
    vx = np.ones(shape=(total_rank, 4), dtype="float32") * (rank + 1)
    vx = raf.array(vx, device=device)
    vy = run_vm_model(model, device, [vx])
    check(y, vy)
    target_y_components = []
    for i in range(total_rank):
        target_y_components.append(np.ones(shape=(1, 4), dtype="float32") * (i + 1))
    target_y = np.concatenate(target_y_components, axis=0)
    if rank == 0:
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
    check(y, target_y)
    raf.distributed.Synchronize()

def test_alltoallv(n_chunks):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, send_cnts):
            return raf._all_to_allv([x], [send_cnts])

    if raf.build.with_nccl() < 20700:
        print("alltoall is not supported in NCCL < 2.10")
        return

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x = np.ones(shape=(total_rank * n_chunks, 4), dtype="float32") * (rank + 1)
    send_cnts = []
    for i in range(total_rank):
        for j in range(n_chunks):
            send_cnts.append((i + j + local_rank) % 4 + 1)
    send_cnts = np.array(send_cnts, dtype="uint64")
    x = raf.array(x, device=device)
    send_cnts = raf.array(send_cnts, device=device)
    # if rank == 0:
    print(f"{rank} - X: ", x)
    print(f"{rank} - send_cnts: ", send_cnts)
    model.to(device=device)
    ret = model(x, send_cnts)
    y = ret[0]
    recv_cnts = ret[1]
    vx = np.ones(shape=(total_rank * n_chunks, 4), dtype="float32") * (rank + 1)
    vx = raf.array(vx, device=device)
    vret = run_vm_model(model, device, [vx, send_cnts])
    vy = vret[0]
    vrecv_cnts = vret[1]
    raf.distributed.Synchronize()
    check(y, vy)
    check(recv_cnts, vrecv_cnts)
    recv_cnts = recv_cnts.numpy()
    # if rank == 0:
    print(f"{rank} - recv_cnts: ", recv_cnts)
    y = y.numpy()

    target_y_components = []
    for i in range(total_rank):
        for j in range(n_chunks):
            target_y_components.append(np.ones(shape=(1, 4), dtype="float32") * (i + 1))
    target_y = np.concatenate(target_y_components, axis=0)
    # zero out all the elements that are not received
    for i in range(total_rank):
        for j in range(n_chunks):
            n_recved_elements = (i + j + local_rank) % 4 + 1
            target_y[i * n_chunks + j, n_recved_elements:] = 0
    for i in range(total_rank):
        for j in range(n_chunks):
            n_recved_elements = recv_cnts[i * n_chunks + j]
            y[i * n_chunks + j, n_recved_elements:] = 0
    # if rank == 0:
    print(f"{rank} - Y: ", y)
    print(f"{rank} - T: ", target_y)
    check(y, target_y)

# somehow the test will fail if run consecutively
if __name__ == "__main__":
    # test_alltoall()
    test_alltoallv(1)
    # test_alltoallv(2)
    # test_alltoallv(3)