# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
import sys
import os
import pytest
import numpy as np

os.environ["LANCET_COMM_PARTITION_SIZE"] = 32

import raf
from raf import distributed as dist
from raf._core.ndarray import Symbol
from raf.testing import check, get_dist_info, skip_dist_test, run_vm_model, run_model

dctx = dist.get_context()
SKIP_REASON = "Distribution is not enabled or #rank is not expected"

@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_partition_allreduce(computation):
    dctx.enable_data_parallel = True
    print("Testing allreduce.")
    shape = (4, 4)

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1):
            x = x1 + x1
            x = raf.allreduce([x], computation=computation)
            x = x + x
            return x

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=shape, dtype="float32") * (rank + 1)
    x1 = raf.array(x1, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1])
    model.to(device=device)
    m_out = run_vm_model(model, device, [x1], partition_comm=True)
    if rank == 0:
        ones = np.ones(shape=shape, dtype="float32") * 2
        if computation == "sum":
            target_y = ones * sum(range(1, total_rank + 1))
            target_y = target_y * 2
        elif computation == "prod":
            target_y = ones * np.prod(range(1, total_rank + 1)) * (2**total_rank)
        elif computation == "min":
            target_y = ones * 2
        elif computation == "max":
            target_y =ones * total_rank * 2
        elif computation == "avg":
            target_y = ones * sum(range(1, total_rank + 1)) * 2
            target_y = target_y / total_rank
        else:
            assert False, "Invalid computation"
        print(f"{rank} - Y: ", m_out.numpy())
        print(f"{rank} - T: ", target_y)
        check(m_out, target_y)

@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
def test_partition_allgather():
    dctx.enable_data_parallel = True
    print("Testing allgather.")
    shape = (4, 4)

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1):
            x1 = x1 + x1
            x = raf.allgather([x1], axis=0)
            x = x + x
            return x

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=shape, dtype="float32") * (rank + 1)
    x1 = raf.array(x1, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1])
    model.to(device=device)
    y = run_vm_model(model, device, [x1], partition_comm=True)
    if rank == 0:
        x1 = x1.numpy() * 2
        target_y1 = np.concatenate([x1 * (r + 1) for r in range(total_rank)], axis=0) * 2
        print(f"{rank} - Y: ", y.numpy())
        print(f"{rank} - T: ", target_y1)
        check(y, target_y1)

@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_partition_reduce_scatter(computation):
    dctx.enable_data_parallel = True
    shape0 = (12, 12)
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y):
            x = x + x
            y = y + y
            shapes = list(shape0)
            shape_indices = [2]
            xy = raf.concatenate([x, y], axis=0)
            out = raf.reduce_scatter([xy], shapes, shape_indices, computation=computation)
            out = out + out
            return out

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=shape0, dtype="float32")
    n_x = n_ones * (rank + 1)
    n_y = -n_ones * (rank + 1)
    m_x, m_y = raf.array(n_x, device=device), raf.array(n_y, device=device)
    model.to(device=device)
    m_out = run_vm_model(model, device, [m_x, m_y], partition_comm=True)
    if rank == 0:
        if computation == "sum":
            n_out = n_ones * sum(range(1, total_rank + 1)) * 4
        elif computation == "prod":
            n_out = 2 * n_ones * np.prod(range(1, total_rank + 1)) * (2**total_rank)
        elif computation == "min":
            n_out = n_ones * min(1, total_rank) * 4
        elif computation == "max":
            n_out = n_ones * max(1, total_rank) * 4
        elif computation == "avg":
            n_out = n_ones * sum(range(1, total_rank + 1)) * 4
            n_out = n_out / total_rank
        else:
            assert False, "Invalid computation"
        check(m_out, n_out)
    elif rank == 1:
        if computation == "sum":
            n_out = -n_ones * sum(range(1, total_rank + 1)) * 4
        elif computation == "prod":
            n_out = 2 * n_ones * np.prod(range(1, total_rank + 1)) * (2**total_rank)
        elif computation == "min":
            n_out = -n_ones * max(1, total_rank) * 4
        elif computation == "max":
            n_out = -n_ones * min(1, total_rank) * 4
        elif computation == "avg":
            n_out = -n_ones * sum(range(1, total_rank + 1)) * 4
            n_out = n_out / total_rank
        check(m_out, n_out)

if __name__ == "__main__":
    exit_code = pytest.main([__file__])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
