# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test collective_communication, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_collective_communication.py`
(in ci/tash_python_unittest.sh)
"""
import sys
import pytest
import numpy as np

import raf
from raf import distributed as dist
from raf._core.ndarray import Symbol
from raf.testing import check, get_dist_info, skip_dist_test, run_vm_model, run_model

dctx = dist.get_context()
SKIP_REASON = "Distribution is not enabled or #rank is not expected"

@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_allreduce_with_tensor_list(computation):
    print("Testing allreduce with a list of tensors as input.")

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):
            x = raf.allreduce([x1, x2], computation=computation)
            a = x[0]
            b = x[1]
            return raf.concatenate((a, b))

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = model(x1, x2)
    vx1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    vx2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    vx1 = raf.array(vx1, device=device)
    vx2 = raf.array(vx2, device=device)
    run_vm_model(model, device, [vx1, vx2])
    y = run_model(model, [x1, x2], device)
    if rank == 0:
        ones = np.ones(shape=(4, 4), dtype="float32")
        if computation == "sum":
            target_y = np.concatenate(
                [ones * sum(range(1, total_rank + 1)), ones * -sum(range(1, total_rank + 1))]
            )
        elif computation == "prod":
            sign = 1 if total_rank % 2 == 0 else -1
            target_y = np.concatenate(
                [
                    ones * np.prod(range(1, total_rank + 1)),
                    ones * sign * np.prod(range(1, total_rank + 1)),
                ]
            )
        elif computation == "min":
            target_y = np.concatenate([ones, ones * -total_rank])
        elif computation == "max":
            target_y = np.concatenate([ones * total_rank, ones * -1])
        elif computation == "avg":
            target_y = np.concatenate(
                [ones * sum(range(1, total_rank + 1)), ones * -sum(range(1, total_rank + 1))]
            )
            target_y = target_y / total_rank
        else:
            assert False, "Invalid computation"
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)
        vy = np.concatenate([vx1.numpy(), vx2.numpy()])
        check(vy, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
@pytest.mark.parametrize("axis", [0])
def test_allgather_with_tensor_list(axis):
    print("Testing allgather with a list of tensors as input.")

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x1, x2):
            x = raf.allgather([x1, x2], axis=axis)
            return raf.concatenate(x)

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    x1 = np.ones(shape=(4, 4), dtype="float32") * (rank + 1)
    x2 = np.ones(shape=(4, 4), dtype="float32") * (-rank - 1)
    x1 = raf.array(x1, device=device)
    x2 = raf.array(x2, device=device)
    if rank == 0:
        print(f"{rank} - X: ", [x1, x2])
    model.to(device=device)
    y = run_model(model, [x1, x2], device)
    if rank == 0:
        x1 = x1.numpy()
        x2 = x2.numpy()
        target_y1 = np.concatenate([x1 * (r + 1) for r in range(total_rank)], axis=axis)
        target_y2 = np.concatenate([x2 * (r + 1) for r in range(total_rank)], axis=axis)
        target_y = np.concatenate([target_y1, target_y2])
        print(f"{rank} - Y: ", y)
        print(f"{rank} - T: ", target_y)
        check(y, target_y)


@pytest.mark.skipif(skip_dist_test(min_rank_num=2, require_exact_rank=True), reason=SKIP_REASON)
@pytest.mark.parametrize("computation", ["sum", "prod", "min", "max"])
def test_reduce_scatter_with_tensor_list(computation):
    shape0 = (12, 12)
    shape1 = (5, 5)
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y, z, i):
            shapes = list(shape0) + list(shape1)
            shape_indices = [2, 4]
            xy = raf.concatenate([x, y], axis=0)
            zi = raf.concatenate([z, i], axis=0)
            out = raf.reduce_scatter([xy, zi], shapes, shape_indices, computation=computation)
            return out

    if computation == "avg" and raf.build.with_nccl() < 21000:
        pytest.skip("avg is not supported in NCCL < 2.10")

    model = TestModel()
    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    n_ones = np.ones(shape=shape0, dtype="float32")
    n_ones1 = np.ones(shape=shape1, dtype="float32")
    n_x = n_ones * (rank + 1)
    n_y = -n_ones * (rank + 1)
    n_z = -n_ones1 * (rank + 4)
    n_i = n_ones1 * (rank + 4)
    m_x, m_y = raf.array(n_x, device=device), raf.array(n_y, device=device)
    m_z, m_i = raf.array(n_z, device=device), raf.array(n_i, device=device)
    model.to(device=device)
    m_out, m_out1 = run_model(model, [m_x, m_y, m_z, m_i], device)
    if rank == 0:
        if computation == "sum":
            n_out = n_ones * sum(range(1, total_rank + 1))
            n_out1 = -n_ones1 * sum(range(4, total_rank + 4))
        elif computation == "prod":
            n_out = n_ones * np.prod(range(1, total_rank + 1))
            n_out1 = n_ones1 * np.prod(range(4, total_rank + 4))
        elif computation == "min":
            n_out = n_ones * min(1, total_rank)
            n_out1 = -n_ones1 * max(4, total_rank + 3)
        elif computation == "max":
            n_out = n_ones * max(1, total_rank)
            n_out1 = -n_ones1 * min(4, total_rank + 3)
        elif computation == "avg":
            n_out = n_ones * sum(range(1, total_rank + 1))
            n_out = n_out / total_rank
            n_out1 = -n_ones1 * sum(range(4, total_rank + 4))
            n_out1 = n_out1 / total_rank
        else:
            assert False, "Invalid computation"
        check(m_out, n_out)
        check(m_out1, n_out1)
    elif rank == 1:
        if computation == "sum":
            n_out = -n_ones * sum(range(1, total_rank + 1))
            n_out1 = n_ones1 * sum(range(4, total_rank + 4))
        elif computation == "prod":
            n_out = n_ones * np.prod(range(1, total_rank + 1))
            n_out1 = n_ones1 * np.prod(range(4, total_rank + 4))
        elif computation == "min":
            n_out = -n_ones * max(1, total_rank)
            n_out1 = n_ones1 * min(4, total_rank + 3)
        elif computation == "max":
            n_out = -n_ones * min(1, total_rank)
            n_out1 =  n_ones1 * max(4, total_rank + 3)
        elif computation == "avg":
            n_out = -n_ones * sum(range(1, total_rank + 1))
            n_out = n_out / total_rank
            n_out1 = n_ones1 * sum(range(4, total_rank + 4))
            n_out1 = n_out1 / total_rank
        check(m_out, n_out)
        check(m_out1, n_out1)

if __name__ == "__main__":
    exit_code = pytest.main([__file__])
    dist.RemoveCommunicator()
    sys.exit(exit_code)
