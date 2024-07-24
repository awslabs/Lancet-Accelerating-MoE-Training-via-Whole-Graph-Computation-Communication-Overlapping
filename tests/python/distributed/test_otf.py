# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use,too-many-arguments
import os

import raf
import numpy as np
from raf._core.device import Device
from raf.ir import PassContext
from raf._ffi.pass_ import DispatchDialect, InferType, EnforceSync
from raf.distributed import get_context
from raf.testing import run_vm_model, check, get_dist_info

def test_allreduce_otf():
    print("test otf")
    dctx = get_context()
    local_rank = dctx.local_rank
    with Device(f"cuda({local_rank})"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.add(x, x)
                x = raf.concatenate((x, x), axis=1)
                x_1 = raf.add(x, x)
                x_2 = raf.add(x, x_1)
                x_3 = raf.add(x, x_2)
                x_4 = raf.add(x, x_3)
                x = (x_1, x_2, x_3, x_4)
                x = raf.allreduce(x)
                x = raf.add(x[0], x[1])
                return x

        shape = (5, 5)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")

        with PassContext(
            config = {
                "raf.enforce_sync.on_the_fly_gradient_copy": True,
            }
        ):
            mod = model._internal(x).mod
            mod = DispatchDialect()(mod)
            mod = InferType()(mod)
            print("Original mod: ", raf.ir.AsText(mod))
            mod = EnforceSync(dctx.size)(mod)
            print("After EnforceSync: ", raf.ir.AsText(mod))

def test_allreduce_otf_1():
    print("test otf 1")
    dctx = get_context()
    local_rank = dctx.local_rank
    with Device(f"cuda({local_rank})"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.add(x, x)
                x = raf.concatenate((x, x), axis=1)
                x = (x, x, x, x)
                x = raf.allreduce(x)
                x = raf.add(x[0], x[1])
                return x

        shape = (5, 5)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")

        with PassContext(
            config = {
                "raf.enforce_sync.on_the_fly_gradient_copy": True,
            }
        ):
            mod = model._internal(x).mod
            mod = DispatchDialect()(mod)
            mod = InferType()(mod)
            print("Original mod: ", raf.ir.AsText(mod))
            mod = EnforceSync(dctx.size)(mod)
            print("After EnforceSync: ", raf.ir.AsText(mod))

def test_reduce_scatter_otf():
    print("test reduce_scatter otf")
    dctx = get_context()
    dctx.enable_data_parallel = True
    local_rank = dctx.local_rank
    device = f"cuda({local_rank})"
    with Device(device):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y):
                x = raf.concatenate((x, y), axis=0)
                x_1 = raf.add(x, x)     # 4, 8
                x_2 = raf.add(x, x_1)   # 6, 12
                x_3 = raf.add(x, x_2)   # 8, 16
                x_4 = raf.add(x, x_3)   # 10, 20
                x = (x_1, x_2, x_3, x_4)
                x = raf.reduce_scatter(x, (2, 8, 2, 8, 2, 8, 2, 8), (2, 4, 6, 8))
                x0 = x[0] + x[0]        # 24
                x1 = x[1] + x[1]        # 36
                x2 = x[2] + x[2]        # 48
                x3 = x[3] + x[3]        # 60
                return x0, x1, x2, x3

        shape = (4, 4)
        model = Model()
        n_x = np.ones(shape=shape, dtype="float32") * (local_rank + 1)
        m_x = raf.array(np.ones(shape=shape, dtype="float32") * (local_rank + 1), dtype="float32")
        model.to(device=device)

        with PassContext(
            config = {
                "raf.enforce_sync.on_the_fly_gradient_copy": True,
            }
        ):
            mod = model._internal(m_x).mod
            mod = DispatchDialect()(mod)
            mod = InferType()(mod)
            print("Original mod: ", raf.ir.AsText(mod))
            mod = EnforceSync(dctx.size)(mod)
            mod = InferType()(mod)
            print("After EnforceSync: ", raf.ir.AsText(mod))

def test_reduce_scatter_with_tensor_list(computation):
    shape0 = (12, 12)
    shape1 = (5, 5)
    dctx = get_context()
    dctx.enable_data_parallel = True
    local_rank = dctx.local_rank
    device = f"cuda({local_rank})"
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
            o0 = out[0] + out[0]
            o1 = out[1] + out[1]
            return o0, o1

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
    m_out, m_out1 = run_vm_model(model, device,[m_x, m_y, m_z, m_i])
    if rank == 0:
        if computation == "sum":
            n_out = n_ones * sum(range(1, total_rank + 1)) * 2
            n_out1 = -n_ones1 * sum(range(4, total_rank + 4)) * 2
        elif computation == "prod":
            n_out = n_ones * np.prod(range(1, total_rank + 1)) * 2
            n_out1 = n_ones1 * np.prod(range(4, total_rank + 4)) * 2
        elif computation == "min":
            n_out = n_ones * min(1, total_rank) * 2
            n_out1 = -n_ones1 * max(4, total_rank + 3) * 2
        elif computation == "max":
            n_out = n_ones * max(1, total_rank) * 2
            n_out1 = -n_ones1 * min(4, total_rank + 3) * 2
        elif computation == "avg":
            n_out = n_ones * sum(range(1, total_rank + 1)) * 2
            n_out = n_out / total_rank
            n_out1 = -n_ones1 * sum(range(4, total_rank + 4)) * 2
            n_out1 = n_out1 / total_rank
        else:
            assert False, "Invalid computation"
        check(m_out, n_out)
        check(m_out1, n_out1)
    elif rank == 1:
        if computation == "sum":
            n_out = -n_ones * sum(range(1, total_rank + 1)) * 2
            n_out1 = n_ones1 * sum(range(4, total_rank + 4)) * 2
        elif computation == "prod":
            n_out = n_ones * np.prod(range(1, total_rank + 1)) * 2
            n_out1 = n_ones1 * np.prod(range(4, total_rank + 4)) * 2
        elif computation == "min":
            n_out = -n_ones * max(1, total_rank) * 2
            n_out1 = n_ones1 * min(4, total_rank + 3) * 2
        elif computation == "max":
            n_out = -n_ones * min(1, total_rank) * 2
            n_out1 =  n_ones1 * max(4, total_rank + 3) * 2
        elif computation == "avg":
            n_out = -n_ones * sum(range(1, total_rank + 1)) * 2
            n_out = n_out / total_rank
            n_out1 = n_ones1 * sum(range(4, total_rank + 4)) * 2
            n_out1 = n_out1 / total_rank
        check(m_out, n_out)
        check(m_out1, n_out1)

if __name__ == "__main__":
    os.environ["DEBUG_CM_FIXED_PARAMS"] = "20;100000"
    # test_allreduce_otf()
    # test_allreduce_otf_1()
    # test_reduce_scatter_otf()
    test_reduce_scatter_with_tensor_list("sum")
    test_reduce_scatter_with_tensor_list("prod")
    test_reduce_scatter_with_tensor_list("min")
    test_reduce_scatter_with_tensor_list("max")
    test_reduce_scatter_with_tensor_list("avg")
    raf.distributed.RemoveCommunicator()
