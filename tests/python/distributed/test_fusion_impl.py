# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use,too-many-arguments
import os

import raf
import numpy as np
from raf._core.device import Device
from raf._ffi.pass_ import TestFusionImpl, DispatchDialect, InferType

def test_allreduce():
    print("test allreduce")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.add(x, x)
                x = raf.concatenate((x, x), axis=1)
                # single input allreduce
                x_1 = raf.allreduce(x)
                x_2 = raf.allreduce(x)
                x = raf.add(x_1, x_2)
                x = (x, x)
                # multi input allreduce
                x_1 = raf.allreduce(x)
                x_2 = raf.allreduce(x)
                x = raf.add(x_1[0], x_2[0])
                return x

        shape = (500, 500)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        mod = InferType()(mod)
        print("Original mod: ", raf.ir.AsText(mod))
        mod = TestFusionImpl()(mod)
        print("After TestFusionImpl: ", raf.ir.AsText(mod))

def test_reduce_scatter():
    print("test reduce_scatter")
    shapes = [500, 1000]
    shape_indices = [2]
    shapes_ = [500, 1000, 500, 1000]
    shape_indices_ = [2, 4]
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.add(x, x)
                x = raf.concatenate((x, x), axis=1)
                # single input reduce_scatter
                x_1 = raf.reduce_scatter(x, shapes, shape_indices)
                x_2 = raf.reduce_scatter(x, shapes, shape_indices)
                x = raf.add(x_1, x_2)
                x = (x, x)
                # multi input allreduce
                x_1 = raf.reduce_scatter(x, shapes_, shape_indices_)
                x_2 = raf.reduce_scatter(x, shapes_, shape_indices_)
                x = raf.add(x_1[0], x_2[0])
                return x

        shape = (500, 500)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        print("Original mod: ", raf.ir.AsText(mod))
        mod = InferType()(mod)
        print("Original mod: ", raf.ir.AsText(mod))
        mod = TestFusionImpl()(mod)
        print("After TestFusionImpl: ", raf.ir.AsText(mod))

def test_allgather():
    print("test allgather")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.add(x, x)
                x = raf.concatenate((x, x), axis=1)
                # single input allgather
                x_1 = raf.allgather(x, 1)
                x_2 = raf.allgather(x, 1)
                x = raf.add(x_1, x_2)
                x = (x, x)
                x_1 = raf.allgather(x, 1)
                x_2 = raf.allgather(x, 1)
                x = raf.add(x_1[0], x_2[0])
                return x

        shape = (5, 5)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        print("Original mod: ", raf.ir.AsText(mod))
        mod = InferType()(mod)
        print("Original mod: ", raf.ir.AsText(mod))
        mod = TestFusionImpl()(mod)
        print("After TestFusionImpl: ", raf.ir.AsText(mod))

if __name__ == "__main__":
    os.environ["DEBUG_CM_FIXED_PARAMS"] = "20;100000"
    test_allreduce()
    test_reduce_scatter()
    test_allgather()
