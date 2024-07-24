# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use,too-many-arguments
import os
from copy import deepcopy
import tvm

import raf
import numpy as np
from raf._core.device import Device
from raf.ir import const
from raf._ffi.pass_ import TestPartitionImpl, DispatchDialect, InferType, ToGraphNormalForm, ToBasicBlockNormalForm, FuseTVM
from raf._core.ir_ext import extended_var
from raf.ir import ANFBuilder
from raf.testing import with_dialect

def make_function(func):
    mod = tvm.IRModule.from_expr(func)
    mod = ToGraphNormalForm()(mod)
    mod = ToBasicBlockNormalForm()(mod)
    mod = FuseTVM()(mod)
    mod = InferType()(mod)
    return mod["main"]

def expand_grid(grid):
    def expand_grid_(grid, combinations):
        if len(grid) is 0:
            return combinations
        else:
            key = list(grid.keys())[0]
            result = []
            for value in grid[key]:
                combinations_ = deepcopy(combinations)
                for item in combinations_:
                    item[key] = value
                result.extend(combinations_)
            del grid[key]
            return expand_grid_(grid, result)

    combinations = [{}]
    return expand_grid_(grid, combinations)

def set_env(envs, orig_envs):
    os.environ = orig_envs
    for key, value in envs.items():
        os.environ[key] = value

def run_tests(env_grid, funcs):
    orig_envs = os.environ
    env_settings = expand_grid(env_grid)
    for env_setting in env_settings:
        print("current env setting ", env_setting)
        set_env(env_setting, orig_envs)
        for func in funcs:
            func()

def test_simple():
    print("test simple")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, z):
                xpy = raf.add(x, y)
                xmz = raf.subtract(x, z)
                xmz = raf.all_to_all(xmz)
                mul = raf.multiply(xpy, xmz)
                return mul

        shape = (500, 500)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        z = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y, z).mod
        
        print("Original mod ", raf.ir.AsText(mod))
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print("Partitioned mod ", raf.ir.AsText(mod))


def test_split_tuple_tgi():
    print("test split tuple tgi")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y):
                x1 = raf.add(x, y)
                xs = raf.split(x1, 2, 1)
                xmz = raf.all_to_all(xs[0])
                mul = raf.multiply(xmz, xs[1])
                return mul

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

def test_matmul():
    print("test matmul")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, mm_w0, mm_w1):
                x1 = raf.matmul(x, mm_w0)
                x2 = raf.all_to_all(x1)
                x3 = raf.matmul(x2, mm_w1)
                return x3

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        z = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y, z).mod
        print("Original mod ", raf.ir.AsText(mod))
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print("Partitioned mod ", raf.ir.AsText(mod))

def test_matmul_all_reduce():
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, mm_w0, mm_w1):
                x1 = raf.matmul(x, mm_w0)
                x2 = raf.allreduce(x1)
                x3 = raf.matmul(x2[0], mm_w1)
                return x3

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        z = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y, z).mod
        print("Original mod ", raf.ir.AsText(mod))
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print("Partitioned mod ", raf.ir.AsText(mod))

def test_find_roi():
    print("test find roi")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, mm_w0, mm_w1):
                x1 = raf.matmul(x, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x2 = raf.all_to_all(x1)
                x3 = raf.matmul(x2, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                return x3

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        z = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y, z).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

def test_find_roi_complex():
    print("test find roi complex")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, mm_w0, mm_w1):
                x1 = raf.add(x, y)
                x2 = raf.subtract(x, y)
                x3 = raf.matmul(x1, x2)
                x4 = raf.matmul(x3, mm_w0)
                x5 = raf.all_to_all(x4)
                x6 = raf.matmul(x5, mm_w1)
                x7 = raf.multiply(x6, x2)
                return x7

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        w0 = raf.array(np.random.randn(*shape), dtype="float32")
        w1 = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y, w0, w1).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

def test_find_roi_unregistered():
    print("test find roi unregistered")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y):
                x1 = raf.add(x, y)
                x2 = raf.relu(x1) # this should be a currently unregistered op
                x3 = raf.all_to_all(x1)
                x4 = raf.add(x3, x2)
                return x4

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_function():
    print("test function")
    shape = (1024, 1024)
    with Device("cuda(0)"):
        def construct_test_func():
            builder = ANFBuilder()
            x = extended_var("x", shape=shape, dtype="float32")
            y = extended_var("y", shape=shape, dtype="float32")
            z = extended_var("z", shape=shape, dtype="float32")
            x_1 = builder.call("_all_to_all", [x])
            x_2 = builder.call("multiply", [x_1, y])
            x_2 = builder.call("multiply", [x_2, x])
            x_2 = builder.call("multiply", [x_2, x])
            x_2 = builder.call("multiply", [x_2, x])
            x_3 = builder.call("matmul", [x_2, z])
            return make_function(tvm.relay.Function([x, y, z], builder.ret(x_3)))

        mod = tvm.IRModule()
        mod["main"] = construct_test_func()
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

def test_e2e_opt_multiple_rois():
    print("test e2e op multiple rois")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, mm_w0, mm_w1):
                x1 = raf.matmul(x, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x1 = raf.matmul(x1, mm_w0)
                x2 = raf.all_to_all(x1)
                x3 = raf.matmul(x2, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x3 = raf.matmul(x3, mm_w1)
                x4 = raf.all_to_all(x3)
                x5 = raf.matmul(x4, mm_w1)
                x5 = raf.matmul(x5, mm_w1)
                x5 = raf.matmul(x5, mm_w1)
                x5 = raf.matmul(x5, mm_w1)
                return x5

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        z = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y, z).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

def test_obstacle():
    print("test obstacle")
    # matmul exec time is approximately 200 ms under current setting
    # all2all time computed by cost model is around 600 ms (3 times of matmul)
    # we put different count of matmuls between all2alls to see
    # whether the algorithm can overcome the obstacle of matmuls 
    # to find a large pipeline
    # TODO(@ye-tian): adjust cm params to make all2all time around 600 ms
    # or any proper exec time.
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, mm_w0):
                x = raf.matmul(x, mm_w0)
                x = raf.all_to_all(x)
                x = raf.matmul(x, mm_w0)
                x = raf.all_to_all(x)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.all_to_all(x)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.all_to_all(x)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.all_to_all(x)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.matmul(x, mm_w0)
                x = raf.all_to_all(x)
                return x

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x, y).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_concat():
    print("test concat")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.concatenate((x, x, x, x), 1)
                x = raf.all_to_all(x)
                return x

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_allreduce():
    print("test allreduce")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.concatenate((x, x, x, x), 1)
                x = raf.allreduce(x)
                return x

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_cast():
    print("test cast")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x):
                x = raf.cast(x, "float32")
                x = raf.all_to_all(x)
                return x

        shape = (1024, 1024)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float16")

        mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))


@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_bias_add():
    print("test bias add")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y):
                x = raf.bias_add(x, y, 0)
                x = raf.all_to_all(x)
                return x

        shape = (1024, 1024)
        shape_ = (1024,)
        model = Model()
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape_), dtype="float32")

        mod = model._internal(x, y).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))


@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_where():
    print("test where")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, cond, x, y, w):
                x = raf.matmul(x, w)
                x = raf.where(cond, x, y)
                x = raf.all_to_all(x)
                return x

        shape = (1024, 1024)
        model = Model()
        cond = raf.array(np.random.randn(*shape), dtype="float32")
        x = raf.array(np.random.randn(*shape), dtype="float32")
        y = raf.array(np.random.randn(*shape), dtype="float32")
        w = raf.array(np.random.randn(*shape), dtype="float32")

        mod = model._internal(cond, x, y, w).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_moe():
    print("test moe")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, z):
                x = raf.matmul_tn(x, y)
                x = raf.reshape(x, (16, 128, 768))
                x = raf.all_to_all(x)
                x = raf.reshape(x, (8, 2, 128, 768))
                x = raf.split(x, [1], 1)
                x = raf.reshape(x[0], (1024, 768))
                x = raf.matmul_nt(x, z)
                return x

        x_shape = (2048, 2048)
        y_shape = (2048, 768)
        z_shape = (3072, 768)
        model = Model()
        x = raf.array(np.random.randn(*x_shape), dtype="float32")
        y = raf.array(np.random.randn(*y_shape), dtype="float32")
        z = raf.array(np.random.randn(*z_shape), dtype="float32")

        mod = model._internal(x, y, z).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))


@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_argmax():
    print("test argmax")
    with Device("cuda(0)"):
        def create_test_mod(axis=[0], keepdims=False, exclude=False):
            class Model(raf.Model):
                def build(self):
                    pass

                @raf.model.trace
                def forward(self, x, w):
                    x = raf.matmul(x, w)
                    x = raf.argmax(x, axis=axis, keepdims=keepdims, exclude=exclude)
                    x = raf.all_to_all(x)
                    return x

            shape = (1024, 1024)
            model = Model()
            x = raf.array(np.random.randn(*shape), dtype="float32")
            w = raf.array(np.random.randn(*shape), dtype="float32")

            mod = model._internal(x, w).mod
            return mod

        args_grid = {
            "axis": [[0], [1], [0, 1]],
            "keepdims": [True, False],
            "exclude": [False]
        }
        for args in expand_grid(args_grid):
            print("Args: ", args)
            mod = create_test_mod(**args)
            mod = DispatchDialect()(mod)
            mod = TestPartitionImpl()(mod)
            print(raf.ir.AsText(mod))
        args_grid = {
            "axis": [[0], [1]],
            "keepdims": [True, False],
            "exclude": [True]
        }
        for args in expand_grid(args_grid):
            print("Args: ", args)
            mod = create_test_mod(**args)
            mod = DispatchDialect()(mod)
            mod = TestPartitionImpl()(mod)
            print(raf.ir.AsText(mod))


@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_moe_fw():
    print("test moe fw")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, z, scale, bias, gate):
                x = raf.add(x, y)
                x = raf.reshape(x, (32, 128, 768))
                x = raf.add(x, z)
                x1 = raf.layer_norm(x, scale, bias)
                x2 = raf.reshape(x1, (4096, 768))
                encode_out = raf.moe_encode(x2, gate)
                gate_s = encode_out[0]
                indices_s = encode_out[1]
                locations_s = encode_out[2]
                dispatched = encode_out[3]
                dispatched = raf.all_to_all(dispatched)
                decoded = raf.moe_decode(dispatched, gate_s, indices_s, locations_s)
                res_decoded = raf.reshape(decoded, (32, 128, 768))
                out = raf.add(x, res_decoded)
                return out

        x_shape = (4096, 768)
        y_shape = (768, )
        z_shape = (32, 128, 768)
        gate_shape = (4096, 32)
        model = Model()
        x = raf.array(np.random.randn(*x_shape), dtype="float32")
        y = raf.array(np.random.randn(*y_shape), dtype="float32")
        z = raf.array(np.random.randn(*z_shape), dtype="float32")
        scale = raf.array(np.random.randn(*y_shape), dtype="float32")
        bias = raf.array(np.random.randn(*y_shape), dtype="float32")
        gate = raf.array(np.random.randn(*gate_shape), dtype="float32")

        mod = model._internal(x, y, z, scale, bias, gate).mod
        func = mod["main"]
        fused_func = make_function(func)
        mod = tvm.IRModule()
        mod["main"] = fused_func
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_tuple():
    print("test tuple")
    with Device("cuda(0)"):
        def construct_test_func():
            builder = ANFBuilder()
            x = extended_var("x", shape=(4096,), dtype="float32")
            y = extended_var("y", shape=(4096,), dtype="int32")
            z = extended_var("z", shape=(4096,), dtype="int32")
            i = extended_var("i", shape=(32, 128, 768), dtype="float32")
            x_1 = builder.make_tuple([x, y, z, i])
            x_2 = builder.get_tuple_item(x_1, 3)
            x_3 = builder.make_tuple([x_2])
            x_4 = builder.call("_all_to_all", [x_3])
            return tvm.relay.Function([x, y, z, i], builder.ret(x_4))

        mod = tvm.IRModule()
        mod["main"] = construct_test_func()
        # mod = model._internal(x).mod
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_broadcast():
    print("test moe fw")
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, z, scale, bias):
                x = raf.add(x, y)
                x = raf.reshape(x, (32, 128, 768))
                x = raf.add(x, z)
                x1 = raf.layer_norm(x, scale, bias)
                x1 = raf.all_to_all(x1)
                return x1

        x_shape = (4096, 768)
        y_shape = (768, )
        z_shape = (32, 128, 768)
        model = Model()
        x = raf.array(np.random.randn(*x_shape), dtype="float32")
        y = raf.array(np.random.randn(*y_shape), dtype="float32")
        z = raf.array(np.random.randn(*z_shape), dtype="float32")
        scale = raf.array(np.random.randn(*y_shape), dtype="float32")
        bias = raf.array(np.random.randn(*y_shape), dtype="float32")

        mod = model._internal(x, y, z, scale, bias).mod
        func = mod["main"]
        fused_func = make_function(func)
        mod = tvm.IRModule()
        mod["main"] = fused_func
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_batch_matmul_nt():
    print("test batch_matmul_nt")

    x_shape = (384, 128, 128)
    y_shape = (32, 128, 768)
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, scale, bias):
                y = raf.reshape(y, (32, 12, 128, 64))
                y = raf.transpose(y, [0, 2, 3, 1])
                y = raf.reshape(y, [384, 64, 128])
                a1 = raf.batch_matmul_nt(x, y)
                a2 = raf.reshape(a1, (32, 12, 128, 64))
                a3 = raf.transpose(a2, [0, 2, 1, 3])
                a4 = raf.reshape(a3, [4096, 768])
                a5 = raf.layer_norm(a4, scale, bias)
                out = raf.all_to_all(a5)
                return out

        scale_shape = (768, )
        model = Model()
        x = raf.array(np.random.randn(*x_shape), dtype="float32")
        y = raf.array(np.random.randn(*y_shape), dtype="float32")
        scale = raf.array(np.random.randn(*scale_shape), dtype="float32")
        bias = raf.array(np.random.randn(*scale_shape), dtype="float32")

        mod = model._internal(x, y, scale, bias).mod
        func = mod["main"]
        fused_func = make_function(func)
        mod = tvm.IRModule()
        mod["main"] = fused_func
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_batch_matmul_nt_1():
    print("test batch_matmul_nt_1")

    x_shape = (32, 128, 768)
    y_shape = (32, 128, 768)
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, scale, bias):
                x = raf.reshape(x, (32, 128, 12, 64))
                x = raf.transpose(x, [0, 2, 1, 3])
                x = raf.reshape(x, [384, 128, 64])

                y = raf.reshape(y, (32, 128, 12, 64))
                y = raf.transpose(y, [0, 2, 1, 3])
                y = raf.reshape(y, [384, 128, 64])
                a1 = raf.batch_matmul_nt(x, y)
                a5 = raf.layer_norm(a1, scale, bias)
                out = raf.all_to_all(a5)
                return out

        scale_shape = (768, )
        model = Model()
        x = raf.array(np.random.randn(*x_shape), dtype="float32")
        y = raf.array(np.random.randn(*y_shape), dtype="float32")
        scale = raf.array(np.random.randn(*scale_shape), dtype="float32")
        bias = raf.array(np.random.randn(*scale_shape), dtype="float32")

        mod = model._internal(x, y, scale, bias).mod
        func = mod["main"]
        fused_func = make_function(func)
        mod = tvm.IRModule()
        mod["main"] = fused_func
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cudnn", "cublas", "nccl", "cuda", "tvm"])
def test_batch_matmul_nt_2():
    print("test batch_matmul_nt_2")

    x_shape = (64, 12, 128, 128)
    y_shape = (64, 128, 768)
    w0_shape = (768, 768)
    w1_shape = (768, )
    w2_shape = (64, 128, 768)
    sb_shape = (768, )
    with Device("cuda(0)"):
        class Model(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, y, w0, w1, w2, scale, bias):
                x = raf.softmax(x, -1)
                x = raf.reshape(x, (768, 128, 128))

                y = raf.reshape(y, (64, 128, 12, 64))
                y = raf.transpose(y, [0, 2, 3, 1])
                y = raf.reshape(y, (768, 64, 128))

                z = raf.batch_matmul_nt(x, y)
                z = raf.reshape(z, (64, 12, 128, 64))
                z = raf.transpose(z, [0, 2, 1, 3])
                z = raf.reshape(z, (8192, 768))

                z = raf.matmul(z, w0)

                z = raf.add(z, w1)
                z = raf.reshape(z, (64, 128, 768))
                z = raf.add(z, w2)

                z = raf.layer_norm(z, scale, bias)

                out = raf.all_to_all(z)
                return out

        model = Model()
        x = raf.array(np.random.randn(*x_shape), dtype="float32")
        y = raf.array(np.random.randn(*y_shape), dtype="float32")
        w0 = raf.array(np.random.randn(*w0_shape), dtype="float32")
        w1 = raf.array(np.random.randn(*w1_shape), dtype="float32")
        w2 = raf.array(np.random.randn(*w2_shape), dtype="float32")
        scale = raf.array(np.random.randn(*sb_shape), dtype="float32")
        bias = raf.array(np.random.randn(*sb_shape), dtype="float32")

        mod = model._internal(x, y, w0, w1, w2, scale, bias).mod
        func = mod["main"]
        fused_func = make_function(func)
        mod = tvm.IRModule()
        mod["main"] = fused_func
        mod = DispatchDialect()(mod)
        mod = TestPartitionImpl()(mod)
        print(raf.ir.AsText(mod))

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_where_broadcast():
    print("test where")

    cond_shape = (1, 768)
    x_shape = (4096, 768)
    # y_shape = (4096, 768)
    y_shape = ()
    gate_shape = (4096, 32)
    with Device("cuda(0)"):
        config = {"relay.backend.use_auto_scheduler": True}
        with tvm.transform.PassContext(
            config=config,
            disabled_pass={"AutoSchedulerLayoutRewrite"},
        ):
            class Model(raf.Model):
                def build(self):
                    pass

                @raf.model.trace
                def forward(self, cond, x, y, gate):
                    x = raf.where(cond, x, y)
                    encode_out = raf.moe_encode(x, gate)
                    out = raf.all_to_all(encode_out[3])
                    return out

            model = Model()
            cond = raf.array(np.random.randn(*cond_shape), dtype="float32")
            x = raf.array(np.random.randn(*x_shape), dtype="float32")
            y = raf.array(np.random.randn(*y_shape), dtype="float32")
            gate = raf.array(np.random.randn(*gate_shape), dtype="float32")

            mod = model._internal(cond, x, y, gate).mod
            mod = DispatchDialect()(mod)
            mod = TestPartitionImpl()(mod)
            print(raf.ir.AsText(mod))

if __name__ == "__main__":
    env_grid = {
        "DISABLE_FUSION": ["1"],
        # "TEST_PARTITION": ["1"],
        # "ALWAYS_APPLY_PARTITION": ["1"],
        "MIN_IDLE_TIME": ["10"],
        "MAX_PARTITION": ["4"],
        "DEBUG_CM_FIXED_PARAMS": ["20;100000"],
        "BALANCE_CONSTRAINT_PARAMS": ["3;1000"],
        "MAX_ITERATIONS": ["3"],
        "PIPELINE_PRUNE_THRESHOLD": ["1"]
        # "SIMULATION_DEBUG_PREFIX": ["./test_moe_sim"]
    }
    funcs = [
        test_batch_matmul_nt_2
        # test_allreduce,
        # test_cast
    ]
    run_tests(env_grid, funcs)
