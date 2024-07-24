# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
from typing import Dict, List
import pytest
import numpy as np
import tvm

import raf
from raf._core.device import Device
from raf.testing import check, with_dialect
from raf._core.ir_ext import extended_var
from raf.ir import ScopeBuilder
from raf._core.ndarray import get_ndarray_handle, ndarray
from raf.testing.utils import get_vm_executor
from raf.utils.tuner import run_tuning
from raf._ffi.model import RunModel
from raf._ffi.pass_ import InferType


class ANFBuilder:
    def __init__(self):
        self.scope_builder = ScopeBuilder()
        self.operators: Dict[str, tvm.ir.Op] = {}

    def get_operator(self, op_name: str) -> tvm.ir.Op:
        if op_name not in self.operators:
            self.operators[op_name] = raf._ffi.op.GetOp(f"raf.op.{op_name}")
        return self.operators[op_name]

    def const(self, value):
        return raf.ir.const(value)
    
    def int_const(self, value):
        return raf.ir.const(int(value))
    
    def int_tuple_const(self, fields):
        return raf.ir.const(tuple([int(value) for value in fields]))

    def make_tuple(self, fields):
        return self.scope_builder.let('', tvm.relay.Tuple(fields))

    def get_tuple_item(self, tup, index):
        return self.scope_builder.let('', tvm.relay.TupleGetItem(tup, index))

    def call(self, op_name: str, args: List[tvm.relay.Expr]) -> tvm.relay.Var:
        return self.scope_builder.let('', tvm.relay.Call(self.get_operator(op_name), args))

    def reshape(self, tensor, shape):
        return self.call("reshape", [tensor, self.int_tuple_const(shape)])

    def set_stream(self, device_id: int, stream_id: int):
        device_id = raf.ir.const(device_id)
        stream_id = raf.ir.const(stream_id)
        return self.call("set_stream", [device_id, stream_id])

    def add_event(self, event_id: int, stream_id: int):
        event_id = raf.ir.const(event_id)
        stream_id = raf.ir.const(stream_id)
        return self.call("add_event", [event_id, stream_id])

    def wait_event(self, event_id: int, stream_id: int):
        event_id = raf.ir.const(event_id)
        stream_id = raf.ir.const(stream_id)
        return self.call("wait_event", [event_id, stream_id])

    def concatenate(self, x, axis: int):
        return self.call("concatenate", [x, self.const(axis)])

    def split(self, x, indices, axis):
        return self.call("split", [x, self.int_tuple_const(indices), self.int_const(axis)])

    def copy_inplace(self, dst_tensor, src_tensor, size: int, dst_offset: int, src_offset: int):
        size = self.int_const(size)
        dst_offset = self.int_const(dst_offset)
        src_offset = self.int_const(src_offset)
        return self.call("copy_inplace", [dst_tensor, src_tensor, size, dst_offset, src_offset])

    def ret(self, body: tvm.relay.Expr) -> tvm.relay.Expr:
        self.scope_builder.ret(body)
        return self.scope_builder.get()

@with_dialect(["tvm", "cuda"])
def test_tvm_layernorm_dx(shape, dtype):
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=shape, dtype=dtype)
            scale = extended_var("y", shape=(shape[2], ), dtype=dtype)
            dy = extended_var("dy", shape=shape, dtype=dtype)
            out = builder.call("layer_norm_dx", [x, scale, dy])
            return tvm.relay.Function([x, scale, dy], builder.ret(out))

        x_np = np.random.rand(*shape).astype(dtype)
        x = raf.array(x_np, device="cuda(0)")

        scale_np = np.random.rand(*(shape[2], )).astype(dtype)
        scale = raf.array(scale_np, device="cuda(0)")

        dy_np = np.random.rand(*shape).astype(dtype)
        dy = raf.array(dy_np, device="cuda(0)")

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        # tune the op
        vm_exec = get_vm_executor(test_mod, "cuda")
        out = vm_exec(x, scale, dy)

test_tvm_layernorm_dx((64, 197, 768), "float32")
