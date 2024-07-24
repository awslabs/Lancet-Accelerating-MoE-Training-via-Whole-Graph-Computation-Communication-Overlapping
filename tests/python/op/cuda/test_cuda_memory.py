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
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [(64, 128), (128, 256)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_tensor_fusion_and_defusion(shape, dtype):
    size = 1
    for axis in shape:
        size *= axis
    sizes = [size, size]
    tuple_shape = shape + shape
    indices = [len(shape), 2 * len(shape)]

    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=shape, dtype=dtype)
            x_1 = builder.make_tuple([x, x])
            a_1 = builder.call("fuse_tensor", [x_1])
            a_2 = builder.call("defuse_tensor", [a_1, raf.ir.const(sizes),
                                                 raf.ir.const(tuple_shape), raf.ir.const(indices)])
            a_3 = builder.get_tuple_item(a_2, 0)
            return tvm.relay.Function([x], builder.ret(a_3))

        def compare():
            builder = ANFBuilder()
            x = extended_var("x", shape=shape, dtype=dtype)
            return tvm.relay.Function([x], builder.ret(x))

        x = np.ones(shape, dtype=dtype)
        x = raf.array(x, device="cuda(0)")
        compare_mod = tvm.IRModule()
        compare_mod["main"] = compare()
        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(compare_mod, device)
        y_c = executor(x)
        executor = get_vm_executor(test_mod, device)
        y_t = executor(x)
        rtol = 1e-4 if dtype == "float32" else 4e-2
        atol = 1e-4 if dtype == "float32" else 4e-2
        # assert False
        check(y_t, y_c, rtol=rtol, atol=atol)

@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [((16, 4), (32, 4), (16, 4), (64, 4)), ((128, 1), (32, 2), (64, 4))])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("use_vm", [True, False])
def test_copy_inplace_small_to_large(shape, dtype, use_vm):
    dtype_bytes = int(dtype[-2:]) / 8
    src_tensor_shapes = list(shape)
    total_elements = 0
    sizes = []
    src_offsets = [0] * len(shape)
    dst_offsets = []
    for shape in src_tensor_shapes:
        dst_offsets.append(total_elements * dtype_bytes)
        elem_count = 1
        for dim in shape:
            elem_count *= dim
        sizes.append(elem_count * dtype_bytes)
        total_elements += elem_count

    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            buffer = extended_var("buffer", shape=(total_elements,), dtype=dtype)
            out_buffer = None
            src_tensors = [extended_var("src_{}".format(i), shape=src_tensor_shapes[i], dtype=dtype) for i in range(len(src_tensor_shapes))]
            for idx, tensor in enumerate(src_tensors):
                if out_buffer is None:
                    out_buffer = builder.copy_inplace(buffer, tensor, sizes[idx], dst_offsets[idx], src_offsets[idx])
                else:
                    out_buffer = builder.copy_inplace(out_buffer, tensor, sizes[idx], dst_offsets[idx], src_offsets[idx])
            return tvm.relay.Function([buffer, *src_tensors], builder.ret(out_buffer))

        def compare():
            builder = ANFBuilder()
            src_tensors = [extended_var("src_{}".format(i), shape=src_tensor_shapes[i], dtype=dtype) for i in range(len(src_tensor_shapes))]
            result = None
            for idx, tensor in enumerate(src_tensors):
                flattened_tensor = builder.reshape(tensor, [sizes[idx] / dtype_bytes,])
                if result is None:
                    result = flattened_tensor
                else:
                    result = builder.concatenate(builder.make_tuple([result, flattened_tensor]), 0)
            return tvm.relay.Function([*src_tensors], builder.ret(result))

        m_buffer = raf.zeros((total_elements,), dtype=dtype, device="cuda(0)")
        m_src_tensors = []
        for i in range(len(src_tensor_shapes)):
            np_src_tensor = np.random.rand(*src_tensor_shapes[i]).astype(dtype) * i
            m_src_tensors.append(raf.array(np_src_tensor, dtype=dtype, device="cuda(0)"))
        compare_mod = tvm.IRModule()
        compare_mod["main"] = compare()
        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        if(use_vm):
            executor = get_vm_executor(compare_mod, device)
            y_c = executor(*m_src_tensors)
            executor = get_vm_executor(test_mod, device)
            y_t = executor(m_buffer, *m_src_tensors)
        else:
            src_inputs = [get_ndarray_handle(arg) for arg in m_src_tensors]
            y_c = RunModel(compare_mod, src_inputs)
            y_c = ndarray(y_c)
            test_inputs = [get_ndarray_handle(m_buffer)] + src_inputs
            y_t = RunModel(test_mod, test_inputs)
            y_t = ndarray(y_t)

        rtol = 1e-4 if dtype == "float32" else 4e-2
        atol = 1e-4 if dtype == "float32" else 4e-2
        check(y_t, y_c, rtol=rtol, atol=atol)
        check(y_t, m_buffer, rtol=rtol, atol=atol)

@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("shape", [((16, 4), (32, 4), (16, 4), (64, 4)), ((128, 1), (32, 2), (64, 4))])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("use_vm", [False, True])
def test_copy_inplace_large_to_small(shape, dtype, use_vm):
    dtype_bytes = int(dtype[-2:]) / 8
    dst_tensor_shapes = list(shape)
    total_elements = 0
    sizes = []
    src_offsets = []
    dst_offsets = [0] * len(shape)
    split_indices = []
    for shape in dst_tensor_shapes:
        src_offsets.append(total_elements * dtype_bytes)
        elem_count = 1
        for dim in shape:
            elem_count *= dim
        sizes.append(elem_count * dtype_bytes)
        total_elements += elem_count
        split_indices.append(total_elements)

    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            buffer = extended_var("buffer", shape=(total_elements,), dtype=dtype)
            dst_tensors = [extended_var("src_{}".format(i), shape=dst_tensor_shapes[i], dtype=dtype) for i in range(len(dst_tensor_shapes))]
            result_dst_tensors = []
            for idx, tensor in enumerate(dst_tensors):
                out_dst_tensor = builder.copy_inplace(tensor, buffer, sizes[idx], dst_offsets[idx], src_offsets[idx])
                result_dst_tensors.append(out_dst_tensor)
            out = builder.make_tuple(result_dst_tensors)
            return tvm.relay.Function([buffer, *dst_tensors], builder.ret(out))

        def compare():
            builder = ANFBuilder()
            buffer = extended_var("buffer", shape=(total_elements,), dtype=dtype)
            result_tuple = builder.split(buffer, split_indices, 0)
            result_tensors = []
            for i in range(len(dst_tensor_shapes)):
                tensor_at_i = builder.get_tuple_item(result_tuple, i)
                result_tensors.append(builder.reshape(tensor_at_i, dst_tensor_shapes[i]))
            out = builder.make_tuple(result_tensors)
            return tvm.relay.Function([buffer], builder.ret(out))

        m_buffer = np.random.rand(total_elements).astype(dtype)
        m_buffer = raf.array(m_buffer, dtype=dtype, device="cuda(0)")
        m_dst_tensors = []
        for i in range(len(dst_tensor_shapes)):
            np_dst_tensor = np.zeros(dst_tensor_shapes[i], dtype=dtype)
            m_dst_tensors.append(raf.array(np_dst_tensor, dtype=dtype, device="cuda(0)"))
        compare_mod = tvm.IRModule()
        compare_mod["main"] = compare()
        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        out_tensors_c = []
        out_tensors_t = []
        if(use_vm):
            executor = get_vm_executor(compare_mod, device)
            y_c = executor(m_buffer)
            for out_tensor in y_c:
                out_tensors_c.append(out_tensor)
            executor = get_vm_executor(test_mod, device)
            y_t = executor(m_buffer, *m_dst_tensors)
            for out_tensor in y_t:
                out_tensors_t.append(out_tensor)
        else:
            y_c = RunModel(compare_mod, [get_ndarray_handle(m_buffer)])
            for out_tensor in y_c:
                out_tensors_c.append(ndarray(out_tensor))
            test_inputs = [get_ndarray_handle(m_buffer)] + [get_ndarray_handle(arg) for arg in m_dst_tensors]
            y_t = RunModel(test_mod, test_inputs)
            for out_tensor in y_t:
                out_tensors_t.append(ndarray(out_tensor))

        rtol = 1e-4 if dtype == "float32" else 4e-2
        atol = 1e-4 if dtype == "float32" else 4e-2
        for i in range(len(out_tensors_c)):
            check(out_tensors_t[i], out_tensors_c[i], rtol=rtol, atol=atol)
            check(out_tensors_t[i], m_dst_tensors[i], rtol=rtol, atol=atol)

if __name__ == "__main__":
    pytest.main([__file__])
