# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-locals,too-many-arguments,protected-access,no-self-use, line-too-long, superfluous-parens
from typing import Dict, List
from collections import defaultdict
import pytest
import numpy as np
import tvm

import raf
from raf.testing import run_vm_model
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


@with_dialect(["cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_encode(S, M, E, dtype):
    C = int(np.ceil(S / E))
    input_shape = (S, M)
    gate_shape = (S, E)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=input_shape, dtype=dtype)
            gate = extended_var("gate", shape=gate_shape, dtype=dtype)
            used_capacity = extended_var("used_capacity", shape=(E,), dtype="int32")
            out = builder.call("moe_encode", [x, gate, used_capacity, raf.ir.const(1.0)])
            return tvm.relay.Function([x, gate, used_capacity], builder.ret(out))

        x_np = np.random.rand(*input_shape).astype(dtype)
        x = raf.array(x_np, device="cuda(0)")

        gate_np = np.random.rand(*gate_shape).astype(dtype)
        gate = raf.array(gate_np, device="cuda(0)")

        used_capacity_np = np.zeros((E,), dtype="int32")
        used_capacity = raf.array(used_capacity_np, device="cuda(0)")

        def ref():
            indices_s = np.argmax(gate_np, axis=1)
            masks_se = np.eye(E)[indices_s]
            masked_gates_se = gate_np * masks_se
            masked_gates_s = np.sum(masked_gates_se, axis=1)
            locations_cumsum = np.cumsum(masks_se, axis=0).astype("int32")
            locations1 = locations_cumsum - 1
            masked_locations1 = locations1 * masks_se
            locations_s_float = np.sum(masked_locations1, axis=1)
            locations_s = locations_s_float.astype("int32")
            dispatched_input = np.zeros((E, C, M), dtype="float32")
            for i in range(S):
                if locations_s[i] < C and indices_s[i] < E:
                    dispatched_input[indices_s[i], locations_s[i], :] = x_np[i, :]
            out_capused1_e = np.zeros((E,), dtype="int32")
            elements_per_expert = np.zeros((E,), dtype="uint64")
            for i in range(S):
                loc = min(locations_s[i], C - 1)
                out_capused1_e[indices_s[i]] = max(out_capused1_e[indices_s[i]], loc + 1)
                elements_per_expert[indices_s[i]] = (out_capused1_e[indices_s[i]] - used_capacity_np[indices_s[i]]) * M
                if locations_s[i] >= (C - used_capacity_np[indices_s[i]]):
                    indices_s[i] = -1
            indices_locations = np.concatenate([indices_s[np.newaxis, :], locations_s[np.newaxis, :]], axis=0)
            return masked_gates_s, indices_locations, out_capused1_e, elements_per_expert, dispatched_input

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        rgate_s, rindices_locations, rcapused1_e, relements_per_expert, rdispatched_input = ref()
        tgates_s, tindices_locations, tcapused1_e, telements_per_expert, tdispatched_input = executor(x, gate, used_capacity)
        rtol = 1e-4
        atol = 1e-4
        check(tgates_s, rgate_s, rtol=rtol, atol=atol)
        check(tindices_locations, rindices_locations, rtol=rtol, atol=atol)
        check(tcapused1_e, rcapused1_e, rtol=rtol, atol=atol)
        check(telements_per_expert, relements_per_expert, rtol=rtol, atol=atol)
        check(tdispatched_input, rdispatched_input, rtol=rtol, atol=atol)
        print("OK")
        import code
        code.interact(local=locals())

@with_dialect(["cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_encode_batch_prioritized(S, M, E, dtype):
    C = int(np.ceil(S / E))
    input_shape = (S, M)
    gate_shape = (S, E)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=input_shape, dtype=dtype)
            gate = extended_var("gate", shape=gate_shape, dtype=dtype)
            n_partitions = raf.ir.const(1)
            partition_id = raf.ir.const(0)
            out = builder.call("moe_encode_batch_prioritized", [x, gate, n_partitions, partition_id, raf.ir.const(1.0)])
            return tvm.relay.Function([x, gate], builder.ret(out))

        x_np = np.random.rand(*input_shape).astype(dtype)
        x = raf.array(x_np, device="cuda(0)")

        gate_np = np.random.rand(*gate_shape).astype(dtype)
        gate = raf.array(gate_np, device="cuda(0)")

        def ref():
            indices_s = np.argmax(gate_np, axis=1)
            masks_se = np.eye(E)[indices_s]
            masked_gates_se = gate_np * masks_se
            masked_gates_s = np.sum(masked_gates_se, axis=1)
            # argsort by gate score
            sorted_indices = np.argsort(-masked_gates_s)
            dropped = [False] * S
            per_expert_counts = defaultdict(int)
            for i in range(S):
                idx = sorted_indices[i]
                expert_idx = indices_s[idx]
                if per_expert_counts[expert_idx] < C:
                    per_expert_counts[expert_idx] += 1
                else:
                    # dropped
                    dropped[idx] = True
            # now we have dropped indices, calculate the locations
            locations_s = np.zeros((S,), dtype="int32")
            per_expert_counts_ordered = defaultdict(int)
            for i in range(S):
                expert_idx = indices_s[i]
                if not dropped[i]:
                    locations_s[i] = per_expert_counts_ordered[expert_idx]
                    per_expert_counts_ordered[expert_idx] += 1
                else:
                    locations_s[i] = -1
                    indices_s[i] = -1
            dispatched_input = np.zeros((E, C, M), dtype="float32")
            for i in range(S):
                if locations_s[i] < C and locations_s[i] >= 0 and indices_s[i] < E and indices_s[i] >= 0:
                    dispatched_input[indices_s[i], locations_s[i], :] = x_np[i, :]
            out_capused1_e = np.zeros((E,), dtype="int32")
            elements_per_expert = np.zeros((E,), dtype="uint64")
            for i in range(E):
                out_capused1_e[i] = per_expert_counts_ordered[i]
                elements_per_expert[i] = per_expert_counts_ordered[i] * M
            indices_locations = np.concatenate([indices_s[np.newaxis, :], locations_s[np.newaxis, :]], axis=0)
            return masked_gates_s, indices_locations, out_capused1_e, elements_per_expert, dispatched_input

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        rgate_s, rindices_locations, rcapused1_e, relements_per_expert, rdispatched_input = ref()
        tgates_s, tindices_locations, tcapused1_e, telements_per_expert, tdispatched_input = executor(x, gate)
        rtol = 1e-4
        atol = 1e-4
        check(tgates_s, rgate_s, rtol=rtol, atol=atol)
        check(tindices_locations, rindices_locations, rtol=rtol, atol=atol)
        check(tcapused1_e, rcapused1_e, rtol=rtol, atol=atol)
        check(telements_per_expert, relements_per_expert, rtol=rtol, atol=atol)
        check(tdispatched_input, rdispatched_input, rtol=rtol, atol=atol)
        print("OK")

@with_dialect(["cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_encode_batch_prioritized_partitioned(S, M, E, n_partitions, partition_id, dtype):
    np.random.seed(0)
    C = int(np.ceil(S / E))
    orig_S = S
    S = S // n_partitions
    input_shape = (orig_S, M)
    gate_shape = (orig_S, E)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=input_shape, dtype=dtype)
            gate = extended_var("gate", shape=gate_shape, dtype=dtype)
            np = raf.ir.const(n_partitions)
            pid = raf.ir.const(partition_id)
            out = builder.call("moe_encode_batch_prioritized", [x, gate, np, pid, raf.ir.const(float(n_partitions))])
            return tvm.relay.Function([x, gate], builder.ret(out))

        x_np = np.random.rand(*input_shape).astype(dtype)
        x = raf.array(x_np, device="cuda(0)")

        gate_np = np.random.rand(*gate_shape).astype(dtype)
        gate = raf.array(gate_np, device="cuda(0)")

        def ref():
            indices_s = np.argmax(gate_np, axis=1)
            masks_se = np.eye(E)[indices_s]
            masked_gates_se = gate_np * masks_se
            masked_gates_s = np.sum(masked_gates_se, axis=1)
            # argsort by gate score
            sorted_indices = np.argsort(-masked_gates_s)
            dropped = [False] * orig_S
            per_expert_counts = defaultdict(int)
            for i in range(orig_S):
                idx = sorted_indices[i]
                expert_idx = indices_s[idx]
                if per_expert_counts[expert_idx] < C:
                    per_expert_counts[expert_idx] += 1
                else:
                    # dropped
                    dropped[idx] = True
            # now we have dropped indices, calculate the locations
            locations_s = np.zeros((S,), dtype="int32")
            per_expert_counts_ordered = defaultdict(int)
            s_offset = S * partition_id
            for i in range(S):
                expert_idx = indices_s[s_offset + i]
                if not dropped[s_offset + i]:
                    locations_s[i] = per_expert_counts_ordered[expert_idx]
                    per_expert_counts_ordered[expert_idx] += 1
                else:
                    locations_s[i] = -1
                    indices_s[s_offset + i] = -1
            dispatched_input = np.zeros((E, C, M), dtype="float32")
            for i in range(S):
                if locations_s[i] < C and locations_s[i] >= 0 and indices_s[s_offset + i] < E and indices_s[s_offset + i] >= 0:
                    dispatched_input[indices_s[s_offset + i], locations_s[i], :] = x_np[s_offset + i, :]
            out_capused1_e = np.zeros((E,), dtype="int32")
            elements_per_expert = np.zeros((E,), dtype="uint64")
            for i in range(E):
                out_capused1_e[i] = per_expert_counts_ordered[i]
                elements_per_expert[i] = per_expert_counts_ordered[i] * M
            indices_locations = np.concatenate([indices_s[np.newaxis, s_offset:s_offset + S], locations_s[np.newaxis, :]], axis=0)
            return masked_gates_s[s_offset: s_offset + S], indices_locations, out_capused1_e, elements_per_expert, dispatched_input

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        rgate_s, rindices_locations, rcapused1_e, relements_per_expert, rdispatched_input = ref()

        tgates_s, tindices_locations, tcapused1_e, telements_per_expert, tdispatched_input = executor(x, gate)
        rtol = 1e-4
        atol = 1e-4
        check(tgates_s, rgate_s, rtol=rtol, atol=atol)
        check(tindices_locations, rindices_locations, rtol=rtol, atol=atol)
        check(tcapused1_e, rcapused1_e, rtol=rtol, atol=atol)
        check(telements_per_expert, relements_per_expert, rtol=rtol, atol=atol)
        check(tdispatched_input, rdispatched_input, rtol=rtol, atol=atol)
        print("OK")


@with_dialect(["cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_partitioned_moe_encode(S, M, E, dtype):
    C = int(np.ceil(2 * S / E))
    input_shape = (S, M)
    gate_shape = (S, E)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=input_shape, dtype=dtype)
            x1 = extended_var("x1", shape=input_shape, dtype=dtype)
            gate = extended_var("gate", shape=gate_shape, dtype=dtype)
            gate1 = extended_var("gate1", shape=gate_shape, dtype=dtype)
            used_capacity = extended_var("used_capacity", shape=(E,), dtype="int32")
            out = builder.call("moe_encode", [x, gate, used_capacity, raf.ir.const(2.0)])
            used_capacity1 = builder.get_tuple_item(out, 2)
            out1 = builder.call("moe_encode", [x1, gate1, used_capacity1, raf.ir.const(2.0)])
            indices = builder.make_tuple([builder.get_tuple_item(out, 1), builder.get_tuple_item(out1, 1)])
            return tvm.relay.Function([x, x1, gate, gate1, used_capacity], builder.ret(indices))

        x_np = np.random.rand(*input_shape).astype(dtype)
        x = raf.array(x_np, device="cuda(0)")

        x1_np = np.random.rand(*input_shape).astype(dtype)
        x1 = raf.array(x1_np, device="cuda(0)")

        gate_np = np.random.rand(*gate_shape).astype(dtype)
        gate = raf.array(gate_np, device="cuda(0)")

        gate1_np = np.random.rand(*gate_shape).astype(dtype)
        gate1 = raf.array(gate1_np, device="cuda(0)")

        used_capacity_np = np.zeros((E,), dtype="int32")
        used_capacity = raf.array(used_capacity_np, device="cuda(0)")

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        tinds_locs, tinds_locs1 = executor(x, x1, gate, gate1, used_capacity)
        concated = np.concatenate([tinds_locs.numpy()[0,:], tinds_locs1.numpy()[0,:]], axis=0)
        exp_cnt = defaultdict(int)
        for i in range(S * 2):
            exp_cnt[concated[i]] += 1
        for i in range(E):
            assert exp_cnt[i] <= C


@with_dialect(["cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("n_partitions", [2, 4])
def test_partitioned_moe_encode_merge_masks_and_data(n_partitions):

    S_before_partition = 1024
    M = 512
    E = 2

    class TestModel(raf.Model):
        def build(self, n_partitions, device):
            self.n_partitions = n_partitions
            self.device = device

        @raf.model.trace
        def forward(self, x, gate_w):
            if self.n_partitions == 1:
                partitioned_xs = [x]
            else:
                partitioned_xs = raf.split(x, self.n_partitions, axis=0)
            # run dispatch
            capused = raf.zeros(shape=(E,), dtype="int32", device=self.device)
            gate_ss = []
            indices_locationss = []
            dispatched_inputs = []
            for i in range(self.n_partitions):
                xi = partitioned_xs[i]
                # gate
                gate_se = raf.matmul(xi, gate_w)
                # dispatch
                encode_out = raf.moe_encode(xi, gate_se, capused, self.n_partitions)
                gate_s = encode_out[0]
                indices_locations = encode_out[1]
                capused = encode_out[2]
                dispatched_input = encode_out[4]
                gate_ss.append(gate_s)
                indices_locationss.append(indices_locations)
                dispatched_inputs.append(dispatched_input)
            if self.n_partitions > 1:
                # redispatch
                redispatched_indices_locations = raf.moe_merge_masks(indices_locationss, E)
                redispatched_data = raf.moe_redispatch(dispatched_inputs, indices_locationss)
                # also calc gates
                redispatched_gates = raf.concatenate(gate_ss, axis=0)
                return redispatched_data, redispatched_gates, redispatched_indices_locations
            else:
                return dispatched_inputs[0], gate_ss[0], indices_locationss[0]

    device = "cuda(0)"
    partitioned_model = TestModel(n_partitions, device)
    reference_model = TestModel(1, device)

    partitioned_model.to(device=device)
    reference_model.to(device=device)

    x = np.random.randn(S_before_partition, M).astype("float32")
    gate_w = np.random.randn(M, E).astype("float32")

    x = raf.array(x, device=device)
    gate_w = raf.array(gate_w, device=device)

    partitioned_out = run_vm_model(partitioned_model, device, [x, gate_w])
    pdata = partitioned_out[0].numpy()
    pgate_s = partitioned_out[1].numpy()
    pindices_locations = partitioned_out[2].numpy()
    reference_out = run_vm_model(reference_model, device, [x, gate_w])
    rdata = reference_out[0].numpy()
    rgate_s = reference_out[1].numpy()
    rindices_locations = reference_out[2].numpy()

    capacity_per_expert = defaultdict(int)
    for i in range(S_before_partition):
        if rindices_locations[0, i] != -1:
            capacity_per_expert[rindices_locations[0, i]] += 1
    # zero out unused capacity
    for i in range(S_before_partition):
        if pindices_locations[0, i] == -1:
            pgate_s[i] = 0
            pindices_locations[1, i] = 0
        if rindices_locations[0, i] == -1:
            rgate_s[i] = 0
            rindices_locations[1, i] = 0
    for i in range(E):
        pdata[i, capacity_per_expert[i]:, :] = 0
        rdata[i, capacity_per_expert[i]:, :] = 0
    check(pdata, rdata, atol=1e-5, rtol=1e-5)
    check(pgate_s, rgate_s, atol=1e-5, rtol=1e-5)
    check(pindices_locations, rindices_locations, atol=1e-5, rtol=1e-5)

@with_dialect(["cudnn", "tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_partitioned_moe_encode_decode(S, M, E, dtype):
    C = int(np.ceil(S / E))
    S = S // 2
    input_shape = (S, M)
    gate_shape = (S, E)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            x = extended_var("x", shape=input_shape, dtype=dtype)
            x1 = extended_var("x1", shape=input_shape, dtype=dtype)
            gate = extended_var("gate", shape=gate_shape, dtype=dtype)
            gate1 = extended_var("gate1", shape=gate_shape, dtype=dtype)
            expert_weight0 = extended_var("expert_weight0", shape=(M, 4*M), dtype=dtype)
            expert_weight1 = extended_var("expert_weight1", shape=(4*M, M), dtype=dtype)
            used_capacity = extended_var("used_capacity", shape=(E,), dtype="int32")
            out = builder.call("moe_encode", [x, gate, used_capacity, raf.ir.const(2.0)])
            gate_s0 = builder.get_tuple_item(out, 0)
            indices_locations_s0 = builder.get_tuple_item(out, 1)
            used_capacity1 = builder.get_tuple_item(out, 2)
            dispatched0 = builder.get_tuple_item(out, 4)
            out1 = builder.call("moe_encode", [x1, gate1, used_capacity1, raf.ir.const(2.0)])
            gate_s1 = builder.get_tuple_item(out1, 0)
            indices_locations_s1 = builder.get_tuple_item(out1, 1)
            dispatched1 = builder.get_tuple_item(out1, 4)
            exp_inp0 = builder.reshape(dispatched0, [E * C, M])
            exp_inp1 = builder.reshape(dispatched1, [E * C, M])
            exp_interm = builder.call("matmul", [exp_inp0, expert_weight0])
            exp_interm1 = builder.call("matmul", [exp_inp1, expert_weight0])
            exp_out0 = builder.call("matmul", [exp_interm, expert_weight1])
            exp_out1 = builder.call("matmul", [exp_interm1, expert_weight1])
            exp_out_reshaped0 = builder.reshape(exp_out0, [E, C, M])
            exp_out_reshaped1 = builder.reshape(exp_out1, [E, C, M])
            # data, gate_s, indices, locations
            recon0 = builder.call("moe_decode", [exp_out_reshaped0, gate_s0, indices_locations_s0])
            recon1 = builder.call("moe_decode", [exp_out_reshaped1, gate_s1, indices_locations_s1])
            out = builder.make_tuple([recon0, recon1])
            return tvm.relay.Function([x, x1, gate, gate1, expert_weight0, expert_weight1, used_capacity], builder.ret(out))

        x_np = np.random.rand(*input_shape).astype(dtype)
        x = raf.array(x_np, device="cuda(0)")

        x1_np = np.random.rand(*input_shape).astype(dtype)
        x1 = raf.array(x1_np, device="cuda(0)")

        gate_np = np.random.rand(*gate_shape).astype(dtype)
        gate = raf.array(gate_np, device="cuda(0)")

        gate1_np = np.random.rand(*gate_shape).astype(dtype)
        gate1 = raf.array(gate1_np, device="cuda(0)")

        used_capacity_np = np.zeros((E,), dtype="int32")
        used_capacity = raf.array(used_capacity_np, device="cuda(0)")

        exp_weight0_np = np.random.rand(M, 4*M).astype(dtype)
        exp_weight0 = raf.array(exp_weight0_np, device="cuda(0)")

        exp_weight1_np = np.random.rand(4*M, M).astype(dtype)
        exp_weight1 = raf.array(exp_weight1_np, device="cuda(0)")

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        def ref():
            concated_gate = np.concatenate([gate_np, gate1_np], axis=0)
            concated_x = np.concatenate([x_np, x1_np], axis=0)
            indices_s = np.argmax(concated_gate, axis=1)
            masks_se = np.eye(E)[indices_s]
            masked_gates_se = concated_gate * masks_se
            masked_gates_s = np.sum(masked_gates_se, axis=1)
            locations_cumsum = np.cumsum(masks_se, axis=0).astype("int32")
            locations1 = locations_cumsum - 1
            masked_locations1 = locations1 * masks_se
            locations_s_float = np.sum(masked_locations1, axis=1)
            locations_s = locations_s_float.astype("int32")
            dispatched_input = np.zeros((E, C, M), dtype="float32")
            for i in range(2 * S):
                if locations_s[i] < C and indices_s[i] < E:
                    dispatched_input[indices_s[i], locations_s[i], :] = concated_x[i, :]
            out_capused1_e = np.zeros((E,), dtype="int32")
            for i in range(2 * S):
                loc = min(locations_s[i], C - 1)
                out_capused1_e[indices_s[i]] = max(out_capused1_e[indices_s[i]], loc + 1)
                if locations_s[i] >= (C - used_capacity_np[indices_s[i]]):
                    indices_s[i] = -1
            dispatched_input = np.reshape(dispatched_input, (E * C, M))
            dispatched_exp_int = np.matmul(dispatched_input, exp_weight0_np)
            dispatched_exp = np.matmul(dispatched_exp_int, exp_weight1_np)
            dispatched_reshaped = np.reshape(dispatched_exp, (E, C, M))
            rout = np.zeros((2 * S, M), dtype=dtype)
            for i in range(2 * S):
                loc = locations_s[i]
                idx = indices_s[i]
                if idx < E and loc < C:
                    rout[i] = dispatched_reshaped[idx, loc] * masked_gates_s[i]
            return rout

        recon0, recon1 = executor(x, x1, gate, gate1, exp_weight0, exp_weight1, used_capacity)
        concated = np.concatenate([recon0.numpy(), recon1.numpy()], axis=0)
        ref_out = ref()
        check(concated, ref_out, rtol=1e-5, atol=1e-5)


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_encode_dx(S, M, E, dtype):
    C = (S + E - 1) // E
    dy_shape = (E, C, M)
    mask_shape = (2, S)
    with Device("cuda(0)"):
        dy_np = np.random.rand(*dy_shape).astype(dtype)
        dy = raf.array(dy_np, device="cuda(0)")

        indices_np = np.random.randint(-1, E, size=(S, ), dtype=np.int32)

        per_expert_tokens = defaultdict(int)
        locations_np = np.zeros_like(indices_np)
        for i in range(S):
            if indices_np[i] >= 0:
                curr_tokens = per_expert_tokens[indices_np[i]]
                if curr_tokens < C:
                    locations_np[i] = curr_tokens
                    per_expert_tokens[indices_np[i]] += 1
                else:
                    indices_np[i] = -1
                    locations_np[i] = -1

        ind_locs_np = np.concatenate([np.expand_dims(indices_np, 0), np.expand_dims(locations_np, 0)], axis=0)
        ind_locs = raf.array(ind_locs_np, device="cuda(0)")
        # with np.load('./moe_encode_grad_inouts.npz') as data:
        #     dy_np = data['data_grad']
        #     ind_locs_np = data['ind_loc']
        #     dx_np = data['dx']

        # with open("./device_0_inputs.bin", "rb") as f:
        #     param_bytes = f.read()
        # raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        # dy_raf = raf_inputs["input_0"]
        # ind_locs_raf = raf_inputs["input_1"]
        # dy_raf_np = dy_raf.numpy()
        # ind_locs_raf_np = ind_locs_raf.numpy()

        dy = raf.array(dy_np, device="cuda(0)")
        ind_locs = raf.array(ind_locs_np, device="cuda(0)", dtype="int32")

        dy_shape = tuple(dy_np.shape)
        mask_shape = tuple(ind_locs_np.shape)

        def test():
            builder = ANFBuilder()
            dy = extended_var("dy", shape=dy_shape, dtype=dtype)
            indices_locations = extended_var("indices_locations", shape=mask_shape, dtype="int32")
            out = builder.call("moe_encode_dx", [dy, indices_locations])
            return tvm.relay.Function([dy, indices_locations], builder.ret(out))

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        out = executor(dy, ind_locs)

        rout = out.numpy()

        # indices_np = ind_locs_np[0, :]
        # locations_np = ind_locs_np[1, :]
        ref_out = np.zeros((S, M), dtype=dtype)
        # ref_out = np.zeros_like(dx_np, dtype=dtype)
        for i in range(S):
            idx_e = indices_np[i]
            idx_c = locations_np[i]
            if idx_e >= 0 and idx_c >= 0:
                ref_out[i, :] = dy_np[idx_e, idx_c, :]

        check(rout, ref_out, rtol=1e-4, atol=1e-4)

@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_encode_dg(S, E, dtype):
    dy_shape = (S,)
    mask_shape = (2, S)
    with Device("cuda(0)"):
        dy_np = np.random.rand(*dy_shape).astype(dtype)

        ind_locs_np = np.random.randint(-1, E, size=mask_shape, dtype=np.int32)

        # with np.load('./moe_encode_grad_inouts.npz') as data:
        #     dy_np = data['gate_grad']
        #     ind_locs_np = data['ind_loc']
        #     dg_np = data['dg']

        dy = raf.array(dy_np, device="cuda(0)")
        ind_locs = raf.array(ind_locs_np, device="cuda(0)", dtype="int32")

        dy_shape = tuple(dy_np.shape)
        mask_shape = tuple(ind_locs_np.shape)
        n_experts = E
        # n_experts = dg_np.shape[1]
        def test():
            builder = ANFBuilder()
            dy_var = extended_var("dy", shape=dy_shape, dtype=dtype)
            indices_locations_var = extended_var("indiceslocations", shape=mask_shape, dtype="int32")
            n_experts_var = builder.int_const(n_experts)
            out = builder.call("moe_encode_dg", [dy_var, indices_locations_var, n_experts_var])
            return tvm.relay.Function([dy_var, indices_locations_var], builder.ret(out))

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        out = executor(dy, ind_locs)

    ref_out = np.zeros((S, E), dtype=dtype)
    # ref_out = np.zeros_like(dg_np, dtype=dtype)
    for i in range(S):
        idx = ind_locs_np[0, i]
        if idx >= 0:
            ref_out[i, idx] = dy_np[i]

    rout = out.numpy()

    check(rout, ref_out, rtol=1e-5, atol=1e-5)

def test_moe_redispatch():
    with Device("cuda(0)"):
        in_dev_0_a2a_inputs = []
        in_dev_0_a2a_send_cnts = []
        in_dev_0_a2a_outputs = []
        in_dev_0_a2a_recv_cnts = []
        out_dev_0_a2a_inputs = []
        out_dev_0_a2a_send_cnts = []
        out_dev_0_a2a_outputs = []
        out_dev_0_a2a_recv_cnts = []
        in_dev_1_a2a_inputs = []
        in_dev_1_a2a_send_cnts = []
        in_dev_1_a2a_outputs = []
        in_dev_1_a2a_recv_cnts = []
        out_dev_1_a2a_inputs = []
        out_dev_1_a2a_send_cnts = []
        out_dev_1_a2a_outputs = []
        out_dev_1_a2a_recv_cnts = []
        for part_id in range(2):
            with open(f"/models/opt_ios/device_0_raf.op.nccl._all_to_allv_inouts_{part_id}.bin", "rb") as f:
                param_bytes = f.read()
            raf_inputs = tvm.runtime.load_param_dict(param_bytes)
            in_dev_0_a2a_inputs.append(raf_inputs["input_0_0"].numpy())
            in_dev_0_a2a_send_cnts.append(raf_inputs["input_1_0"].numpy())
            in_dev_0_a2a_outputs.append(raf_inputs["output_0"].numpy())
            in_dev_0_a2a_recv_cnts.append(raf_inputs["output_1"].numpy())

            with open(f"/models/opt_ios/device_0_raf.op.nccl._all_to_allv_inouts_{part_id + 2}.bin", "rb") as f:
                param_bytes = f.read()
            raf_inputs = tvm.runtime.load_param_dict(param_bytes)
            out_dev_0_a2a_inputs.append(raf_inputs["input_0_0"].numpy().reshape(-1, 2048, 768))
            out_dev_0_a2a_send_cnts.append(raf_inputs["input_1_0"].numpy())
            out_dev_0_a2a_outputs.append(raf_inputs["output_0"].numpy().reshape(-1, 2048, 768))
            out_dev_0_a2a_recv_cnts.append(raf_inputs["output_1"].numpy())

            with open(f"/models/opt_ios/device_1_raf.op.nccl._all_to_allv_inouts_{part_id}.bin", "rb") as f:
                param_bytes = f.read()
            raf_inputs = tvm.runtime.load_param_dict(param_bytes)
            in_dev_1_a2a_inputs.append(raf_inputs["input_0_0"].numpy())
            in_dev_1_a2a_send_cnts.append(raf_inputs["input_1_0"].numpy())
            in_dev_1_a2a_outputs.append(raf_inputs["output_0"].numpy())
            in_dev_1_a2a_recv_cnts.append(raf_inputs["output_1"].numpy())
            
            with open(f"/models/opt_ios/device_1_raf.op.nccl._all_to_allv_inouts_{part_id + 2}.bin", "rb") as f:
                param_bytes = f.read()
            raf_inputs = tvm.runtime.load_param_dict(param_bytes)
            out_dev_1_a2a_inputs.append(raf_inputs["input_0_0"].numpy().reshape(-1, 2048, 768))
            out_dev_1_a2a_send_cnts.append(raf_inputs["input_1_0"].numpy())
            out_dev_1_a2a_outputs.append(raf_inputs["output_0"].numpy().reshape(-1, 2048, 768))
            out_dev_1_a2a_recv_cnts.append(raf_inputs["output_1"].numpy())

        def simulate_a2av(dev0_input, dev1_input, dev0_sendcnt, dev1_sendcnt):
            dev0_out = np.zeros(dev0_input.shape, dtype=dev0_input.dtype)
            dev1_out = np.zeros(dev1_input.shape, dtype=dev1_input.dtype)
            for i in range(dev0_sendcnt.shape[0]):
                if i < (dev0_sendcnt.shape[0] // 2):
                    dev0_out[i, :dev0_sendcnt[i]] = dev0_input[i, :dev0_sendcnt[i]]
                    if i == 0:
                        print(f"Assigning dev0_out[0, :{dev0_sendcnt[i]}] with dev0_input[0, :{dev0_sendcnt[i]}]")
                    dev0_out[i + (dev0_sendcnt.shape[0] // 2), :dev1_sendcnt[i]] = dev1_input[i, :dev1_sendcnt[i]]
                else:
                    dev1_out[i - (dev0_sendcnt.shape[0] // 2), :dev0_sendcnt[i]] = dev0_input[i, :dev0_sendcnt[i]]
                    dev1_out[i, :dev1_sendcnt[i]] = dev1_input[i, :dev1_sendcnt[i]]
            return dev0_out, dev1_out

        in_simulated_dev_0_outputs = []
        in_simulated_dev_1_outputs = []
        out_simulated_dev_0_outputs = []
        out_simulated_dev_1_outputs = []
        for part_id in range(2):
            dev0_out, dev1_out = simulate_a2av(in_dev_0_a2a_inputs[part_id], in_dev_1_a2a_inputs[part_id], in_dev_0_a2a_send_cnts[part_id] // 768, in_dev_1_a2a_send_cnts[part_id] // 768)
            in_simulated_dev_0_outputs.append(dev0_out)
            in_simulated_dev_1_outputs.append(dev1_out)
            dev0_out, dev1_out = simulate_a2av(out_dev_0_a2a_inputs[part_id].reshape(-1, 2048, 768), out_dev_1_a2a_inputs[part_id].reshape(-1, 2048, 768), out_dev_0_a2a_send_cnts[part_id] // 768, out_dev_1_a2a_send_cnts[part_id] // 768)
            out_simulated_dev_0_outputs.append(dev0_out)
            out_simulated_dev_1_outputs.append(dev1_out)

        for part_id in range(2):
            for expert_id in range(4):
                is_identical = np.allclose(in_simulated_dev_0_outputs[part_id][expert_id, :int(in_dev_0_a2a_recv_cnts[part_id][expert_id] // 768)], in_dev_0_a2a_outputs[part_id][expert_id, :int(in_dev_0_a2a_recv_cnts[part_id][expert_id] // 768)], atol=1e-4, rtol=1e-4)
                print(f"in_simulated_dev_0_outputs[{part_id}][{expert_id}] == in_dev_0_a2a_outputs[{part_id}][{expert_id}]: ", is_identical)
                is_identical = np.allclose(in_simulated_dev_1_outputs[part_id][expert_id, :int(in_dev_1_a2a_recv_cnts[part_id][expert_id] // 768)], in_dev_1_a2a_outputs[part_id][expert_id, :int(in_dev_1_a2a_recv_cnts[part_id][expert_id] // 768)], atol=1e-4, rtol=1e-4)
                print(f"in_simulated_dev_1_outputs[{part_id}][{expert_id}] == in_dev_1_a2a_outputs[{part_id}][{expert_id}]: ", is_identical)
                is_identical = np.allclose(out_simulated_dev_0_outputs[part_id][expert_id, :int(out_dev_0_a2a_recv_cnts[part_id][expert_id] // 768)], out_dev_0_a2a_outputs[part_id][expert_id, :int(out_dev_0_a2a_recv_cnts[part_id][expert_id] // 768)], atol=1e-4, rtol=1e-4)
                print(f"out_simulated_dev_0_outputs[{part_id}][{expert_id}] == out_dev_0_a2a_outputs[{part_id}][{expert_id}]: ", is_identical)
                is_identical = np.allclose(out_simulated_dev_1_outputs[part_id][expert_id, :int(out_dev_1_a2a_recv_cnts[part_id][expert_id] // 768)], out_dev_1_a2a_outputs[part_id][expert_id, :int(out_dev_1_a2a_recv_cnts[part_id][expert_id] // 768)], atol=1e-4, rtol=1e-4)
                print(f"out_simulated_dev_1_outputs[{part_id}][{expert_id}] == out_dev_1_a2a_outputs[{part_id}][{expert_id}]: ", is_identical)

        for part_id in range(2):
            dev0_first_a2a_output = in_dev_0_a2a_outputs[part_id] # [E, C, M]
            recv_cnts = in_dev_0_a2a_recv_cnts[part_id] // 768
            for expert_id in range(2):
                masked_le0 = dev0_first_a2a_output[expert_id, :recv_cnts[expert_id]]
                masked_le1 = dev0_first_a2a_output[expert_id + 2, :recv_cnts[expert_id + 2]]
                max_diff = np.max(np.abs(masked_le0 - masked_le1))
                print(f"dev0_first_a2a_output max_diff: part {part_id}, expert: {expert_id}: ", max_diff)

        for part_id in range(2):
            dev1_first_a2a_output = in_dev_1_a2a_outputs[part_id] # [E, C, M]
            recv_cnts = in_dev_1_a2a_recv_cnts[part_id] // 768
            for expert_id in range(2):
                masked_le0 = dev1_first_a2a_output[expert_id, :recv_cnts[expert_id]]
                masked_le1 = dev1_first_a2a_output[expert_id + 2, :recv_cnts[expert_id + 2]]
                max_diff = np.max(np.abs(masked_le0 - masked_le1))
                print(f"dev1_first_a2a_output max_diff: part {part_id}, expert: {expert_id}: ", max_diff)

        for part_id in range(2):
            dev0_second_a2a_input = out_dev_0_a2a_inputs[part_id] # [E, C, M]
            send_cnts = out_dev_0_a2a_send_cnts[part_id] // 768
            for expert_id in range(2):
                masked_le0 = dev0_second_a2a_input[expert_id, :send_cnts[expert_id]]
                masked_le1 = dev0_second_a2a_input[expert_id + 2, :send_cnts[expert_id + 2]]
                max_diff = np.max(np.abs(masked_le0 - masked_le1))
                print(f"dev0_second_a2a_input max_diff: part {part_id}, expert: {expert_id}: ", max_diff)

        for part_id in range(2):
            dev1_second_a2a_input = out_dev_1_a2a_inputs[part_id]
            send_cnts = out_dev_1_a2a_send_cnts[part_id] // 768
            for expert_id in range(2):
                masked_le0 = dev1_second_a2a_input[expert_id, :send_cnts[expert_id]]
                masked_le1 = dev1_second_a2a_input[expert_id + 2, :send_cnts[expert_id + 2]]
                max_diff = np.max(np.abs(masked_le0 - masked_le1))
                print(f"dev1_second_a2a_input max_diff: part {part_id}, expert: {expert_id}: ", max_diff)
        
        for part_id in range(2):
            dev0_second_a2a_output = out_dev_0_a2a_outputs[part_id] # [E, C, M]
            dev1_second_a2a_output = out_dev_1_a2a_outputs[part_id] # [E, C, M]
            recv_cnts = out_dev_0_a2a_recv_cnts[part_id] // 768
            for expert_id in range(4):
                masked_dev0 = dev0_second_a2a_output[expert_id, :recv_cnts[expert_id]]
                masked_dev1 = dev1_second_a2a_output[expert_id, :recv_cnts[expert_id]]
                max_diff = np.max(np.abs(masked_dev0 - masked_dev1))
                print(f"dev0_second_a2a_output max_diff: part {part_id}, expert: {expert_id}: ", max_diff)

        with open("/models/opt_ios/device_0_raf.op.cublas.sparse_expert_matmul_nt_inouts_0.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_in_exp1_data = raf_inputs["input_0"].numpy()
        dev0_in_exp1_weight = raf_inputs["input_1"].numpy()
        dev0_in_exp1_nelements = raf_inputs["input_2"].numpy()
        dev0_in_exp1_out = raf_inputs["output"].numpy()

        with open("/models/opt_ios/device_0_raf.op.cublas.sparse_expert_matmul_nt_inouts_1.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_in_exp0_data = raf_inputs["input_0"].numpy()
        dev0_in_exp0_weight = raf_inputs["input_1"].numpy()
        dev0_in_exp0_nelements = raf_inputs["input_2"].numpy()
        dev0_in_exp0_out = raf_inputs["output"].numpy()

        with open("/models/opt_ios/device_0_raf.op.cublas.sparse_expert_matmul_nt_inouts_2.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_out_exp1_data = raf_inputs["input_0"].numpy()
        dev0_out_exp1_weight = raf_inputs["input_1"].numpy()
        dev0_out_exp1_nelements = raf_inputs["input_2"].numpy()
        dev0_out_exp1_out = raf_inputs["output"].numpy()

        with open("/models/opt_ios/device_0_raf.op.cublas.sparse_expert_matmul_nt_inouts_3.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_out_exp0_data = raf_inputs["input_0"].numpy()
        dev0_out_exp0_weight = raf_inputs["input_1"].numpy()
        dev0_out_exp0_nelements = raf_inputs["input_2"].numpy()
        dev0_out_exp0_out = raf_inputs["output"].numpy()

        with open("/models/opt_ios/device_1_raf.op.cublas.sparse_expert_matmul_nt_inouts_2.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev1_out_exp1_data = raf_inputs["input_0"].numpy()
        dev1_out_exp1_weight = raf_inputs["input_1"].numpy()
        dev1_out_exp1_nelements = raf_inputs["input_2"].numpy()
        dev1_out_exp1_out = raf_inputs["output"].numpy()

        with open("/models/opt_ios/device_1_raf.op.cublas.sparse_expert_matmul_nt_inouts_3.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev1_out_exp0_data = raf_inputs["input_0"].numpy()
        dev1_out_exp0_weight = raf_inputs["input_1"].numpy()
        dev1_out_exp0_nelements = raf_inputs["input_2"].numpy()
        dev1_out_exp0_out = raf_inputs["output"].numpy()

        send_cnts = out_dev_0_a2a_send_cnts[0] // 768
        recv_cnts = out_dev_0_a2a_recv_cnts[0] // 768
        print("out a2a send counts: ", out_dev_0_a2a_send_cnts[0] // 768)
        print("out exp0 n elements: ", dev0_out_exp0_nelements // 768)
        print("out exp1 n elements: ", dev0_out_exp1_nelements // 768)

        a0, b0, a1, b1 = np.split(out_dev_0_a2a_inputs[0], 4, axis=0)
        a0[:, send_cnts[0]:, :] = 0
        b0[:, send_cnts[1]:, :] = 0
        a1[:, send_cnts[2]:, :] = 0
        b1[:, send_cnts[3]:, :] = 0

        dev0_out_a0, dev0_out_a1 = np.split(dev0_out_exp0_out.transpose((1, 0, 2)), 2, axis=0)
        dev0_out_data_a0, dev0_out_data_a1 = np.split(dev0_out_exp0_data.transpose((1, 0, 2)), 2, axis=0)
        dev0_out_b0, dev0_out_b1 = np.split(dev0_out_exp1_out.transpose((1, 0, 2)), 2, axis=0)
        dev0_out_data_b0, dev0_out_data_b1 = np.split(dev0_out_exp1_data.transpose((1, 0, 2)), 2, axis=0)
        dev0_out_a0[:, send_cnts[0]:, :] = 0
        dev0_out_b0[:, send_cnts[1]:, :] = 0
        dev0_out_a1[:, send_cnts[2]:, :] = 0
        dev0_out_b1[:, send_cnts[3]:, :] = 0
        dev0_out_data_a0[:, send_cnts[0]:, :] = 0
        dev0_out_data_b0[:, send_cnts[1]:, :] = 0
        dev0_out_data_a1[:, send_cnts[2]:, :] = 0
        dev0_out_data_b1[:, send_cnts[3]:, :] = 0



        dev0_in_a0, dev0_in_a1 = np.split(dev0_in_exp0_out.transpose((1, 0, 2)), 2, axis=0)
        dev0_in_data_a0, dev0_in_data_a1 = np.split(dev0_in_exp0_data.transpose((1, 0, 2)), 2, axis=0)
        dev0_in_b0, dev0_in_b1 = np.split(dev0_in_exp1_out.transpose((1, 0, 2)), 2, axis=0)
        dev0_in_data_b0, dev0_in_data_b1 = np.split(dev0_in_exp1_data.transpose((1, 0, 2)), 2, axis=0)
        dev0_in_a0[:, send_cnts[0]:, :] = 0
        dev0_in_b0[:, send_cnts[1]:, :] = 0
        dev0_in_a1[:, send_cnts[2]:, :] = 0
        dev0_in_b1[:, send_cnts[3]:, :] = 0
        dev0_in_data_a0[:, send_cnts[0]:, :] = 0
        dev0_in_data_b0[:, send_cnts[1]:, :] = 0
        dev0_in_data_a1[:, send_cnts[2]:, :] = 0
        dev0_in_data_b1[:, send_cnts[3]:, :] = 0

        print("Max diff dev0_out_a0: ", np.max(np.abs(dev0_out_a0 - dev0_out_a1)))
        print("Max diff dev0_out_b0: ", np.max(np.abs(dev0_out_b0 - dev0_out_b1)))

        with open("/models/opt_ios/device_0_raf.op.cuda.moe_decode_inouts_0.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_data = raf_inputs["input_0"].numpy()
        dev0_gate = raf_inputs["input_1"].numpy()
        dev0_ind_locs = raf_inputs["input_2"].numpy()
        dev0_out = raf_inputs["output"].numpy()

        with open("/models/opt_ios/device_1_raf.op.cuda.moe_decode_inouts_0.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev1_data = raf_inputs["input_0"].numpy()
        dev1_gate = raf_inputs["input_1"].numpy()
        dev1_ind_locs = raf_inputs["input_2"].numpy()
        dev1_out = raf_inputs["output"].numpy()

        count_per_expert = defaultdict(int)
        for i in range(dev0_ind_locs.shape[1]):
            if dev0_ind_locs[0, i] >= 0:
                count_per_expert[dev0_ind_locs[0, i]] += 1


        print("Max diff dev0_decode_out: ", np.max(np.abs(dev0_out - dev1_out)))
        with open("/models/opt_ios/device_0_raf.op.cuda.moe_redispatch_expert_input_inouts_0.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_out_exp1_data_p0 = raf_inputs["input_0_0"].numpy().reshape(2048, 2, -1)
        dev0_out_exp1_data_p1 = raf_inputs["input_0_1"].numpy().reshape(2048, 2, -1)
        dev0_out_exp1_recon = raf_inputs["output"].numpy().reshape(2048, 2, -1)

        with open("/models/opt_ios/device_0_raf.op.cuda.moe_redispatch_expert_input_inouts_1.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_in_exp1_data_p0 = raf_inputs["input_0_0"].numpy().reshape(2048, 2, -1)
        dev0_in_exp1_data_p1 = raf_inputs["input_0_1"].numpy().reshape(2048, 2, -1)
        dev0_in_exp1_recon = raf_inputs["output"].numpy().reshape(2048, 2, -1)

        with open("/models/opt_ios/device_0_raf.op.cuda.moe_redispatch_expert_input_inouts_2.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_out_exp0_data_p0 = raf_inputs["input_0_0"].numpy().reshape(2048, 2, -1)
        dev0_out_exp0_data_p1 = raf_inputs["input_0_1"].numpy().reshape(2048, 2, -1)
        dev0_out_exp0_recon = raf_inputs["output"].numpy().reshape(2048, 2, -1)

        with open("/models/opt_ios/device_0_raf.op.cuda.moe_redispatch_expert_input_inouts_3.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_in_exp0_data_p0 = raf_inputs["input_0_0"].numpy().reshape(2048, 2, -1)
        dev0_in_exp0_data_p1 = raf_inputs["input_0_1"].numpy().reshape(2048, 2, -1)
        dev0_in_exp0_recon = raf_inputs["output"].numpy().reshape(2048, 2, -1)

        send_cnts = out_dev_0_a2a_send_cnts[0] // 768
        for le in range(2):
            print("Max diff dev0_in_exp0_p0: ", np.max(np.abs(dev0_in_exp0_data_p0[:send_cnts[le*2],le,:] - dev0_in_exp0_data[:send_cnts[le*2],le,:])))
            print("Max diff dev0_in_exp1_p0: ", np.max(np.abs(dev0_in_exp1_data_p0[:send_cnts[le*2+1],le,:] - dev0_in_exp1_data[:send_cnts[le*2+1],le,:])))
            print("Max diff dev0_out_exp0_p0: ", np.max(np.abs(dev0_out_exp0_data_p0[:send_cnts[le*2],le,:] - dev0_out_exp0_data[:send_cnts[le*2],le,:])))
            print("Max diff dev0_out_exp1_p0: ", np.max(np.abs(dev0_out_exp1_data_p0[:send_cnts[le*2+1],le,:] - dev0_out_exp1_data[:send_cnts[le*2+1],le,:])))

        with open("/models/non_opt_ios/device_0_raf.op.cublas.matmul_nt_inouts_1.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_ref_in_exp0_data = raf_inputs["input_0"].numpy().reshape(2048, 2, -1)
        print("dev0_ref_in_exp0_data.shape: ", dev0_ref_in_exp0_data.shape)

        with open("/models/non_opt_ios/device_0_raf.op.cublas.matmul_nt_inouts_2.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_ref_in_exp1_data = raf_inputs["input_0"].numpy().reshape(2048, 2, -1)
        print("dev0_ref_in_exp1_data.shape: ", dev0_ref_in_exp1_data.shape)

        with open("/models/non_opt_ios/device_0_raf.op.cublas.matmul_nt_inouts_3.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_ref_out_exp0_data = raf_inputs["input_0"].numpy().reshape(2048, 2, -1)
        print("dev0_ref_out_exp0_data.shape: ", dev0_ref_out_exp0_data.shape)

        with open("/models/non_opt_ios/device_0_raf.op.cublas.matmul_nt_inouts_4.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        dev0_ref_out_exp1_data = raf_inputs["input_0"].numpy().reshape(2048, 2, -1)
        print("dev0_ref_out_exp1_data.shape: ", dev0_ref_out_exp1_data.shape)

        for le in range(2):
            print("Max diff dev0_in_exp0_recon: ", np.max(np.abs(dev0_in_exp0_recon[:send_cnts[le*2],le,:] - dev0_ref_in_exp0_data[:send_cnts[le*2],le,:])))
            print("Max diff dev0_in_exp1_recon: ", np.max(np.abs(dev0_in_exp1_recon[:send_cnts[le*2+1],le,:] - dev0_ref_in_exp1_data[:send_cnts[le*2+1],le,:])))
            print("Max diff dev0_out_exp0_recon: ", np.max(np.abs(dev0_out_exp0_recon[:send_cnts[le*2],le,:] - dev0_ref_out_exp0_data[:send_cnts[le*2],le,:])))
            print("Max diff dev0_out_exp1_recon: ", np.max(np.abs(dev0_out_exp1_recon[:send_cnts[le*2+1],le,:] - dev0_ref_out_exp1_data[:send_cnts[le*2+1],le,:])))

        import code
        code.interact(local=locals())
        exit(0)


        data_shape = tuple(raf_input_datas[0].shape)
        mask_shape = tuple(raf_input_masks[0].shape)
        dtype = "float32"
        def test():
            builder = ANFBuilder()
            data_0 = extended_var("data0", shape=data_shape, dtype=dtype)
            mask_0 = extended_var("mask0", shape=mask_shape, dtype="int32")
            data_1 = extended_var("data1", shape=data_shape, dtype=dtype)
            mask_1 = extended_var("mask1", shape=mask_shape, dtype="int32")
            data_2 = extended_var("data2", shape=data_shape, dtype=dtype)
            mask_2 = extended_var("mask2", shape=mask_shape, dtype="int32")
            data_3 = extended_var("data3", shape=data_shape, dtype=dtype)
            mask_3 = extended_var("mask3", shape=mask_shape, dtype="int32")

            out = builder.call("moe_redispatch", [builder.make_tuple([data_0, data_1, data_2, data_3]), builder.make_tuple([mask_0, mask_1, mask_2, mask_3])])
            return tvm.relay.Function([data_0, data_1, data_2, data_3, mask_0, mask_1, mask_2, mask_3], builder.ret(out))

        def ref():
            ref_out = np.zeros_like(data_np)
            current_loc_per_expert = defaultdict(int)
            for partition in range(4):
                data = np.reshape(raf_input_datas[partition], (-1, data_np.shape[1], data_np.shape[2]))
                mask = raf_input_masks[partition]
                for idx in range(mask.shape[1]):
                    ind = mask[0, idx]
                    loc = mask[1, idx]
                    if ind >= 0:
                        out_loc = current_loc_per_expert[ind]
                        ref_out[ind, out_loc, :] = data[ind, loc, :]
                        current_loc_per_expert[ind] += 1
                        if out_loc == 0:
                            print(f"Out[{ind}, {out_loc}, :] is assigned using data_{partition}[{ind}, {loc}, :], sequence: {idx}")

            return ref_out

        data_arrays = [raf.array(data, device="cuda(0)", dtype="float32") for data in raf_input_datas]
        mask_arrays = [raf.array(mask, device="cuda(0)", dtype="int32") for mask in raf_input_masks]

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        tout = executor(*data_arrays, *mask_arrays)
        rout = ref()

        import code
        code.interact(local=locals())

@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_decode(S, M, E, dtype):
    C = (S + E - 1) // E
    data_shape = (E, C, M)
    mask_shape = (2, S)
    gate_shape = (S,)
    with Device("cuda(0)"):
        # with np.load('./moe_decode_fw_inouts_8.npz') as npz_data:
        #     data_np = npz_data['data']
        #     gate_np = npz_data['gate']
        #     ind_locs_np = npz_data['ind_locs']
        #     out_np = npz_data['out']
        # indices_np = ind_locs_np[0, :]
        # locations_np = ind_locs_np[1, :]

        with open("/models/non_optimized_inouts/device_0_raf.op.cuda.moe_decode_inouts_0.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        data_np = raf_inputs["input_0"].numpy()
        gate_np = raf_inputs["input_1"].numpy()
        ind_locs_np = raf_inputs["input_2"].numpy()
        out_np = raf_inputs["output"].numpy()

        with open("/models/optimized_inouts/device_0_raf.op.cuda.moe_decode_inouts_0.bin", "rb") as f:
            param_bytes = f.read()
        raf_inputs = tvm.runtime.load_param_dict(param_bytes)
        data_raf_np = raf_inputs["input_0"].numpy()
        gate_raf_np = raf_inputs["input_1"].numpy()
        ind_locs_raf_np = raf_inputs["input_2"].numpy()
        out_raf_np = raf_inputs["output"].numpy()

        for i in range(ind_locs_np.shape[1]):
            if ind_locs_np[0, i] == -1:
                ind_locs_np[1, i] = -1

        print("data_np allclose: {}".format(np.allclose(data_np, data_raf_np, rtol=1e-4, atol=1e-4)))
        print("gate_np allclose: {}".format(np.allclose(gate_np, gate_raf_np, rtol=1e-4, atol=1e-4)))
        print("ind_locs_np allclose: {}".format(np.allclose(ind_locs_np, ind_locs_raf_np, rtol=1e-4, atol=1e-4)))
        import code
        code.interact(local=locals())

        check(data_np, data_raf_np, rtol=1e-5, atol=1e-5)
        check(gate_np, gate_raf_np, rtol=1e-5, atol=1e-5)
        check(ind_locs_np, ind_locs_raf_np, rtol=1e-5, atol=1e-5)
        exit(0)

        data_np = np.random.rand(*data_shape).astype(dtype)
        gate_np = np.random.rand(*gate_shape).astype(dtype)

        indices_np = np.random.randint(0, E, size=gate_shape, dtype=np.int32)
        locations_np = np.random.randint(0, C, size=gate_shape, dtype=np.int32)
        ind_locs_np = np.stack([indices_np, locations_np], axis=0)

        data_shape = tuple(data_np.shape)
        gate_shape = tuple(gate_np.shape)
        mask_shape = tuple(ind_locs_np.shape)

        def test():
            builder = ANFBuilder()
            data = extended_var("data", shape=data_shape, dtype=dtype)
            gate_s = extended_var("gate", shape=gate_shape, dtype=dtype)
            indices_locations = extended_var("indices_locations", shape=mask_shape, dtype="int32")
            out = builder.call("moe_decode", [data, gate_s, indices_locations])
            return tvm.relay.Function([data, gate_s, indices_locations], builder.ret(out))

        data = raf.array(data_np, device="cuda(0)")
        gate = raf.array(gate_np, device="cuda(0)")
        indices_locations = raf.array(ind_locs_np, device="cuda(0)", dtype="int32")
        C = data_np.shape[1]
        E = data_np.shape[0]
        def ref():
            rout = np.zeros((S, M), dtype=dtype)
            # rout = np.zeros_like(out_np, dtype=dtype)
            for i in range(S):
                loc = locations_np[i]
                idx = indices_np[i]
                if idx < E and loc < C:
                    rout[i] = data_np[idx, loc] * gate_np[i]
            return rout

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        tout = executor(data, gate, indices_locations)
        rout = ref()

        check(tout, rout, rtol=1e-5, atol=1e-5)

@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_decode_dx(S, M, E, dtype):
    C = (S + E - 1) // E
    data_shape = (E, C, M)
    mask_shape = (2, S)
    dy_shape = (S, M)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            dy = extended_var("dy", shape=dy_shape, dtype=dtype)
            gate_s = extended_var("gate", shape=(S, ), dtype=dtype)
            indices_locations = extended_var("indices_locations", shape=mask_shape, dtype="int32")
            
            out = builder.call("moe_decode_dx", [dy, gate_s, indices_locations, builder.int_const(E), builder.int_const(C)])
            return tvm.relay.Function([dy, gate_s, indices_locations], builder.ret(out))

        dy_np = np.random.rand(*dy_shape).astype(dtype)
        dy = raf.array(dy_np, device="cuda(0)")

        gate_np = np.random.rand(*(S, )).astype(dtype)
        gate = raf.array(gate_np, device="cuda(0)")

        indices_np = np.random.randint(-1, E, size=(S, ), dtype=np.int32)

        per_expert_tokens = defaultdict(int)
        locations_np = np.zeros_like(indices_np)
        for i in range(S):
            if indices_np[i] >= 0:
                curr_tokens = per_expert_tokens[indices_np[i]]
                if curr_tokens < C:
                    locations_np[i] = curr_tokens
                    per_expert_tokens[indices_np[i]] += 1
                else:
                    indices_np[i] = -1
                    locations_np[i] = -1

        ind_locs_np = np.concatenate([np.expand_dims(indices_np, 0), np.expand_dims(locations_np, 0)], axis=0)
        ind_locs = raf.array(ind_locs_np, device="cuda(0)")

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)

        out = executor(dy, gate, ind_locs)
        rout = out.numpy()

        ref_out = np.zeros(data_shape, dtype=dtype)
        for i in range(S):
            idx = indices_np[i]
            loc = locations_np[i]
            if idx >= 0:
                ref_out[idx, loc, :] = dy_np[i] * gate_np[i]

        check(rout, ref_out, rtol=1e-5, atol=1e-5)


@with_dialect(["tvm", "cuda"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("M", [128])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("dtype", ["float32"])
def test_moe_decode_dg(S, M, E, dtype):
    C = (S + E - 1) // E
    data_shape = (E, C, M)
    mask_shape = (2, S)
    dy_shape = (S, M)
    with Device("cuda(0)"):
        def test():
            builder = ANFBuilder()
            dy = extended_var("dy", shape=dy_shape, dtype=dtype)
            data = extended_var("data", shape=data_shape, dtype=dtype)
            indices_locations = extended_var("indices", shape=mask_shape, dtype="int32")

            out = builder.call("moe_decode_dg", [dy, data, indices_locations])
            return tvm.relay.Function([dy, data, indices_locations], builder.ret(out))

        dy_np = np.random.rand(*dy_shape).astype(dtype)
        dy = raf.array(dy_np, device="cuda(0)")

        data_np = np.random.rand(*data_shape).astype(dtype)
        data = raf.array(data_np, device="cuda(0)")

        indices_np = np.random.randint(-1, E, size=(S, ), dtype=np.int32)

        per_expert_tokens = defaultdict(int)
        locations_np = np.zeros_like(indices_np)
        for i in range(S):
            if indices_np[i] >= 0:
                curr_tokens = per_expert_tokens[indices_np[i]]
                if curr_tokens < C:
                    locations_np[i] = curr_tokens
                    per_expert_tokens[indices_np[i]] += 1
                else:
                    indices_np[i] = -1
                    locations_np[i] = -1

        ind_locs_np = np.concatenate([np.expand_dims(indices_np, 0), np.expand_dims(locations_np, 0)], axis=0)
        ind_locs = raf.array(ind_locs_np, device="cuda(0)")

        test_mod = tvm.IRModule()
        test_mod["main"] = test()

        device = f"cuda(0)"
        executor = get_vm_executor(test_mod, device)
        out = executor(dy, data, ind_locs)
        rout = out.numpy()

        ref_out = np.zeros((S, ), dtype=dtype)
        for i in range(S):
            idx = indices_np[i]
            loc = locations_np[i]
            if idx >= 0 and loc < C:
                ref_out[i] = np.sum(dy_np[i] * data_np[idx, loc])

        check(rout, ref_out, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    # pytest.main([__file__])
    # test_moe_encode(16, 2, 2, "float32")
    # test_partitioned_moe_encode_merge_masks_and_data(2)
    # test_partitioned_moe_encode_merge_masks_and_data(4)
    # test_partitioned_moe_encode(512, 768, 2, "float32")
    # test_partitioned_moe_encode_decode(512, 768, 2, "float32")
    # test_moe_decode(512, 768, 2, "float32")
    # test_moe_encode_dg(1024, 32, "float32")
    # test_moe_encode_dx(1024, 768, 32, "float32")
    # test_moe_decode_dx(1024, 768, 32, "float32")
    # test_moe_decode_dg(1024, 768, 32, "float32")
    # test_moe_redispatch()
    # test_moe_encode_batch_prioritized(1024, 512, 16, "float32")
    test_moe_encode_batch_prioritized_partitioned(1024, 512, 16, 4, 1, "float32")
