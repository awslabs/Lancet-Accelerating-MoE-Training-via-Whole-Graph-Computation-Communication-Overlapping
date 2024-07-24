# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements,no-self-use,too-many-arguments
import numpy as np
import math

import raf
from raf._core.device import Device
from raf import distributed as dist
from raf._ffi.pass_ import DispatchDialect, InferType, ToGraphNormalForm
from raf.distributed import PartitionExpr
from raf.testing import with_dialect

dctx = dist.get_context()
SKIP_REASON = "Distribution is not enabled or #rank is not expected"

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_simple_matmul():
    with Device("cuda(0)"):
        x_shape = (768, 128, 64)
        w_shape = (768, 768)

        class TestModel(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, w):
                x = raf.reshape(x, (64, 12, 128, 64))
                x = raf.transpose(x, (0, 2, 1, 3))
                x = raf.reshape(x, (8192, 768))
                return raf.matmul(x, w)

        device = "cuda(0)"
        model = TestModel()
        x = np.random.randn(*x_shape).astype("float32")
        w = np.random.randn(*w_shape).astype("float32")
        x = raf.array(x, device=device)
        w = raf.array(w, device=device)
        model.to(device=device)

        record = model._internal(x, w)
        mod = record.mod
        mod = ToGraphNormalForm()(mod)
        mod = InferType()(mod)
        mod = DispatchDialect()(mod)
        func = mod["main"]

        partitioned_func = PartitionExpr(func, 1, 1)
        if partitioned_func is not None:
            print("Partitioned mod ", raf.ir.AsText(partitioned_func))
        else:
            print("Failed to partition.")

def raf_self_attention(input_tensor, qkv_weight, qkv_bias, attn_mask, proj_weight, proj_bias, one_over_sqrt_kv, B, S, M, KV):
    NH = M // KV
    input = raf.layer_norm(input_tensor)        # [B, S, M]
    input = raf.reshape(input, (B * S, M))      # [BS, M]
    qkv = raf.matmul(input, qkv_weight)         # [BS, 3M]
    qkv = raf.add(qkv, qkv_bias)                # [BS, 3M]
    qkv = raf.reshape(qkv, (B, S, 3 * M))       # [B, S, 3M]
    q_k_v = raf.split(qkv, 3, axis=2)           # [B, S, M] * 3
    q_k_v_list = []
    for i in range(3):
        t = q_k_v[i]
        t = raf.reshape(t, (B, S, NH, KV))      # [B, S, NH, KV]
        if i == 0 or i == 1:
            # q or k
            t = raf.transpose(t, (0, 2, 1, 3))  # [B, NH, S, KV]
            t = raf.reshape(t, (B * NH, S, KV)) # [BNH, S, KV]
        else:
            t = raf.transpose(t, (0, 2, 3, 1))  # [B, NH, KV, S]
            t = raf.reshape(t, (B * NH, KV, S)) # [BNH, KV, S]
        q_k_v_list.append(t)
    q, k, v = q_k_v_list
    attn_score = raf.batch_matmul_nt(q, k)              # [BNH, S, S]
    attn_score = raf.reshape(attn_score, (B, NH, S, S)) # [B, NH, S, S]
    attn_score = raf.multiply(attn_score, one_over_sqrt_kv)
    attn_score = raf.where(attn_mask, attn_score, -1e9)
    attn_prob = raf.softmax(attn_score, axis=3)          # [B, NH, S, S]
    attn_prob = raf.reshape(attn_prob, (B * NH, S, S))   # [BNH, S, S]
    attn_out = raf.batch_matmul_nt(attn_prob, v)         # [BNH, S, KV]
    attn_out = raf.reshape(attn_out, (B, NH, S, KV))     # [B, NH, S, KV]
    attn_out = raf.transpose(attn_out, (0, 2, 1, 3))     # [B, S, NH, KV]
    attn_out = raf.reshape(attn_out, (B * S, M))         # [BS, M]
    attn_out = raf.matmul(attn_out, proj_weight)          # [BS, M]
    attn_out = raf.add(attn_out, proj_bias)               # [BS, M]
    attn_out = raf.reshape(attn_out, (B, S, M))           # [B, S, M]
    return attn_out

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_moe_decode_self_attn():
    with Device("cuda(0)"):
        B = 32
        S = 128
        BS = B * S
        M = 768
        E = 16
        KV = 64

        x_shape = (BS, M)
        w_shape = (BS, E)
        qkv_weight_shape = (M, 3 * M)
        qkv_bias_shape = (3 * M, )
        attn_mask_shape = (1, 1, S, S)
        proj_weight_shape = (M, M)
        proj_bias_shape = (M, )

        class TestModel(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, w, zeros, residual, qkv_weight, qkv_bias, attn_mask, proj_weight, proj_bias, one_over_sqrt_kv):
                dispatch_out = raf.moe_encode(x, w, zeros)
                dispatched_data = dispatch_out[4]
                gate_s = dispatch_out[0]
                indloc = dispatch_out[1]
                merged = raf.moe_decode(dispatched_data, gate_s, indloc)
                merged_reshape = raf.reshape(merged, (B, S, M))
                next_layer_in = raf.add(merged_reshape, residual)
                next_layer_out = raf_self_attention(next_layer_in, qkv_weight, qkv_bias, attn_mask, proj_weight, proj_bias, one_over_sqrt_kv, B, S, M, KV)
                return next_layer_out

        device = "cuda(0)"
        model = TestModel()
        x = np.random.randn(*x_shape).astype("float32")
        w = np.random.randn(*w_shape).astype("float32")
        zeros = np.zeros((E, )).astype("int32")
        residual = np.random.randn(B, S, M).astype("float32")
        qkv_weight = np.random.randn(*qkv_weight_shape).astype("float32")
        qkv_bias = np.random.randn(*qkv_bias_shape).astype("float32")
        attn_mask = np.random.randn(*attn_mask_shape).astype("uint8")
        proj_weight = np.random.randn(*proj_weight_shape).astype("float32")
        proj_bias = np.random.randn(*proj_bias_shape).astype("float32")
        one_over_sqrt_kv = 1.0 / math.sqrt(KV)
        x = raf.array(x, device=device)
        w = raf.array(w, device=device)
        zeros = raf.array(zeros, device=device)
        residual = raf.array(residual, device=device)
        qkv_weight = raf.array(qkv_weight, device=device)
        qkv_bias = raf.array(qkv_bias, device=device)
        attn_mask = raf.array(attn_mask, device=device)
        proj_weight = raf.array(proj_weight, device=device)
        proj_bias = raf.array(proj_bias, device=device)
        one_over_sqrt_kv = raf.array(one_over_sqrt_kv, device=device, dtype="float32")
        model.to(device=device)

        record = model._internal(x, w, zeros, residual, qkv_weight, qkv_bias, attn_mask, proj_weight, proj_bias, one_over_sqrt_kv)
        mod = record.mod
        mod = ToGraphNormalForm()(mod)
        mod = InferType()(mod)
        mod = DispatchDialect()(mod)
        func = mod["main"]

        partitioned_func = PartitionExpr(func, 2, E)
        if partitioned_func is not None:
            print("Partitioned mod ", raf.ir.AsText(partitioned_func))
        else:
            print("Failed to partition.")

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_moe_decode_self_attn():
    with Device("cuda(0)"):
        S = 1024
        G = 16
        E = 32
        C = 64
        M = 768

        x_shape = (E, C, M)
        expert_w_shape = (4*M, M)
        expert_w1_shape = (M, 4*M)

        class TestModel(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, expert_w, expert_w1):
                x = raf.all_to_all(x)
                x = raf.reshape(x, (G, E // G, C, M))
                x = raf.split(x, E // G, axis=1)
                xis = []
                for i in range(2):
                    xi = x[i]
                    xi = raf.reshape(xi, (G * C, M))
                    # xi = raf.matmul_nt(xi, expert_w)
                    # xi = raf.matmul_nt(xi, expert_w1)
                    xi = raf.reshape(xi, (G, 1, C, M))
                    xis.append(xi)
                x = raf.concatenate(xis, axis=1)
                # return raf.all_to_all(x)
                return x

        device = "cuda(0)"
        model = TestModel()
        x = np.random.randn(*x_shape).astype("float32")
        w = np.random.randn(*expert_w_shape).astype("float32")
        w1 = np.random.randn(*expert_w1_shape).astype("float32")
        x = raf.array(x, device=device)
        w = raf.array(w, device=device)
        w1 = raf.array(w1, device=device)
        model.to(device=device)

        record = model._internal(x, w, w1)
        mod = record.mod
        mod = ToGraphNormalForm()(mod)
        mod = InferType()(mod)
        mod = DispatchDialect()(mod)
        func = mod["main"]

        partitioned_func = PartitionExpr(func, 16, E)
        if partitioned_func is not None:
            print("Partitioned mod ", raf.ir.AsText(partitioned_func))
        else:
            print("Failed to partition.")

@with_dialect(["cublas", "nccl", "cuda", "tvm"])
def test_moe_encode_a2a_expert_a2a_decode():
    with Device("cuda(0)"):
        S = 1024
        G = 16
        E = 32
        LE = E // G
        C = S // E
        M = 768

        dctx.size = G
        dctx.set_local_rank_for_tuning(0)

        x_shape = (S, M)
        gate_shape = (S, E)
        usedcap_shape = (E, )
        expert_w_shape = (4*M, M)
        expert_w1_shape = (M, 4*M)

        class TestModel(raf.Model):
            def build(self):
                pass

            @raf.model.trace
            def forward(self, x, gate, usedcap, expert_w, expert_w1):
                encode_out = raf.moe_encode(x, gate, usedcap)
                gate_s = encode_out[0]
                indloc = encode_out[1]
                data = encode_out[4]
                data = raf.all_to_all(data)
                data = raf.reshape(data, (G, LE, C, M))
                data = raf.split(data, LE, axis=1)
                data_ls = [data[i] for i in range(LE)]
                data_out = []
                for i in range(LE):
                    x = data_ls[i]
                    x = raf.squeeze(x) # (G, C, M)
                    x = raf.transpose(x, (1, 0, 2)) # (C, G, M)
                    x = raf.reshape(x, (-1, M)) # (C*G, M)
                    x = raf.matmul_nt(x, expert_w) # (C*G, 4*M)
                    x = raf.relu(x) # (C*G, 4*M)
                    x = raf.matmul_nt(x, expert_w1) # (C*G, M)
                    x = raf.reshape(x, (C, G, M)) # (C, G, M)
                    x = raf.transpose(x, (1, 0, 2)) # (G, C, M)
                    x = raf.expand_dims(x, axis=1) # (G, 1, C, M)
                    data_out.append(x)
                data_out = raf.concatenate(data_out, axis=1) # (G, LE, C, M)
                data_out = raf.all_to_all(data_out) # (G, LE, C, M)
                data_out = raf.reshape(data_out, (E, C, M))
                return raf.moe_decode(data_out, gate_s, indloc)

        device = "cuda(0)"
        model = TestModel()
        x = np.random.randn(*x_shape).astype("float32")
        gate = np.random.randn(*gate_shape).astype("float32")
        usedcap = np.zeros(usedcap_shape, dtype="int32")
        w = np.random.randn(*expert_w_shape).astype("float32")
        w1 = np.random.randn(*expert_w1_shape).astype("float32")
        x = raf.array(x, device=device)
        gate = raf.array(gate, device=device)
        usedcap = raf.array(usedcap, device=device)
        w = raf.array(w, device=device)
        w1 = raf.array(w1, device=device)
        model.to(device=device)

        record = model._internal(x, gate, usedcap, w, w1)
        mod = record.mod
        mod = ToGraphNormalForm()(mod)
        mod = InferType()(mod)
        mod = DispatchDialect()(mod)
        func = mod["main"]

        partitioned_func = PartitionExpr(func, 16, E)
        if partitioned_func is not None:
            print("Partitioned mod ", raf.ir.AsText(partitioned_func))
        else:
            print("Failed to partition.")

if __name__ == "__main__":
    # test_simple_matmul()
    test_moe_encode_a2a_expert_a2a_decode()