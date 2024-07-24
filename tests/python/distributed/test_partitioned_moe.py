# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
"""Test collective communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test partitioned_moe, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_partitioned_moe.py`
(in ci/tash_python_unittest.sh)
"""
import sys
from collections import defaultdict
import pytest
import numpy as np

import raf
from raf import distributed as dist
from raf._core.device import Device
from raf.testing import check, get_dist_info, skip_dist_test, run_vm_model, run_model

dctx = dist.get_context()
SKIP_REASON = "Distribution is not enabled or #rank is not expected"

@pytest.mark.skipif(skip_dist_test(min_rank_num=2), reason=SKIP_REASON)
def test_partitioned_moe(n_partitions):
    assert n_partitions >= 2

    S_before_partition = 1024
    M = 256
    E = 2
    S = S_before_partition // n_partitions
    C = S_before_partition // E

    class PartitionedModel(raf.Model):
        def build(self, n_partitions, device):
            self.n_partitions = n_partitions
            self.device = device

        @raf.model.trace
        def forward(self, x, gate_w, expert_w1, expert_w2):
            if self.n_partitions == 1:
                partitioned_xs = [x]
            else:
                partitioned_xs = raf.split(x, self.n_partitions, axis=0)
            # run dispatch
            capused = raf.zeros(shape=(E,), dtype="int32", device=self.device)
            expert_outs = []
            for i in range(self.n_partitions):
                xi = partitioned_xs[i]
                # gate
                gate_se = raf.matmul(xi, gate_w)
                # dispatch
                encode_out = raf.moe_encode(xi, gate_se, capused, self.n_partitions)
                gate_s = encode_out[0]
                indices_locations = encode_out[1]
                capused1_e = encode_out[2]
                capused = capused1_e
                capused_current = encode_out[3]
                dispatched_input = encode_out[4]
                # reshape + a2av + expert + reshape + a2av
                dispatched_input = raf.reshape(dispatched_input, (E*C, M))
                a2aout = raf.all_to_allv(dispatched_input, capused_current)
                received_input = a2aout[0]
                # received_input = dispatched_input
                recved_cnts = a2aout[1]
                expert_out = raf.matmul(received_input, expert_w1)
                expert_out = raf.matmul(expert_out, expert_w2)
                expert_out = raf.reshape(expert_out, (E, C, M))
                expert_out = raf.all_to_allv(expert_out, recved_cnts)[0]
                # gather
                gathered_out = raf.moe_decode(expert_out, gate_s, indices_locations)
                expert_outs.append(gathered_out)
            if len(expert_outs) == 1:
                return expert_outs[0]
            return raf.concatenate(expert_outs, axis=0)

    total_rank, rank, local_rank = get_dist_info(verbose=True)
    device = f"cuda({local_rank})"
    partitioned_model = PartitionedModel(n_partitions, device)
    reference_model = PartitionedModel(1, device)

    x = np.random.randn(S_before_partition, M).astype("float32")
    gate_w = np.random.randn(M, E).astype("float32")
    expert_w1 = np.random.randn(M, 4 * M).astype("float32")
    expert_w2 = np.random.randn(4 * M, M).astype("float32")
    x = raf.array(x, device=device) 
    gate_w = raf.array(gate_w, device=device)
    expert_w1 = raf.array(expert_w1, device=device)
    expert_w2 = raf.array(expert_w2, device=device)

    partitioned_model.to(device=device)
    reference_model.to(device=device)

    partitioned_out = run_vm_model(partitioned_model, device, [x, gate_w, expert_w1, expert_w2])
    reference_out = run_vm_model(reference_model, device, [x, gate_w, expert_w1, expert_w2])
    check(partitioned_out, reference_out, atol=1e-5, rtol=1e-5)


def test_partitioned_moe_encode_dist_merge(n_partitions):

    S_before_partition = 1024
    M = 512
    LE = 2

    total_rank, rank, local_rank = get_dist_info(verbose=True)

    E = LE * total_rank
    C = S_before_partition // E

    with Device(f"cuda({local_rank})"):
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
                expert_ins = [[] for _ in range(LE)]
                recved_cntss = []
                for i in range(self.n_partitions):
                    xi = partitioned_xs[i]
                    # gate
                    gate_se = raf.matmul(xi, gate_w)
                    # dispatch
                    encode_out = raf.moe_encode(xi, gate_se, capused, self.n_partitions)
                    capused = encode_out[2]
                    nelements_per_expert = encode_out[3]
                    dispatched_input = encode_out[4]
                    # alltoallv
                    a2aout = raf.all_to_allv(dispatched_input, nelements_per_expert)
                    recved_cnts = a2aout[1]
                    expert_input = a2aout[0]
                    expert_input = raf.reshape(expert_input, (total_rank, LE, -1, M))
                    expert_inputs = raf.split(expert_input, LE, axis=1)
                    for le in range(LE):
                        exp_in = raf.reshape(expert_inputs[le], (total_rank, 1, C, M))
                        exp_in = raf.transpose(exp_in, (2, 1, 0, 3))
                        exp_in = raf.reshape(exp_in, (-1, M))
                        expert_ins[le].append(exp_in)
                    recved_cntss.append(recved_cnts)
                if self.n_partitions > 1:
                    # redispatch a2aout
                    redispatched_le_ins = []
                    for expert_id in range(LE):
                        redispatched_expert_inputs = raf.moe_redispatch_expert_input(expert_ins[expert_id], recved_cntss, expert_id, LE)
                        redispatched_le_ins.append(redispatched_expert_inputs)
                    return redispatched_le_ins
                else:
                    redispatched_le_ins = []
                    for expert_id in range(LE):
                        redispatched_le_ins.append(expert_ins[expert_id][0])
                    return redispatched_le_ins, recved_cntss[0]

        device = f"cuda({local_rank})"
        partitioned_model = TestModel(n_partitions, device)
        reference_model = TestModel(1, device)

        partitioned_model.to(device=device)
        reference_model.to(device=device)

        x = np.random.randn(S_before_partition, M).astype("float32")
        gate_w = np.random.randn(M, E).astype("float32")

        x = raf.array(x, device=device)
        gate_w = raf.array(gate_w, device=device)

        redispatched_le_ins = run_vm_model(partitioned_model, device, [x, gate_w])
        redispatched_le_np = []
        for le in range(LE):
            redispatched_le_np.append(redispatched_le_ins[le].numpy())
        reference_out = run_vm_model(reference_model, device, [x, gate_w])
        r_redispatched_le_ins = [reference_out[x] for x in range(LE)]
        r_received_cnts = reference_out[LE]
        r_redispatched_le_np = []
        for le in range(LE):
            r_redispatched_le_np.append(r_redispatched_le_ins[le].numpy())
        r_received_cnts_np = (r_received_cnts.numpy() / M).astype("int32")

        # reshape back
        for le in range(LE):
            redispatched_le_np[le] = redispatched_le_np[le].reshape((-1, total_rank, M))
            r_redispatched_le_np[le] = r_redispatched_le_np[le].reshape((-1, total_rank, M))

        # zero out unused capacity
        for g in range(total_rank):
            for le in range(LE):
                redispatched_le_np[le][r_received_cnts_np[g * LE + le]:, g, :] = 0
                r_redispatched_le_np[le][r_received_cnts_np[g * LE + le]:, g, :] = 0
        for le in range(LE):
            check(redispatched_le_np[le], r_redispatched_le_np[le], atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    # test_partitioned_moe(2)
    # test_partitioned_moe(4)
    test_partitioned_moe_encode_dist_merge(2)