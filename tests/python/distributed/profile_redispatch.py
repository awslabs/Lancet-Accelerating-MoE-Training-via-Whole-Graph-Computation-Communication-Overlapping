# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
"""Test all-to-all communication operators in a cluster with 2 GPUs.
As pytest do not support mpirun, thus we skip this test in pytest progress.
To test all-to-all ops, you should run:
`mpirun -np 2 python3 tests/python/distributed/test_alltoall.py`
(in ci/tash_python_unittest.sh)
"""
import pickle
import numpy as np
from collections import defaultdict

import raf
from raf import distributed as dist
from raf.testing import get_dist_info
from raf.testing.utils import profile_vm_model

def generate_dummy_indloc_masks(E, C, n_part):
    # assume each token is assigned to a random expert
    indices = np.random.randint(0, E, size=(E * C,))
    locs = np.zeros_like(indices, dtype="int32")
    total_cnt_for_expert = defaultdict(int)
    for n in range(n_part):
        locs_per_expert = defaultdict(int)
        for i in range(E*C // n_part):
            global_loc = total_cnt_for_expert[indices[i]]
            local_loc = locs_per_expert[indices[i]]
            if global_loc >= C:
                # drop
                indices[i] = -1
                locs[i] = -1
            else:
                locs[i] = local_loc
                total_cnt_for_expert[indices[i]] += 1
                locs_per_expert[indices[i]] += 1
    global_masks = np.concatenate([np.expand_dims(indices, 0), np.expand_dims(locs, 0)], axis=0).astype("int32")
    print(global_masks.shape)
    return np.split(global_masks, n_part, axis=1)

def profile_redispatch(E, C, M, n_part, repeat=100, warmup=40):
    print("Testing redispatch performance.")

    in_shape = (E, C, M)

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, xs, indlocs):
            x = raf.moe_redispatch(xs, indlocs)
            return x

    model = TestModel()
    device = f"cuda(0)"
    model.to(device=device)

    xs = [np.random.rand(*in_shape).astype("float32") for _ in range(n_part)]
    indlocs = generate_dummy_indloc_masks(E, C, n_part)
    xs = [raf.array(x, device=device) for x in xs]
    indlocs = [raf.array(indloc, device=device) for indloc in indlocs]
    exec_times = profile_vm_model(model, device, [xs, indlocs], number=repeat, warmup=warmup)
    print(f"Time per iter: {np.mean(exec_times)} ms")


# somehow the test will fail if run consecutively
if __name__ == "__main__":
    profile_redispatch(32, 256, 768, 2)