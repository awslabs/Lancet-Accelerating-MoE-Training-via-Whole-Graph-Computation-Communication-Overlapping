# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use,invalid-name, protected-access, too-many-locals, too-many-branches
"""
Profile sparse matmul_nt op's performance on a single GPU.
"""
import pickle
import numpy as np

import raf
from raf.testing.utils import profile_vm_model
import argparse

def profile_full_matmul_nt(C, G, M, N, repeat=100, warmup=40, dtype="float32"):
    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w):
            x = raf.matmul_nt(x, w)
            return x

    model = TestModel()
    device = f"cuda(0)"
    model.to(device=device)
    vx = np.random.randn(C * G, M).astype(dtype)
    vw = np.random.randn(N, M).astype(dtype)
    vx = raf.array(vx, device=device)
    vw = raf.array(vw, device=device)
    exec_times = profile_vm_model(model, device, [vx, vw], number=repeat, warmup=warmup)
    return np.mean(exec_times)

def profile_expert_matmul(C, G, M, N, send_counts, repeat=100, warmup=40, dtype="float32"):

    class TestModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, w, send_counts):
            x = raf.sparse_expert_matmul_nt(x, w, send_counts, 0, 1, 1)
            return x


    model = TestModel()
    device = f"cuda(0)"
    model.to(device=device)
    vx = np.random.randn(C, G, M).astype(dtype)
    vw = np.random.randn(N, M).astype(dtype)
    vsend_counts = np.array(send_counts).astype("uint64")
    vx = raf.array(vx, device=device)
    vw = raf.array(vw, device=device)
    vsend_counts = raf.array(vsend_counts, device=device)
    exec_times = profile_vm_model(model, device, [vx, vw, vsend_counts], number=repeat, warmup=warmup)
    return np.mean(exec_times)

# def profile_alltoallv(shape_per_rank=(1024, 1024), send_count=None, repeat=100, warmup=40, dtype="float16", verbose=True):
#     class TestModel(raf.Model):
#         def build(self):
#             pass

#         @raf.model.trace
#         def forward(self, x, send_cnts):
#             return raf._all_to_allv([x], [send_cnts])

#     if raf.build.with_nccl() < 20700:
#         print("alltoall is not supported in NCCL < 2.10")
#         return

#     model = TestModel()
#     total_rank, rank, local_rank = get_dist_info(verbose=True)
#     device = f"cuda({local_rank})"
#     if send_count is None:
#         # one chunk per device, full send
#         send_count = [shape_per_rank[0] * shape_per_rank[1]] * total_rank
#     assert len(send_count) % total_rank == 0, "length of send_count should be a multiple of total_rank"
#     n_chunk_per_rank = len(send_count) // total_rank
#     assert shape_per_rank[0] % n_chunk_per_rank == 0, "shape_per_rank[0] should be a multiple of n_chunk_per_rank"
#     dim0_size_per_chunk = shape_per_rank[0] // n_chunk_per_rank
#     x = np.ones(shape=(total_rank * dim0_size_per_chunk, shape_per_rank[1]), dtype=dtype) * (rank + 1)
#     send_cnts = np.array(send_count, dtype="uint64")
#     x = raf.array(x, device=device)
#     send_cnts = raf.array(send_cnts, device=device)
#     model.to(device=device)
#     exec_times = profile_vm_model(model, device, [x, send_cnts], number=repeat, warmup=warmup)
#     if verbose:
#         print(f"Time per iter: {np.mean(exec_times)} ms")
#     return np.mean(exec_times)

# def gen_random_vectors_sum_to_zero(n, d, scale=1.0, dist="normal"):
#     # 1. generate n linearly independent vectors in the subspace
#     vs = []
#     for i in range(d-1):
#         v = np.zeros(d)
#         v[i] = 1
#         v[-1] = -1
#         vs.append(v)
#     vs = np.array(vs).T
#     # 2. find the set of orthonormal basis
#     q, r = np.linalg.qr(vs)
#     q = q.T
#     # 3. sample from multivariate gaussian
#     if dist == "normal":
#         x = np.random.normal(0, scale=scale, size=(n, d-1))
#     else:
#         assert dist == "uniform"
#         x = np.random.uniform(-scale, scale, size=(n, d-1))
#     # 4. project to the subspace
#     y = np.dot(x, q)
#     return y

# def profile_alltoallv_distribution(shape_per_rank=(1024, 1024), dist="normal", avg_load=0.5, n_chunks=1, n_exps=1000, repeat=100, warmup=40, out_file=None, dtype="float16"):
#     max_send_row_per_chunk = shape_per_rank[0] // n_chunks
#     total_rank, rank, local_rank = get_dist_info(verbose=True)
#     total_chunks = total_rank * n_chunks
#     # generate random send counts
#     if dist == "normal":
#         scale = 0.4
#     else:
#         scale = 1
#     noise = gen_random_vectors_sum_to_zero(n_exps, total_chunks, scale=scale, dist=dist)
#     noise *= max_send_row_per_chunk * avg_load
#     noise = np.clip(noise, -int(avg_load*max_send_row_per_chunk), int((1-avg_load)*max_send_row_per_chunk))

#     result_dict = {}
#     for row in noise:
#         # row is a vector of length total_chunks
#         send_counts = np.array([int(max_send_row_per_chunk * avg_load)] * total_chunks) + row
#         send_counts = np.clip(send_counts, 1, max_send_row_per_chunk)
#         send_counts = send_counts.astype("uint64")
#         send_counts *= shape_per_rank[1]
#         result_dict[tuple(send_counts.tolist())] = profile_alltoallv(shape_per_rank=shape_per_rank, send_count=send_counts, repeat=repeat, warmup=warmup, dtype=dtype, verbose=False)
#     if out_file:
#         with open(out_file, "wb") as f:
#             pickle.dump((result_dict, shape_per_rank, avg_load, n_chunks, total_rank), f)

# somehow the test will fail if run consecutively
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--profile_alltoall", action="store_true")
    # parser.add_argument("--profile_alltoallv", action="store_true")
    # parser.add_argument("--profile_alltoallv_dist", action="store_true")
    # parser.add_argument("--out_file", type=str, default=None)
    # parser.add_argument("--shape_per_rank", type=int, nargs=2, default=(512, 768))
    # args = parser.parse_args()
    # if args.profile_alltoall:
    #     profile_alltoall(shape_per_rank=args.shape_per_rank)
    # elif args.profile_alltoallv:
    #     profile_alltoallv(shape_per_rank=args.shape_per_rank)
    # elif args.profile_alltoallv_dist:
    #     assert args.out_file is not None
    #     profile_alltoallv_distribution(shape_per_rank=args.shape_per_rank, out_file=args.out_file)
    # C = 1024
    # G = 64
    # M = 1024
    # load_factor = 0.5
    # expert_time = profile_expert_matmul(C, G, M, 4 * M, [int(C * load_factor * M)] * G)
    # full_time = profile_full_matmul_nt(C, G, M, 4 * M)
    # print(f"expert_time: {expert_time} ms, full_time: {full_time} ms, theoretical best: {full_time * load_factor} ms")
    with open("./profile_sparse_expert_matmul_nt.csv", "w") as f:
        f.write("C,G,M,load_factor,expert_time,full_time\n")
    for C in [512, 768, 1024, 2048]:
        for G in [16, 32, 64, 128]:
            M = 768
            for load_factor in [0.1, 0.2, 0.3, 0.4, 0.5]:
                expert_time = profile_expert_matmul(C, G, M, 4 * M, [int(C * load_factor * M)] * G)
                full_time = profile_full_matmul_nt(C, G, M, 4 * M)
                with open("./profile_sparse_expert_matmul_nt.csv", "a") as f:
                    f.write(f"{C},{G},{M},{load_factor},{expert_time},{full_time}\n")