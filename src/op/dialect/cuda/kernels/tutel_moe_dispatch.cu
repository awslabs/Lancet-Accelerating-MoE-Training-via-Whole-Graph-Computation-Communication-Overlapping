/*
    MIT License

    Copyright (c) Microsoft Corporation.
    Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
*/

/*!
 * \file src/op/dispatch/cuda/kernels/tutel_moe_dispatch.cu
 * \brief tutel moe fast dispatcher cuda kernel, modified from 
          https://github.com/microsoft/tutel/blob/v0.1.x/tutel/jit_kernels/sparse.py
 */

#include "./kernel_util.cuh"

namespace raf {
namespace op {
namespace cuda {


#define FORWARD_KERNEL_BODY(GATE_STATEMENT) for (int i = blockIdx.x; i < samples; i += gridDim.x) \
    if (locations1_s[i] < capacity && indices1_s[i] >= 0 && indices1_s[i] < n_experts) { \
        _Pragma("unroll") \
        for (int j = threadIdx.x; j < hidden; j += 1024) \
            atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], GATE_STATEMENT * reshaped_input[i * (hidden) + j]); \
    }

#define BACKWARD_DATA_KERNEL_BODY(GATE_STATEMENT, DTYPE_STR) for (int i = blockIdx.x; i < samples; i += gridDim.x) \
    if (locations1_s[i] < capacity && indices1_s[i] >= 0 && indices1_s[i] < n_experts) { \
        _Pragma("unroll") \
        for (int j = threadIdx.x; j < hidden; j += 1024) \
            grad_reshaped_input[i * hidden + j] = GATE_STATEMENT * dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j]; \
    } else { \
        _Pragma("unroll") \
        for (int j = threadIdx.x; j < hidden; j += 1024) \
            grad_reshaped_input[i * hidden + j] = DTYPE_STR(0); \
    }

__host__ __device__ __forceinline__ int CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

#define CREATE_ENCODE_FW_KERNEL(__dtype) \
extern "C" __global__ __launch_bounds__(1024) void encode_forward_kernel_##__dtype(int* __restrict__ indices1_s, int* __restrict__ locations1_s, int* __restrict__ capused1_e, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity, int n_experts) { \
    for (int i = blockIdx.x; i < samples; i += gridDim.x) { \
        if (indices1_s[i] >= 0 && indices1_s[i] < n_experts && locations1_s[i] < capacity && locations1_s[i] < (capacity - capused1_e[indices1_s[i]])) { \
            _Pragma("unroll") \
            for (int j = threadIdx.x; j < hidden; j += 1024) \
                atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], reshaped_input[i * (hidden) + j]); \
        } \
    } \
}

#define CREATE_ENCODE_FW_KERNEL_WO_USED_CAP(__dtype) \
extern "C" __global__ __launch_bounds__(1024) void encode_forward_kernel_wo_usedcap_##__dtype(int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity, int n_experts) { \
    for (int i = blockIdx.x; i < samples; i += gridDim.x) { \
        if (indices1_s[i] >= 0 && indices1_s[i] < n_experts && locations1_s[i] < capacity) { \
            _Pragma("unroll") \
            for (int j = threadIdx.x; j < hidden; j += 1024) \
                atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j], reshaped_input[i * (hidden) + j]); \
        } \
    } \
}

CREATE_ENCODE_FW_KERNEL(float);
CREATE_ENCODE_FW_KERNEL(half);
CREATE_ENCODE_FW_KERNEL_WO_USED_CAP(float);
CREATE_ENCODE_FW_KERNEL_WO_USED_CAP(half);

#define CREATE_ENCODE_BW_DATA_KERNEL(__dtype) \
extern "C" __global__ __launch_bounds__(1024) void encode_backward_data_kernel_##__dtype(int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity, int n_experts) { \
    BACKWARD_DATA_KERNEL_BODY(__dtype(1), __dtype); \
}

CREATE_ENCODE_BW_DATA_KERNEL(float);
CREATE_ENCODE_BW_DATA_KERNEL(half);

#define CREATE_ENCODE_BW_GATE_KERNEL(__dtype) \
extern "C" __global__ __launch_bounds__(32) void encode_backward_gate_kernel_##__dtype(__dtype* __restrict__ dy_gates1_s, int* __restrict__ indices1_s, __dtype* __restrict__ grad_gates, int samples, int n_experts) { \
    int block_offset = blockIdx.x * blockDim.x * 64; \
    for(int loc_idx = 0; loc_idx < 64; loc_idx ++) { \
        int s_idx = block_offset + loc_idx * blockDim.x + threadIdx.x; \
        if(s_idx < samples && indices1_s[s_idx] >= 0) { \
            grad_gates[n_experts * s_idx + indices1_s[s_idx]] = dy_gates1_s[s_idx]; \
        } \
    } \
}

CREATE_ENCODE_BW_GATE_KERNEL(float);
CREATE_ENCODE_BW_GATE_KERNEL(half);

# define CREATE_DECODE_FW_KERNEL(__dtype) \
extern "C" __global__ __launch_bounds__(1024) void decode_forward_kernel_##__dtype(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ grad_reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity, int n_experts) { \
    BACKWARD_DATA_KERNEL_BODY(gates1_s[i], __dtype); \
}

CREATE_DECODE_FW_KERNEL(float);
CREATE_DECODE_FW_KERNEL(half);

# define CREATE_DECODE_BW_DATA_KERNEL(__dtype) \
extern "C" __global__ __launch_bounds__(1024) void decode_backward_data_kernel_##__dtype(__dtype* __restrict__ gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity, int n_experts) { \
    FORWARD_KERNEL_BODY(gates1_s[i]); \
}

CREATE_DECODE_BW_DATA_KERNEL(float);
CREATE_DECODE_BW_DATA_KERNEL(half);

# define CREATE_DECODE_BW_GATE_KERNEL(__dtype) \
extern "C" __global__ __launch_bounds__(32) void decode_backward_gate_kernel_##__dtype(__dtype* __restrict__ grad_gates1_s, int* __restrict__ indices1_s, int* __restrict__ locations1_s, __dtype* __restrict__ reshaped_input, __dtype* __restrict__ dispatched_input, int samples, int hidden, int capacity) { \
    for (int index = blockIdx.x; index < samples; index += gridDim.x) { \
      if (locations1_s[index] >= capacity || locations1_s[index] < 0 || indices1_s[index] < 0) { \
        if (((int)threadIdx.x) == 0) \
          ((__dtype*)grad_gates1_s)[index] = 0; \
        continue; \
      } \
      int indice = indices1_s[index] * capacity + locations1_s[index]; \
      __dtype grad_gates1_s_rf = 0.000000e+00f; \
      for (int i = threadIdx.x; i < hidden; i += 32) \
        grad_gates1_s_rf += dispatched_input[indice * (hidden) + i] * reshaped_input[index * (hidden) + i]; \
      __dtype red_buf0[1]; \
      unsigned int mask[1]; \
      __dtype t0[1]; \
      red_buf0[(0)] = grad_gates1_s_rf; \
      mask[(0)] = __activemask(); \
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32); \
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]); \
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32); \
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]); \
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32); \
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]); \
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32); \
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]); \
      t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32); \
      red_buf0[(0)] = (red_buf0[(0)] + t0[(0)]); \
      red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32); \
      if (((int)threadIdx.x) == 0) \
        ((__dtype*)grad_gates1_s)[index] = red_buf0[(0)]; \
    } \
}

CREATE_DECODE_BW_GATE_KERNEL(float);
CREATE_DECODE_BW_GATE_KERNEL(half);

#define thread_num  1024

extern "C" __global__ void gen_location_kernel(int* indices /* (dim_S) */, int* in_capused /* (dim_E) */, int* out_location /* (dim_S) */, int* out_capused /* (dim_E) */, uint64_t* out_elements_per_expert /* (dim_E) */, int capacity, int dim_S, int dim_M) {
    // [thread_extent] blockIdx.x = dim_E
    // [thread_extent] threadIdx.x = 1024
    __shared__ int temp[thread_num + 1];
    int thid = threadIdx.x, bid = blockIdx.x;
    int last_sum = -1;
    for (int S = 0; S < dim_S; S += thread_num) {
        int offset = 1;
        if (S + thid < dim_S) {
            temp[thid] = indices[S + thid] == bid ? 1 : 0;
        } else {
            temp[thid] = 0;
        }
        for (int d = thread_num >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d)
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                offset *= 2;
        }
        if (thid == 0)
                temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
        for (int d = 1; d < thread_num; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();
        if (S + thid < dim_S && indices[S+thid] == bid) {
            out_location[S+thid] = temp[thid + 1] + last_sum;
            if(out_location[S+thid] >= (capacity - in_capused[bid])) {
                indices[S+thid] = -1;
            }
        }
        __syncthreads();
        last_sum += temp[thread_num];
    }
    if (thid == 0) {
        out_capused[bid] = in_capused[bid] + last_sum + 1;
        if (out_capused[bid] > capacity)
                out_capused[bid] = capacity;
        out_elements_per_expert[bid] = (out_capused[bid] - in_capused[bid]) * dim_M;
    }
}

extern "C" __global__ void merge_masks_kernel(int** dev_in_indices_ptrs, int** dev_in_locations_ptrs,
                                             int* recon_indices1_s, int* recon_locations1_s,
                                             int dim_S_, int n_partition) {
    // [thread_extent] blockIdx.x = dim_E_
    // [thread_extent] threadIdx.x = 1024
    __shared__ int temp[thread_num + 1];
    int thid = threadIdx.x, bid = blockIdx.x;

    int last_sum = 0;
    for (int partition_idx=0; partition_idx<n_partition; partition_idx++) {
        int* dev_locations1 = dev_in_locations_ptrs[partition_idx];
        int* dev_indices1 = dev_in_indices_ptrs[partition_idx];
        // copy data from dev_locations1 to recon_locations1_s
        for (int S = 0; S < dim_S_; S += thread_num) {
            if (S + thid < dim_S_ && dev_indices1[S + thid] == bid) {
                recon_locations1_s[partition_idx * dim_S_ + S + thid] = dev_locations1[S + thid] + last_sum;
                recon_indices1_s[partition_idx * dim_S_ + S + thid] = dev_indices1[S + thid];
            }
            if (bid == 0 && dev_indices1[S + thid] == -1) {
                recon_indices1_s[partition_idx * dim_S_ + S + thid] = -1;
                recon_locations1_s[partition_idx * dim_S_ + S + thid] = -1;
            }
        }
        // parallel summation over dev_indices1 to update last_sum
        for (int S = 0; S < dim_S_; S += thread_num) {
            int offset = 1;
            if (S + thid < dim_S_) {
                temp[thid] = dev_indices1[S + thid] == bid ? 1 : 0;
            } else {
                temp[thid] = 0;
            }
            for (int d = thread_num >> 1; d > 0; d >>= 1) {
                    __syncthreads();
                    if (thid < d)
                            temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                    offset *= 2;
            }
            __syncthreads();
            last_sum += temp[thread_num - 1];
        }
    }
}

#define GET_EXP_IDX_SORTED(idx) (indices[sorted_indices[(idx)]])

extern "C" __global__ void gen_location_bpr_kernel(
    int* indices /* (dim_orig_S) */,
    int* sorted_indices /* (dim_orig_S) */,
    int* output_locations1_s /* (dim_S) */,
    bool* dropping_mask_scratchpad /* (dim_orig_S) */,
    int* out_capused /* (dim_E) */,
    uint64_t* out_elements_per_expert /* (dim_E) */,
    int capacity, int dim_orig_S, int dim_M,
    int n_partitions, int partition_id
    ) {
    // [thread_extent] blockIdx.x = dim_E
    // [thread_extent] threadIdx.x = 1024
    __shared__ int temp[thread_num + 1];
    int dim_S = dim_orig_S / n_partitions;
    int thid = threadIdx.x, bid = blockIdx.x;
    int last_sum = -1;
    // prefix sum over indices
    for (int S = 0; S < dim_orig_S; S += thread_num) {
        int offset = 1;
        if (S + thid < dim_orig_S) {
            temp[thid] = GET_EXP_IDX_SORTED(S + thid) == bid ? 1 : 0;
        } else {
            temp[thid] = 0;
        }
        for (int d = thread_num >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d)
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                offset *= 2;
        }
        if (thid == 0)
                temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
        for (int d = 1; d < thread_num; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();
        if ((S + thid < dim_orig_S) && GET_EXP_IDX_SORTED(S+thid) == bid) {
            int accum_loc = temp[thid + 1] + last_sum;
            if(accum_loc >= capacity) {
                dropping_mask_scratchpad[sorted_indices[S+thid]] = false;
            } else {
                dropping_mask_scratchpad[sorted_indices[S+thid]] = true;
            }
        }
        __syncthreads();
        last_sum += temp[thread_num];
    }
    // now generate locations based on dropping mask
    // while computing the locations, we skip indices that do not belong to
    // the current partition
    last_sum = -1;
    int partition_offset = dim_S * partition_id;
    for (int S = 0; S < dim_S; S += thread_num) {
        int offset = 1;
        if (S + thid < dim_S) {
            temp[thid] = (indices[partition_offset + S + thid] == bid && dropping_mask_scratchpad[partition_offset + S + thid]) ? 1 : 0;
        } else {
            temp[thid] = 0;
        }
        for (int d = thread_num >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d)
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                offset *= 2;
        }
        if (thid == 0)
                temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
        for (int d = 1; d < thread_num; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();
        if (S + thid < dim_S && indices[partition_offset + S + thid] == bid) {
            if (dropping_mask_scratchpad[partition_offset + S + thid]) {
                output_locations1_s[S+thid] = temp[thid + 1] + last_sum;
            } else {
                output_locations1_s[S+thid] = -1;
                indices[partition_offset + S + thid] = -1;
            }
        }
        __syncthreads();
        last_sum += temp[thread_num];
    }
    if (thid == 0) {
        out_capused[bid] = last_sum + 1;
        if (out_capused[bid] > capacity)
            out_capused[bid] = capacity;
        out_elements_per_expert[bid] = (out_capused[bid]) * dim_M;
    }
}

#define CREATE_REDISPATCH_KERNEL(__dtype) \
extern "C" __global__ void redispatch_kernel_##__dtype(__dtype** dev_partitioned_data_ptrs, int** dev_indices1_ptrs, \
                                             __dtype* redispatched_data, \
                                             int dim_S_, int dim_M_, int dim_C_, int dim_E_, int n_partition) { \
    __shared__ int temp[thread_num + 1]; \
    int thid = threadIdx.x, bid = blockIdx.x; \
 \
    int last_sum = 0; \
 \
    for (int partition_idx=0; partition_idx<n_partition; partition_idx++) { \
        int* dev_indices1 = dev_indices1_ptrs[partition_idx]; \
        int curr_sum = 0; \
        for (int S = 0; S < dim_S_; S += thread_num) { \
            int offset = 1; \
            if (S + thid < dim_S_) { \
                temp[thid] = dev_indices1[S + thid] == bid ? 1 : 0; \
            } else { \
                temp[thid] = 0; \
            } \
            for (int d = thread_num >> 1; d > 0; d >>= 1) { \
                    __syncthreads(); \
                    if (thid < d) \
                            temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1]; \
                    offset *= 2; \
            } \
            __syncthreads(); \
            curr_sum += temp[thread_num - 1]; \
        } \
        __dtype* partitioned_data = dev_partitioned_data_ptrs[partition_idx]; \
        for (int SM = 0; SM < curr_sum * dim_M_; SM += thread_num) { \
            if (SM + thid < curr_sum * dim_M_) { \
                int src_index = bid * (dim_C_ * dim_M_) + SM + thid; \
                int target_index = bid * (dim_C_ * dim_M_) + last_sum * dim_M_ + SM + thid; \
                if (target_index < dim_E_ * dim_C_ * dim_M_ && src_index < dim_E_ * dim_C_ * dim_M_) { \
                    redispatched_data[target_index] = partitioned_data[src_index]; \
                } \
            } \
        } \
        last_sum += curr_sum; \
    } \
}

CREATE_REDISPATCH_KERNEL(float);
CREATE_REDISPATCH_KERNEL(half);

#define CREATE_REDISPATCH_EXPERT_INPUT_GCM_KERNEL(__dtype) \
extern "C" __global__ void redispatch_expert_input_kernel_GCM_##__dtype(__dtype** dev_partitioned_data_ptrs, \
                      uint64_t** dev_in_mask_ptrs, __dtype* redispatched_data, int local_expert_id, \
                      int dim_LE_, int dim_C_, int dim_M_, int dim_G_, int n_partition) { \
    int thid = threadIdx.x, bid = blockIdx.x; \
 \
    __dtype* redispatched_target = redispatched_data + dim_C_ * dim_M_ * bid; \
 \
    int last_sum = 0; \
    for (int partition_idx=0; partition_idx<n_partition; partition_idx++) { \
        uint64_t recv_cnt = dev_in_mask_ptrs[partition_idx][dim_LE_ * bid + local_expert_id]; \
        __dtype* dev_data = dev_partitioned_data_ptrs[partition_idx] + dim_C_ * dim_M_ * bid; \
        for (int S = 0; S < recv_cnt; S += thread_num) { \
            if(S + thid < recv_cnt) { \
                redispatched_target[last_sum + S + thid] = dev_data[S + thid]; \
            } \
        } \
        last_sum += recv_cnt; \
    } \
}

CREATE_REDISPATCH_EXPERT_INPUT_GCM_KERNEL(float);
CREATE_REDISPATCH_EXPERT_INPUT_GCM_KERNEL(half);

#define GETIDXCGM(idx_C, idx_G, idx_M) (idx_C * dim_G_ * dim_M_ + idx_G * dim_M_ + idx_M)

#define CREATE_REDISPATCH_EXPERT_INPUT_CGM_KERNEL(__dtype) \
extern "C" __global__ void redispatch_expert_input_kernel_CGM_##__dtype(__dtype** dev_partitioned_data_ptrs, \
                      uint64_t** dev_in_mask_ptrs, __dtype* redispatched_data, int local_expert_id, \
                      int dim_LE_, int dim_C_, int dim_M_, int dim_G_, int n_partition, int recv_cnts_scale) { \
    int thid = threadIdx.x, bid = blockIdx.x; \
    uint64_t last_sum = 0; \
    for (int partition_idx=0; partition_idx<n_partition; partition_idx++) { \
        uint64_t recv_cnt = dev_in_mask_ptrs[partition_idx][dim_LE_ * bid + local_expert_id] * recv_cnts_scale; \
        __dtype* dev_data = dev_partitioned_data_ptrs[partition_idx]; \
        for (uint64_t S = 0; S < recv_cnt; S += thread_num) { \
            int unoffseted_idx_C = (S + thid) / dim_M_; \
            int unoffseted_idx_M = (S + thid) % dim_M_; \
            int offseted_idx_C = (S + thid + last_sum) / dim_M_; \
            int offseted_idx_M = (S + thid + last_sum) % dim_M_; \
            if(S + thid < recv_cnt) { \
                redispatched_data[GETIDXCGM(offseted_idx_C, bid, offseted_idx_M)] = dev_data[GETIDXCGM(unoffseted_idx_C, bid, unoffseted_idx_M)]; \
            } \
        } \
        last_sum += recv_cnt; \
    } \
}

CREATE_REDISPATCH_EXPERT_INPUT_CGM_KERNEL(float);
CREATE_REDISPATCH_EXPERT_INPUT_CGM_KERNEL(half);

#define CREATE_LAUNCH_ENCODE_FW(__dtype) \
void launch_encode_forward(int* indices1_s, int* locations1_s, int* capused1_e, __dtype* reshaped_input, __dtype* dispatched_input, \
                           int samples, int hidden, int capacity, int experts, void* stream) { \
  dim3 grid(512); \
  dim3 block(1024); \
  CUDA_CALL(cudaMemsetAsync(dispatched_input, 0, experts * capacity * hidden * sizeof(__dtype), static_cast<cudaStream_t>(stream))); \
  encode_forward_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      indices1_s, locations1_s, capused1_e, reshaped_input, dispatched_input, samples, hidden, capacity, experts); \
}

#define CREATE_LAUNCH_ENCODE_FW_WO_USED_CAP(__dtype) \
void launch_encode_forward(int* indices1_s, int* locations1_s, __dtype* reshaped_input, __dtype* dispatched_input, \
                           int samples, int hidden, int capacity, int experts, void* stream) { \
  dim3 grid(512); \
  dim3 block(1024); \
  CUDA_CALL(cudaMemsetAsync(dispatched_input, 0, experts * capacity * hidden * sizeof(__dtype), static_cast<cudaStream_t>(stream))); \
  encode_forward_kernel_wo_usedcap_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      indices1_s, locations1_s, reshaped_input, dispatched_input, samples, hidden, capacity, experts); \
}

CREATE_LAUNCH_ENCODE_FW(float);
CREATE_LAUNCH_ENCODE_FW(half);
CREATE_LAUNCH_ENCODE_FW_WO_USED_CAP(float);
CREATE_LAUNCH_ENCODE_FW_WO_USED_CAP(half);

#define CREATE_LAUNCH_ENCODE_BW_DATA(__dtype) \
void launch_encode_backward_data(int* indices1_s, int* locations1_s, __dtype* grad_reshaped_input, __dtype* dy_dispatched_input, \
                           int samples, int hidden, int capacity, int experts, void* stream) { \
  dim3 grid(512); \
  dim3 block(1024); \
  encode_backward_data_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      indices1_s, locations1_s, grad_reshaped_input, dy_dispatched_input, samples, hidden, capacity, experts); \
}

CREATE_LAUNCH_ENCODE_BW_DATA(float);
CREATE_LAUNCH_ENCODE_BW_DATA(half);

#define CREATE_LAUNCH_ENCODE_BW_GATE(__dtype) \
void launch_encode_backward_gate(__dtype* dy_gates1_s, int* indices1_s, __dtype* grad_gates, \
                                 int samples, int n_experts, void* stream) { \
  dim3 grid(CeilDiv(samples, 32*64)); \
  dim3 block(32); \
  CUDA_CALL(cudaMemsetAsync(grad_gates, 0, samples * n_experts * sizeof(__dtype), static_cast<cudaStream_t>(stream))); \
  encode_backward_gate_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      dy_gates1_s, indices1_s, grad_gates, samples, n_experts); \
}

CREATE_LAUNCH_ENCODE_BW_GATE(float);
CREATE_LAUNCH_ENCODE_BW_GATE(half);

#define CREATE_LAUNCH_DECODE_FW(__dtype) \
void launch_decode_forward(__dtype* gates1_s, int* indices1_s, int* locations1_s, __dtype* decoded_output, __dtype* expert_output, \
                           int samples, int hidden, int capacity, int experts, void* stream) { \
  dim3 grid(512); \
  dim3 block(1024); \
  decode_forward_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      gates1_s, indices1_s, locations1_s, decoded_output, expert_output, samples, hidden, capacity, experts); \
}

CREATE_LAUNCH_DECODE_FW(float);
CREATE_LAUNCH_DECODE_FW(half);

#define CREATE_LAUNCH_DECODE_BW_DATA(__dtype) \
void launch_decode_backward_data(__dtype* gates1_s, int* indices1_s, int* locations1_s, __dtype* dy_combined_output, __dtype* grad_expert_output, \
                           int samples, int hidden, int capacity, int experts, void* stream) { \
  dim3 grid(512); \
  dim3 block(1024); \
  CUDA_CALL(cudaMemsetAsync(grad_expert_output, 0, experts * capacity * hidden * sizeof(__dtype), static_cast<cudaStream_t>(stream))); \
  decode_backward_data_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      gates1_s, indices1_s, locations1_s, dy_combined_output, grad_expert_output, samples, hidden, capacity, experts); \
}

CREATE_LAUNCH_DECODE_BW_DATA(float);
CREATE_LAUNCH_DECODE_BW_DATA(half);

#define CREATE_LAUNCH_DECODE_BW_GATE(__dtype) \
void launch_decode_backward_gate(__dtype* grad_gates1_s, int* indices1_s, int* locations1_s, __dtype* dy_combined_output, __dtype* expert_output, \
                           int samples, int hidden, int capacity, void* stream) { \
  dim3 grid(512); \
  dim3 block(32); \
  decode_backward_gate_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
      grad_gates1_s, indices1_s, locations1_s, dy_combined_output, expert_output, samples, hidden, capacity); \
}

CREATE_LAUNCH_DECODE_BW_GATE(float);
CREATE_LAUNCH_DECODE_BW_GATE(half);

void launch_gen_location(int* indices1_s, int* input_capused1_e, int* output_locations1_s, int* output_capused1_e, uint64_t* out_elements_per_expert, int capacity, int samples, int experts, int model_dim, void* stream) {
  dim3 grid(experts);
  dim3 block(1024);
  gen_location_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      indices1_s, input_capused1_e, output_locations1_s, output_capused1_e, out_elements_per_expert, capacity, samples, model_dim);
}

void launch_merge_masks(int** dev_in_indices_ptrs, int** dev_in_locations_ptrs, int* recon_indices1_s, int* recon_locations1_s,
                      int dim_S_, int dim_E_, int n_partition, void* stream) {
    dim3 grid(dim_E_);
    dim3 block(1024);
    merge_masks_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        dev_in_indices_ptrs, dev_in_locations_ptrs,
        recon_indices1_s, recon_locations1_s,
        dim_S_, n_partition);
}

void launch_gen_location_bpr(int* indices1_s, int* sorted_indices_s, int* output_locations1_s, bool* dropping_mask_scratchpad, int* output_capused1_e, uint64_t* out_elements_per_expert, int capacity, int samples_before_partition, int experts, int model_dim, int n_partitions, int partition_id, void* stream) {
    dim3 grid(experts);
    dim3 block(1024);
    gen_location_bpr_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        indices1_s, sorted_indices_s, output_locations1_s, dropping_mask_scratchpad, output_capused1_e, out_elements_per_expert, capacity, samples_before_partition, model_dim, n_partitions, partition_id);
}

#define CREATE_LAUNCH_REDISPATCH(__dtype) \
void launch_redispatch(__dtype** dev_partitioned_data_ptrs, \
                      int** dev_in_indices_ptrs, __dtype* redispatched_data, \
                      int dim_S_, int dim_M_, int dim_C_, int dim_E_, int n_partition, void* stream) { \
    dim3 grid(dim_E_); \
    dim3 block(1024); \
    redispatch_kernel_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
        dev_partitioned_data_ptrs, dev_in_indices_ptrs, redispatched_data, \
        dim_S_, dim_M_, dim_C_, dim_E_, n_partition); \
}

CREATE_LAUNCH_REDISPATCH(float);
CREATE_LAUNCH_REDISPATCH(half);

#define CREATE_LAUNCH_REDISPATCH_EXPERT_INPUT(__dtype) \
void launch_redispatch_expert_input(__dtype** dev_partitioned_data_ptrs, \
                      uint64_t** dev_in_mask_ptrs, __dtype* redispatched_data, int local_expert_id, \
                      int dim_LE_, int dim_C_, int dim_M_, int dim_G_, int n_partition, int recv_cnts_scale, void* stream) { \
    dim3 grid(dim_G_); \
    dim3 block(1024); \
    redispatch_expert_input_kernel_CGM_##__dtype<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>( \
        dev_partitioned_data_ptrs, dev_in_mask_ptrs, redispatched_data, local_expert_id, \
        dim_LE_, dim_C_, dim_M_, dim_G_, n_partition, recv_cnts_scale); \
}

CREATE_LAUNCH_REDISPATCH_EXPERT_INPUT(float);
CREATE_LAUNCH_REDISPATCH_EXPERT_INPUT(half);

}  // namespace cuda
}  // namespace op
}  // namespace raf

