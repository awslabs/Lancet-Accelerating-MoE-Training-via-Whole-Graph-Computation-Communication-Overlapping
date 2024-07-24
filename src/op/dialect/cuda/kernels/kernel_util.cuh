/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dispatch/cuda/kernel/kernel_utils.h
 * \brief Helper functions for CUDA kernels
 */
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <vector>
#include "../../../../common/cuda_utils.h"

namespace raf {
namespace op {
namespace cuda {

template <typename scalar_t, typename accscalar_t>
void embedding_dense_backward_cuda(const scalar_t* grad, accscalar_t* output,
                                   const int64_t* indices, int num, int range,
                                   int stride, void* stream, int64_t element);

template <typename T>
void multi_tensor_lans_cuda(int chunk_size, std::vector<T*> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int bias_correction, const float bias_correction1,
                            const float bias_correction2, const float beta3,
                            const float weight_decay, const int grad_averaging, const int mode,
                            const bool normalize_grad, const std::vector<int> numels, void* stream,
                            float* output_per_tensor, float* grad_norm_tensor,
                            float* param_norm_tensor, float* update_m_norm, float* q_norm_tensor,
                            int max_chunks_per_tensor);

#define CREATE_MOE_LAUNCH_KERNEL_DEFS(__dtype) \
void launch_encode_forward(int* indices1_s, int* locations1_s, int* capused1_e, __dtype* reshaped_input, __dtype* dispatched_input, \
                           int samples, int hidden, int capacity, int experts, void* stream); \
void launch_encode_forward(int* indices1_s, int* locations1_s, __dtype* reshaped_input, __dtype* dispatched_input, \
                           int samples, int hidden, int capacity, int experts, void* stream); \
void launch_encode_backward_data(int* indices1_s, int* locations1_s, __dtype* grad_reshaped_input, __dtype* dy_dispatched_input, \
                           int samples, int hidden, int capacity, int experts, void* stream); \
void launch_encode_backward_gate(__dtype* dy_gates1_s, int* indices1_s, __dtype* grad_gates, \
                                 int samples, int n_experts, void* stream); \
void launch_decode_forward(__dtype* gates1_s, int* indices1_s, int* locations1_s, __dtype* decoded_output, __dtype* expert_output, \
                           int samples, int hidden, int capacity, int experts, void* stream); \
void launch_decode_backward_data(__dtype* gates1_s, int* indices1_s, int* locations1_s, __dtype* dy_combined_output, __dtype* grad_expert_output, \
                           int samples, int hidden, int capacity, int experts, void* stream); \
void launch_decode_backward_gate(__dtype* grad_gates1_s, int* indices1_s, int* locations1_s, __dtype* dy_combined_output, __dtype* expert_output, \
                           int samples, int hidden, int capacity, void* stream); \
void launch_redispatch(__dtype** dev_partitioned_data_ptrs, \
                      int** dev_in_indices_ptrs, __dtype* redispatched_data, \
                      int dim_S_, int dim_M_, int dim_C_, int dim_E_, int n_partition, void* stream); \
void launch_redispatch_expert_input(__dtype** dev_partitioned_data_ptrs, \
                      uint64_t** dev_in_mask_ptrs, __dtype* redispatched_data, int local_expert_id, \
                      int dim_LE_, int dim_C_, int dim_M_, int dim_G_, int n_partition, int recv_cnts_scale, void* stream);

CREATE_MOE_LAUNCH_KERNEL_DEFS(float);
CREATE_MOE_LAUNCH_KERNEL_DEFS(half);

void launch_gen_location(int* indices1_s, int* input_capused1_e, int* output_locations1_s, int* output_capused1_e, uint64_t* out_elements_per_expert, int capacity, int samples, int experts, int model_dim, void* stream);

void launch_merge_masks(int** dev_in_indices_ptrs, int** dev_in_locations_ptrs, int* recon_indices1_s, int* recon_locations1_s,
                      int dim_S_, int dim_E_, int n_partition, void* stream);

void launch_gen_location_bpr(int* indices1_s, int* sorted_indices_s, int* output_locations1_s, bool* dropping_mask_scratchpad, int* output_capused1_e, uint64_t* out_elements_per_expert, int capacity, int samples_before_partition, int experts, int model_dim, int n_partitions, int partition_id, void* stream);

#define CREATE_SPARSE_MEMCPY_DEFS(__dtype) \
void launch_memcpy_sparse_to_cont_3d( \
    __dtype* src, __dtype* dst, uint64_t* nelements_per_dim, \
    int nelements_stride, int nelements_offset, \
    int axis, int dim0, int dim1, int dim2, void* stream, \
    int scale_element_size_by, int shrink_element_size_by); \
 \
void launch_memcpy_cont_to_sparse_3d( \
    __dtype* src, __dtype* dst, uint64_t* nelements_per_dim, \
    int nelements_stride, int nelements_offset, \
    int axis, int dim0, int dim1, int dim2, void* stream, \
    int scale_element_size_by, int shrink_element_size_by);

CREATE_SPARSE_MEMCPY_DEFS(float);
CREATE_SPARSE_MEMCPY_DEFS(half);

}  // namespace cuda
}  // namespace op
}  // namespace raf

