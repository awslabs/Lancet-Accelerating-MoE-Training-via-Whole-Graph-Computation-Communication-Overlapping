#include "./kernel_util.cuh"

namespace raf {
namespace op {
namespace cuda {

#define thread_num 1024

#define INDEX_NELEMENTS(idx) ((idx) * nelements_stride + nelements_offset)

#define COPY_KERNEL_BODY(SRC_OFFSET_STMT, DST_OFFSET_STMT)                                                  \
    __shared__ uint64_t temp[thread_num + 1];                                                               \
    int thid = threadIdx.x, bid = blockIdx.x;                                                               \
    temp[thid] = 0;                                                                                         \
    uint64_t block_offset = 0;                                                                              \
    for (int S = 0; S < bid; S += thread_num) {                                                             \
        int offset = 1;                                                                                     \
        if (S + thid < bid) {                                                                               \
            temp[thid] = nelements_per_dim[INDEX_NELEMENTS(S + thid)];                                      \
        } else {                                                                                            \
            temp[thid] = 0;                                                                                 \
        }                                                                                                   \
        for (int d = thread_num >> 1; d > 0; d >>= 1) {                                                     \
                __syncthreads();                                                                            \
                if (thid < d)                                                                               \
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];             \
                offset *= 2;                                                                                \
        }                                                                                                   \
        __syncthreads();                                                                                    \
        block_offset += temp[thread_num - 1];                                                               \
    }                                                                                                       \
    block_offset *= scale_element_size_by;                                                                  \
    block_offset /= shrink_element_size_by;                                                                 \
    uint64_t n_elements_to_cpy = nelements_per_dim[INDEX_NELEMENTS(bid)]                                    \
                            * scale_element_size_by / shrink_element_size_by;                               \
    for (uint64_t S=0; S < n_elements_to_cpy; S += thread_num) {                                            \
        uint64_t thread_offset = S + thid;                                                                  \
        if (thread_offset < n_elements_to_cpy) {                                                            \
            dst[DST_OFFSET_STMT] = src[SRC_OFFSET_STMT];                                                    \
        }                                                                                                   \
    }

#define INDEX3D(idx0, idx1, idx2) ((idx0) * (dim1) * (dim2) + (idx1) * (dim2) + (idx2))

#define INDEX3Ddim0 (INDEX3D(bid, thread_offset / dim2, thread_offset % dim2))
#define INDEX3Ddim1 (INDEX3D(thread_offset / dim2, bid, thread_offset % dim2))

#define DEF_KERNEL_3D(DIMENSION, __dtype)                                                   \
extern "C" __global__ void memcpy_sparse_to_cont_3d_dim##DIMENSION##_kernel_##__dtype(      \
    __dtype* src, __dtype* dst, uint64_t* nelements_per_dim,                                \
    int nelements_stride, int nelements_offset, int axis,                                   \
    int dim0, int dim1, int dim2, int scale_element_size_by, int shrink_element_size_by) {  \
    COPY_KERNEL_BODY(INDEX3Ddim##DIMENSION, block_offset + thread_offset);                  \
}                                                                                           \
                                                                                            \
extern "C" __global__ void memcpy_cont_to_sparse_3d_dim##DIMENSION##_kernel_##__dtype(      \
    __dtype* src, __dtype* dst, uint64_t* nelements_per_dim,                                \
    int nelements_stride, int nelements_offset, int axis,                                   \
    int dim0, int dim1, int dim2, int scale_element_size_by, int shrink_element_size_by) {  \
    COPY_KERNEL_BODY(block_offset + thread_offset, INDEX3Ddim##DIMENSION);                  \
}                                                                                           \
                                                                                            \

#define LAUNCH_KERNEL_3D(KERN_TYPE, DIMENSION, __dtype) memcpy_##KERN_TYPE##_3d_dim##DIMENSION##_kernel_##__dtype<<<grid, block, 0, (cudaStream_t)stream>>>(src, dst, nelements_per_dim, nelements_stride, nelements_offset, axis, dim0, dim1, dim2, scale_element_size_by, shrink_element_size_by);

#define DEF_LAUNCH_KERNEL_3D(KERN_TYPE, __dtype)                                            \
void launch_memcpy_##KERN_TYPE##_3d(                                                        \
    __dtype* src, __dtype* dst, uint64_t* nelements_per_dim,                                \
    int nelements_stride, int nelements_offset, int axis,                                   \
    int dim0, int dim1, int dim2, void* stream,                                             \
    int scale_element_size_by, int shrink_element_size_by) {                                \
    int sparse_dim = (axis == 0) ? dim0 : (axis == 1) ? dim1 : dim2;                        \
    dim3 grid(sparse_dim);                                                                  \
    dim3 block(thread_num);                                                                 \
    switch (axis) {                                                                         \
    case 0:                                                                                 \
        LAUNCH_KERNEL_3D(KERN_TYPE, 0, __dtype);                                            \
        break;                                                                              \
    case 1:                                                                                 \
        LAUNCH_KERNEL_3D(KERN_TYPE, 1, __dtype);                                            \
        break;                                                                              \
    default:                                                                                \
        LOG(FATAL) << "Axis " << axis << " not supported.";                                 \
        break;                                                                              \
    }                                                                                       \
}                                                                                           \

DEF_KERNEL_3D(0, float);
DEF_KERNEL_3D(1, float);
DEF_KERNEL_3D(0, half);
DEF_KERNEL_3D(1, half);

DEF_LAUNCH_KERNEL_3D(cont_to_sparse, float);
DEF_LAUNCH_KERNEL_3D(sparse_to_cont, float);
DEF_LAUNCH_KERNEL_3D(cont_to_sparse, half);
DEF_LAUNCH_KERNEL_3D(sparse_to_cont, half);

} // namespace cuda
}  // namespace op
}  // namespace raf