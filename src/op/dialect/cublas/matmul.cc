/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cublas/cublas_utils.cc
 * \brief Helper functions for cuBLAS
 */
#include <cublas.h>
#include "raf/op.h"
#include "raf/device_api.h"
#include "raf/stream_pool.h"

#include "./cublas_utils.h"
#include "../../schema/ufunc.h"
#include "../../schema/moe.h"
#include "../cuda/kernels/kernel_util.cuh"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"
#include "../../../profiler/cuda/cuda_profiler.h"

namespace raf {
namespace op {
namespace cublas {
namespace manual {

using namespace raf::value;

void GemmImpl(DLTensor* a, bool transpose_a, DLTensor* b, bool transpose_b, DLTensor* c) {
  auto handle = CUBlasThreadEntry::ThreadLocal()->handle;

  cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  int m = c->shape[1];
  int n = c->shape[0];
  int k = b->shape[transb != CUBLAS_OP_N];

  int ldb = std::max(1, transpose_b ? k : m);
  int lda = std::max(1, transpose_a ? n : k);

  if (c->dtype.code == kDLFloat) {
    switch (c->dtype.bits) {
      case 16: {
        CUBLAS_CALL(cublasGemmEx(handle, transb, transa, m, n, k, const_addr<1>(CUDA_R_32F),
                                 b->data, cudaDataType_t(DType(b->dtype)), ldb, a->data,
                                 cudaDataType_t(DType(a->dtype)), lda, const_addr<0>(CUDA_R_32F),
                                 c->data, cudaDataType_t(DType(c->dtype)), m, CUDA_R_32F,
                                 CUBLAS_GEMM_DFALT_TENSOR_OP));
        return;
      }
      case 32: {
        CUBLAS_CALL(
            cublasSgemm(handle, transb, transa, m, n, k,
                        static_cast<const float*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<float*>(b->data), ldb, static_cast<float*>(a->data), lda,
                        static_cast<const float*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<float*>(c->data), m));
        return;
      }
      case 64: {
        CUBLAS_CALL(
            cublasDgemm(handle, transb, transa, m, n, k,
                        static_cast<const double*>(const_addr<1>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<double*>(b->data), ldb, static_cast<double*>(a->data), lda,
                        static_cast<const double*>(const_addr<0>(cudaDataType_t(DType(c->dtype)))),
                        static_cast<double*>(c->data), m));
        return;
      }
    }
  }
  CUBLAS_CALL(cublasGemmEx(
      handle, transb, transa, m, n, k, const_addr<1>(cudaDataType_t(DType(c->dtype))), b->data,
      cudaDataType_t(DType(b->dtype)), ldb, a->data, cudaDataType_t(DType(a->dtype)), lda,
      const_addr<0>(cudaDataType_t(DType(c->dtype))), c->data, cudaDataType_t(DType(c->dtype)), m,
      cudaDataType_t(DType(c->dtype)), CUBLAS_GEMM_DEFAULT));
}

template <bool transpose_a, bool transpose_b>
class MatmulImpl : public raf::op::OpEnv {
  std::string env_name_;

 public:
  explicit MatmulImpl(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.matmul");
    static auto fschema_index = op::GetOpAttr<op::FRAFSchemaFieldIndex>(op, "FRAFSchemaFieldIndex");
    this->arg_indices = {
        fschema_index("x1"),
        fschema_index("x2"),
    };
    auto args = cv->args.as<op::schema::BinaryArgs>();
    CHECK(args != nullptr);
    std::string op_name = "raf.op.cublas.matmul";
    if (transpose_a || transpose_b) {
      op_name += "_";
      op_name += (transpose_a) ? "t" : "n";
      op_name += (transpose_b) ? "t" : "n";
    }
    env_name_ = TruncateName(GetUniqueName(op_name));
  }

  std::string name() const override {
    return env_name_;
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::BinaryArgs>();
    std::string op_name = tvm::runtime::Downcast<value::OpValue>(cv->callee)->op->name;
    GemmImpl(args->x1, transpose_a, args->x2, transpose_b, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    DLTensor* x1 = ir::Downcast<TensorValue>(inputs[0]);
    DLTensor* x2 = ir::Downcast<TensorValue>(inputs[1]);
    DLTensor* out = ir::Downcast<TensorValue>(output);
    GemmImpl(x1, transpose_a, x2, transpose_b, out);
  }

  static OpEnv* make(const CallValues& cv) {
    return new MatmulImpl<transpose_a, transpose_b>(cv);
  }
};

using MatmulNN = MatmulImpl<false, false>;
using MatmulNT = MatmulImpl<false, true>;
using MatmulTN = MatmulImpl<true, false>;
using MatmulTT = MatmulImpl<true, true>;

inline std::vector<int64_t> GetShapeFromTensorValue(const Value& value) {
  ICHECK(value.defined());
  std::vector<int64_t> shape;
  if (const auto* tv = value.as<TensorValueObj>()) {
    DLTensor* tensor = GetRef<TensorValue>(tv);
    for (size_t i = 0; i < tensor->ndim; ++i) {
      shape.push_back(tensor->shape[i]);
    }
  } else {
    LOG(FATAL) << "Unsupported value type " << value;
  }
  return shape;
}

class SparseExpertMatmulNtImpl : public raf::op::OpEnv {
  std::string env_name_;

 public:
  explicit SparseExpertMatmulNtImpl(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.sparse_expert_matmul_nt");
    static auto fschema_index = op::GetOpAttr<op::FRAFSchemaFieldIndex>(op, "FRAFSchemaFieldIndex");
    this->arg_indices = {
        fschema_index("data"),
        fschema_index("weight"),
        fschema_index("nelements_per_gpu"),
    };
    auto args = cv->args.as<op::schema::SparseExpertMatmulNtArgs>();
    CHECK(args != nullptr);
    std::string op_name = "raf.op.cublas.sparse_expert_matmul_nt";
    env_name_ = TruncateName(GetUniqueName(op_name));
    nelements_stride_ = args->nelements_stride;
    nelements_offset_ = args->nelements_offset;
    nelements_scale_ = args->nelements_scale;

    std::vector<int64_t> data_shape = GetShapeFromTensorValue(args->data);      // [C, G, M]
    std::vector<int64_t> weight_shape = GetShapeFromTensorValue(args->weight);  // [N, M]
    std::vector<int64_t> nelements_per_gpu_shape = GetShapeFromTensorValue(args->nelements_per_gpu); // [G*nelements_stride_]

    CHECK_EQ(data_shape.size(), 3) << "Expected input data to be 3D.";
    CHECK_EQ(weight_shape.size(), 2) << "Expected input weight to be 2D.";
    CHECK_EQ(nelements_per_gpu_shape.size(), 1) << "Expected input nelements_per_gpu to be 1D.";

    const DLTensor* data_tensor = args->data;
    dev_ = data_tensor->device;

    CHECK_EQ(data_shape[2], weight_shape[1]) << "Input data and weight should have matching shape in dim 2 and 1, respectively.";
    dim_G_ = data_shape[1];
    dim_C_ = data_shape[0];
    dim_M_ = data_shape[2];
    dim_N_ = weight_shape[0];
    dim_nelements_ = nelements_per_gpu_shape[0];
    // CHECK_EQ(data_tensor->dtype.bits, 32) << "SparseExpertMatmulNt only supports float32 for now.";
    if (data_tensor->dtype.bits == 32) {
      input_bits_ = 32;
    } else if (data_tensor->dtype.bits == 16) {
      input_bits_ = 16;
    } else {
      LOG(FATAL) << "Only support datatype float32 and float16.";
    }
    CUDA_CALL(cudaEventCreateWithFlags(&sync_event_, cudaEventDisableTiming));
    // RequestStream(&memcpy_stream_, cv->device, stream_pool::StreamTagEnum::MemCpyCudaToCpu());
    // allocate scratch space. we need as many scratch space as the input size
    RequestWorkspace(&scratch_space_in_, dev_, dim_C_ * dim_G_ * dim_M_ * (input_bits_ / 8));
    RequestWorkspace(&scratch_space_out_, dev_, dim_C_ * dim_G_ * dim_N_ * (input_bits_ / 8));
    // we also need a scratch space on CPU to calculate total elements
    CUDA_CALL(cudaMallocHost(&host_nelements_per_gpu_, dim_nelements_ * sizeof(uint64_t)));
  }

  ~SparseExpertMatmulNtImpl() {
    CUDA_CALL(cudaFreeHost(host_nelements_per_gpu_));
  }

  std::string name() const override {
    return env_name_;
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<op::schema::SparseExpertMatmulNtArgs>();
    Execute({args->data, args->weight, args->nelements_per_gpu}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    // first launch the copy kernel to copy the data to the scratch space
    static auto cuda_device_api = device_api::DeviceAPI::Get(DevType::kCUDA());

    DLTensor* data = Downcast<TensorValue>(inputs[0]);
    DLTensor* weight = Downcast<TensorValue>(inputs[1]);
    DLTensor* nelements_per_gpu = Downcast<TensorValue>(inputs[2]);

    DLTensor* out = ir::Downcast<TensorValue>(output);

    // CUDA_CALL(cudaMemcpyAsync(host_nelements_per_gpu_, nelements_per_gpu->data, dim_nelements_ * sizeof(uint64_t), cudaMemcpyDeviceToHost, (cudaStream_t)cuda_device_api->GetStream()));
    // CUDA_CALL(cudaEventRecord(sync_event_, (cudaStream_t)cuda_device_api->GetStream()));
    // if (input_bits_ == 32) {
    //   cuda::launch_memcpy_sparse_to_cont_3d(reinterpret_cast<float*>(data->data),
    //                                         reinterpret_cast<float*>(scratch_space_in_),
    //                                         reinterpret_cast<uint64_t*>(nelements_per_gpu->data),
    //                                         nelements_stride_,
    //                                         nelements_offset_,
    //                                         1, dim_C_, dim_G_, dim_M_,
    //                                         (cudaStream_t)cuda_device_api->GetStream(),
    //                                         nelements_scale_, 1);
    // } else {
    //   cuda::launch_memcpy_sparse_to_cont_3d(reinterpret_cast<__half*>(data->data),
    //                                         reinterpret_cast<__half*>(scratch_space_in_),
    //                                         reinterpret_cast<uint64_t*>(nelements_per_gpu->data),
    //                                         nelements_stride_,
    //                                         nelements_offset_,
    //                                         1, dim_C_, dim_G_, dim_M_,
    //                                         (cudaStream_t)cuda_device_api->GetStream(),
    //                                         nelements_scale_, 1);
    // }
    // // CUDA_CALL(cudaStreamSynchronize((cudaStream_t)memcpy_stream_));
    // CUDA_CALL(cudaEventSynchronize(sync_event_));
    // uint64_t total_elements = 0;
    // for (int i = 0; i < dim_G_; ++i) {
    //   total_elements += (reinterpret_cast<uint64_t*>(host_nelements_per_gpu_)[nelements_offset_ + i * nelements_stride_] * nelements_scale_ / dim_M_);
    // }
    // then launch the gemm kernel
    auto handle = CUBlasThreadEntry::ThreadLocal()->handle;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    int m = dim_N_;
    // int n = total_elements;
    int n = dim_C_ * dim_G_;
    int k = dim_M_;

    int ldb = std::max(1, k);
    int lda = std::max(1, k);
    if (input_bits_ == 32) {
      CUBLAS_CALL(
        cublasSgemm(handle, transb, transa, m, n, k,
                    static_cast<const float*>(const_addr<1>(cudaDataType_t(DType(data->dtype)))),
                    // static_cast<float*>(weight->data), ldb, static_cast<float*>(scratch_space_in_), lda,
                    static_cast<float*>(weight->data), ldb, static_cast<float*>(data->data), lda,
                    static_cast<const float*>(const_addr<0>(cudaDataType_t(DType(data->dtype)))),
                    // static_cast<float*>(scratch_space_out_), m)
                    static_cast<float*>(out->data), m)
      );
    } else {
      CUBLAS_CALL(
        cublasGemmEx(handle, transb, transa, m, n, k, const_addr<1>(CUDA_R_32F),
                          // static_cast<__half*>(weight->data), cudaDataType_t(DType(data->dtype)), ldb, static_cast<__half*>(scratch_space_in_),
                          static_cast<__half*>(weight->data), cudaDataType_t(DType(data->dtype)), ldb, static_cast<__half*>(data->data),
                          cudaDataType_t(DType(data->dtype)), lda, const_addr<0>(CUDA_R_32F),
                          // static_cast<__half*>(scratch_space_out_), cudaDataType_t(DType(data->dtype)), m, CUDA_R_32F,
                          static_cast<__half*>(out->data), cudaDataType_t(DType(data->dtype)), m, CUDA_R_32F,
                          CUBLAS_GEMM_DFALT_TENSOR_OP)
      );
    }

    // then launch the copy kernel to copy the result back to the output
    // DLTensor* out = ir::Downcast<TensorValue>(output);
    // CUDA_CALL(cudaMemsetAsync(out->data, 0, dim_C_ * dim_G_ * dim_N_ * (out->dtype.bits / 8), (cudaStream_t)cuda_device_api->GetStream()));
    // if (input_bits_ == 32) {
    //   cuda::launch_memcpy_cont_to_sparse_3d(reinterpret_cast<float*>(scratch_space_out_),
    //                                       reinterpret_cast<float*>(out->data),
    //                                       reinterpret_cast<uint64_t*>(nelements_per_gpu->data),
    //                                       nelements_stride_,
    //                                       nelements_offset_,
    //                                       1, dim_C_, dim_G_, dim_N_,
    //                                       (cudaStream_t)cuda_device_api->GetStream(),
    //                                       nelements_scale_ * dim_N_, dim_M_);
    // } else {
    //   cuda::launch_memcpy_cont_to_sparse_3d(reinterpret_cast<__half*>(scratch_space_out_),
    //                                       reinterpret_cast<__half*>(out->data),
    //                                       reinterpret_cast<uint64_t*>(nelements_per_gpu->data),
    //                                       nelements_stride_,
    //                                       nelements_offset_,
    //                                       1, dim_C_, dim_G_, dim_N_,
    //                                       (cudaStream_t)cuda_device_api->GetStream(),
    //                                       nelements_scale_ * dim_N_, dim_M_);
    // }
  }

  static OpEnv* make(const CallValues& cv) {
    return new SparseExpertMatmulNtImpl(cv);
  }

private:
  int nelements_stride_;
  int nelements_offset_;
  int nelements_scale_;
  // void* memcpy_stream_;
  void* scratch_space_in_;
  void* scratch_space_out_;
  void* host_nelements_per_gpu_;
  int64_t dim_G_;
  int64_t dim_C_;
  int64_t dim_M_;
  int64_t dim_N_;
  int64_t dim_nelements_;
  int input_bits_;
  cudaEvent_t sync_event_;

  Device dev_;
};

RAF_REGISTER_DIALECT_OP(cublas, matmul, 15);
RAF_REGISTER_DIALECT_OP(cublas, matmul_nt, 15);
RAF_REGISTER_DIALECT_OP(cublas, matmul_tn, 15);
RAF_REGISTER_DIALECT_OP(cublas, matmul_tt, 15);
RAF_REGISTER_DIALECT_OP(cublas, dense, 15);
RAF_REGISTER_DIALECT_OP(cublas, sparse_expert_matmul_nt, 20);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul", MatmulNN::make);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul_nt", MatmulNT::make);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul_tn", MatmulTN::make);
RAF_OP_ENV_MAKER("raf.op.cublas.matmul_tt", MatmulTT::make);
RAF_OP_ENV_MAKER("raf.op.cublas.dense", MatmulNT::make);
RAF_OP_ENV_MAKER("raf.op.cublas.sparse_expert_matmul_nt", SparseExpertMatmulNtImpl::make);

}  // namespace manual
}  // namespace cublas
}  // namespace op
}  // namespace raf
