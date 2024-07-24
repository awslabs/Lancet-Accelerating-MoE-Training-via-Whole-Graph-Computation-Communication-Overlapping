/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/memory.cc
 * \brief Tensor fusion and defusion operators with asynchronous CUDA memory copy.
 */
#include <cuda_runtime.h>
#include <vector>
#include "raf/op_utils.h"
#include "raf/stream_pool.h"
#include "raf/device_api.h"
#include "../../schema/memory.h"
#include "../../schema/init.h"
#include "../../../common/shape_utils.h"
#include "../../../common/cuda_utils.h"

namespace raf {
namespace op {
namespace cuda {

using namespace raf::op::schema;
using raf::common::shape_utils::BytesCompactTensor;
using raf::stream_pool::StreamTagEnum;
using device_api::DeviceAPI;

class CudaFuseTensor : public raf::op::OpEnv {
  void* stream;
  std::vector<int64_t> tuple_sizes;

  explicit CudaFuseTensor(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.fuse_tensor");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("data")};
    RequestStream(&stream, cv->device, StreamTagEnum::MemCudaToCuda1());

    auto args = cv->args.as<FuseTensorArgs>();
    auto& tv = args->data;
    tuple_sizes.clear();
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      tuple_sizes.push_back(BytesCompactTensor(*x));
    }
  }

 public:
  ~CudaFuseTensor() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.fuse_tensor"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<FuseTensorArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->data.begin(), args->data.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    // Fuse Tensor
    DLTensor* out = output;
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t offset = 0;
    for (int i = 0; i < tv->fields.size(); ++i) {
      DLTensor* x = tv->fields[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(out->data) + offset;
      CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i],
                                cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
      offset += tuple_sizes[i];
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaFuseTensor(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cuda, fuse_tensor, 10);
RAF_OP_ENV_MAKER("raf.op.cuda.fuse_tensor", CudaFuseTensor::make);

class CudaFuseReorderTensor : public raf::op::OpEnv {
  void* stream;
  std::vector<int64_t> split_sizes;
  int n_parts;

  explicit CudaFuseReorderTensor(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.fuse_reorder_tensor");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("data")};
    RequestStream(&stream, cv->device, StreamTagEnum::MemCudaToCuda1());

    auto args = cv->args.as<FuseReorderTensorArgs>();
    n_parts = args->n_parts;
    auto& tv = args->data;
    split_sizes.clear();
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      int64_t nbytes = BytesCompactTensor(*x);
      CHECK(nbytes % n_parts == 0) << "Input tensor cannot be divided to " << n_parts << " parts.";
      split_sizes.push_back(nbytes / n_parts);
    }
  }

 public:
  ~CudaFuseReorderTensor() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.fuse_reorder_tensor"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<FuseReorderTensorArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->data.begin(), args->data.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    // Fuse and reorder tensors
    DLTensor* out = output;
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t offset = 0;
    for (int i = 0; i < n_parts; ++i) {
      for (int j = 0; j < tv->fields.size(); ++j) {
        DLTensor* x = tv->fields[j];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(out->data) + offset;
        void* input = reinterpret_cast<uint8_t*>(x->data) + i * split_sizes[j];
        CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, input, split_sizes[j],
                                  cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
        offset += split_sizes[j];
      }
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaFuseReorderTensor(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cuda, fuse_reorder_tensor, 10);
RAF_OP_ENV_MAKER("raf.op.cuda.fuse_reorder_tensor", CudaFuseReorderTensor::make);

class CudaCopyInplace : public raf::op::OpEnv {
  int64_t size;
  int64_t dst_offset;
  int64_t src_offset;

  explicit CudaCopyInplace(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.copy_inplace");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {
      fschema_index[op]("dst_data"), 
      fschema_index[op]("src_data")
    };
    auto args = cv->args.as<CopyInplaceArgs>();
    DLTensor* dst_data = args->dst_data;
    DLTensor* src_data = args->src_data;
    int64_t dst_size = BytesCompactTensor(*dst_data);
    int64_t src_size = BytesCompactTensor(*src_data);
    int64_t size_to_cpy = args->size;
    int64_t destination_offset = args->dst_offset;
    int64_t source_offset = args->src_offset;
    if(size_to_cpy == -1) {
      // default, copy entire tensor
      size_to_cpy = src_size;
    }
    CHECK(size_to_cpy >= 0 && destination_offset >= 0 && source_offset >= 0)
      << "Size and offset must be greater or equal to zero.";
    CHECK_LE(source_offset, src_size) << "Src offset must be smaller or equal to src tensor size.";
    CHECK_LE(destination_offset, dst_size) << "Dst offset must be smaller or equal to dst tensor size.";
    CHECK_LE(size_to_cpy, dst_size - destination_offset) << "Copy will write past the end of dst tensor.";
    CHECK_LE(size_to_cpy, src_size - source_offset) << "Copy will read past the end of src tensor.";
    size = size_to_cpy;
    dst_offset = destination_offset;
    src_offset = source_offset;
  }

 public:
  ~CudaCopyInplace() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.copy_inplace"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<CopyInplaceArgs>();
    Execute({args->dst_data, args->src_data}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());
    DLTensor* dst_tensor = inputs[0];
    DLTensor* src_tensor = inputs[1];
    DLTensor* out = output;
    CHECK_EQ(out->data, dst_tensor->data) << "Dst and output should share the same storage.";
    void* dst_data_at_offset = reinterpret_cast<uint8_t*>(dst_tensor->data) + dst_offset;
    void* src_data_at_offset = reinterpret_cast<uint8_t*>(src_tensor->data) + src_offset;
    CUDA_CALL(cudaMemcpyAsync(dst_data_at_offset, src_data_at_offset, size,
                              cudaMemcpyDeviceToDevice, (cudaStream_t)cuda_device_api->GetStream()));
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaCopyInplace(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cuda, copy_inplace, 10);
RAF_OP_ENV_MAKER("raf.op.cuda.copy_inplace", CudaCopyInplace::make);

class CudaDefuseTensor : public raf::op::OpEnv {
  void* stream;
  std::vector<int64_t> tuple_sizes;

  explicit CudaDefuseTensor(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.defuse_tensor");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("data")};
    RequestStream(&stream, cv->device, StreamTagEnum::MemCudaToCuda2());

    tuple_sizes = cv->args.as<DefuseTensorArgs>()->sizes;
    DLTensor* x = cv->args.as<DefuseTensorArgs>()->data;
    int64_t nbytes = (x->dtype.bits + 7) / 8;
    for (auto& size : tuple_sizes) {
      size *= nbytes;
    }
  }

 public:
  ~CudaDefuseTensor() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.defuse_tensor"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<DefuseTensorArgs>();
    Execute({args->data}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    // Defuse Tensor
    DLTensor* in = inputs[0];
    int64_t nbytes = (in->dtype.bits + 7) / 8;
    auto& of = Downcast<value::TupleValue>(output)->fields;
    size_t offset = 0;
    for (int i = 0; i < tuple_sizes.size(); ++i) {
      DLTensor* x = of[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(in->data) + offset;
      CUDA_CALL(cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i],
                                cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
      offset += tuple_sizes[i];
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaDefuseTensor(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cuda, defuse_tensor, 10);
RAF_OP_ENV_MAKER("raf.op.cuda.defuse_tensor", CudaDefuseTensor::make);

class CudaZeros : public raf::op::OpEnv {
  size_t size_in_bytes;
  // Device device_;

  explicit CudaZeros(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op.zeros");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    auto args = cv->args.as<InitOpArgs>();
    auto shape = GetShapeVecFromValue(args->shape);
    auto dtype = ir::String2DLDataType(args->dtype);
    static const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
    // device_ = Device(static_cast<tvm::Device>((*str2dev)(args->device)));
    // CHECK(device_ == Device::Current(false)) << "Trying to create a tensor on a different device. Current: " << Device::Current(false) << ", requested: " << device_;
    size_in_bytes = 1;
    for (auto dim : shape) {
      size_in_bytes *= dim;
    }
    size_in_bytes *= (dtype.bits + 7) / 8;
  }

 public:
  ~CudaZeros() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.zeros"));
  }

  void Execute(const CallValues& cv) override {
    Execute({}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    static auto cuda_device_api = DeviceAPI::Get(DevType::kCUDA());
    // CHECK(cuda_device_api->GetDevice() == device_.device_id()) << "Trying to create a tensor on a different device. Current: " << cuda_device_api->GetDevice() << ", requested: " << device_->device_id;
    DLTensor* out = output;
    CUDA_CALL(cudaMemsetAsync(out->data, 0, size_in_bytes,
                             (cudaStream_t)cuda_device_api->GetStream()));
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaZeros(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cuda, zeros, 12);
RAF_OP_ENV_MAKER("raf.op.cuda.zeros", CudaZeros::make);

}  // namespace cuda
}  // namespace op
}  // namespace raf
