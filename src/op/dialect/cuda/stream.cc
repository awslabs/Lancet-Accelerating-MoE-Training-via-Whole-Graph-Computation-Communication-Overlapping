/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/stream.cc
 * \brief Communication operators for cuda stream controlling.
 */
#include <cuda_runtime.h>
#include <vector>
#include "raf/op_utils.h"
#include "raf/tensor.h"
#include "../../schema/stream.h"

namespace raf {
namespace op {
namespace communication {
namespace nccl {

class CudaStreamSync : public raf::op::OpEnv {
  void* stream;
  explicit CudaStreamSync(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::StreamArgs>();
    auto& stream_tag_id = args->stream_tag;
    auto op = ir::Op::Get("raf.op.stream_sync");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, stream_tag_id);
  }

 public:
  ~CudaStreamSync() {
    // Nothing
  }
  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cuda.stream_sync"));
  }

  void Execute(const CallValues& cv) override {
    cudaStreamSynchronize((cudaStream_t)stream);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    auto input_x = tvm::runtime::Downcast<value::TupleValue>(inputs[0]);
    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      tensor::Tensor& output_tensor = output;
      output_tensor = input_x->fields[0];
    } else {
      auto out = tvm::runtime::Downcast<value::TupleValue>(output);
      for (size_t i = 0; i < input_x->fields.size(); i++) {
        out->fields[i].operator tensor::Tensor &() = input_x->fields[i];
      }
    }
    cudaStreamSynchronize((cudaStream_t)stream);
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaStreamSync(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cuda, stream_sync, 10);
RAF_OP_ENV_MAKER("raf.op.cuda.stream_sync", CudaStreamSync::make);

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace raf
