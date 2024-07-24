/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/moe.cc
 * \brief Declaration of MoE-specific operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/moe.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

bool CheckDTypeFloat32Or16(DType dtype) {
  return dtype.code == DTypeCode::kFloat() && (dtype.bits == 32 || dtype.bits == 16);
}

void MoeEncodeDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  DLTensor* gate = args->gate;
  DLTensor* used_capacity = args->used_capacity;
  CHECK_EQ(data->ndim, 2) << "Input data should be 2-D.";
  CHECK_EQ(gate->ndim, 2) << "Input gate should be 2-D.";
  CHECK_EQ(used_capacity->ndim, 1) << "Input used_capacity should be 1-D.";
  CHECK(CheckDTypeFloat32Or16(DType(data->dtype))) << "moe_encode only implemented for float32 or float16.";
  float capacity_factor = args->capacity_factor;

  std::vector<int64_t> data_shape(data->shape, data->shape + data->ndim);
  std::vector<int64_t> gate_shape(gate->shape, gate->shape + gate->ndim);

  CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0.";
  int dim_S = data_shape[0];
  int dim_E = gate_shape[1];
  int dim_M = data_shape[1];

  int capacity = static_cast<int>(capacity_factor * ((dim_S + dim_E - 1) / dim_E));

  std::vector<TensorValue> ret;
  std::vector<int64_t> odata_shape = {dim_E, capacity, dim_M};
  std::vector<int64_t> ogate_shape = {dim_S};
  std::vector<int64_t> omask_shape = {2, dim_S};
  std::vector<int64_t> ocapacity_shape = {dim_E};
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kFloat(), 32), ogate_shape)); // gates_s
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), omask_shape)); // indices_locations
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), ocapacity_shape)); // accum_used_capacity
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kUInt(), 64), ocapacity_shape)); // elements_per_expert (used in a2av)
  ret.push_back(TensorValue::Assemble(data->device, data->dtype, odata_shape)); // dispatched_input

  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  call->device = data->device;
}


void MoeEncodeBatchPrioritizedDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeBatchPrioritizedArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  DLTensor* gate = args->gate;
  CHECK_EQ(data->ndim, 2) << "Input data should be 2-D.";
  CHECK_EQ(gate->ndim, 2) << "Input gate should be 2-D.";
  CHECK(CheckDTypeFloat32Or16(DType(data->dtype))) << "moe_encode only implemented for float32 or float16.";
  float capacity_factor = args->capacity_factor;
  int n_partitions = args->n_partitions;
  int partition_id = args->partition_id;
  CHECK_GT(n_partitions, 0) << "n_partitions should be greater than 0.";
  CHECK_LT(partition_id, n_partitions) << "partition_id should be less than n_partitions.";

  std::vector<int64_t> data_shape(data->shape, data->shape + data->ndim);
  std::vector<int64_t> gate_shape(gate->shape, gate->shape + gate->ndim);

  CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0.";
  int orig_dim_S = data_shape[0];
  int dim_S = orig_dim_S / n_partitions;
  int dim_E = gate_shape[1];
  int dim_M = data_shape[1];

  int capacity = static_cast<int>(capacity_factor * ((dim_S + dim_E - 1) / dim_E));

  std::vector<TensorValue> ret;
  std::vector<int64_t> odata_shape = {dim_E, capacity, dim_M};
  std::vector<int64_t> ogate_shape = {dim_S};
  std::vector<int64_t> omask_shape = {2, dim_S};
  std::vector<int64_t> ocapacity_shape = {dim_E};
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kFloat(), 32), ogate_shape)); // gates_s
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), omask_shape)); // indices_locations
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), ocapacity_shape)); // accum_used_capacity
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kUInt(), 64), ocapacity_shape)); // elements_per_expert (used in a2av)
  ret.push_back(TensorValue::Assemble(data->device, data->dtype, odata_shape)); // dispatched_input

  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  call->device = data->device;
}

void MoeMergeMasksDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeMergeMasksArgs>();
  CHECK(args != nullptr);
  auto& indices_locations = args->indices_locations;
  DLTensor* mask0 = indices_locations[0];
  CHECK_EQ(mask0->ndim, 2) << "Input indices_locations should be 2-D."; // shape (2, S)

  std::vector<int64_t> mask_shape(mask0->shape, mask0->shape + mask0->ndim);

  int dim_S = mask_shape[1];
  int n_partitions = indices_locations.size();

  std::vector<int64_t> omask_shape = {2, dim_S * n_partitions};

  call->out = TensorValue::Assemble(mask0->device, DType(DTypeCode::kInt(), 32), omask_shape);
  call->device = mask0->device;
}

void MoeRedispatchDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeRedispatchArgs>();
  CHECK(args != nullptr);
  auto& partitioned_input = args->data;
  auto& indices_locations = args->indices_locations;
  DLTensor* mask0 = indices_locations[0];
  DLTensor* data0 = partitioned_input[0];
  if (data0->ndim != 3 && data0->ndim != 4) {
    LOG(FATAL) << "Input data should be 3 or 4-D.";
  }
  CHECK_EQ(mask0->ndim, 2) << "Input indices_locations should be 2-D."; // shape (2, S)

  std::vector<int64_t> data_shape(data0->shape, data0->shape + data0->ndim);

  call->out = TensorValue::Assemble(data0->device, DType(DTypeCode::kFloat(), 32), data_shape);
  call->device = data0->device;
}

void MoeRedispatchExpertInputDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeRedispatchExpertInputArgs>();
  CHECK(args != nullptr);
  auto& partitioned_input = args->data;
  auto& recv_cnts = args->recv_cnts;
  DLTensor* mask0 = recv_cnts[0];
  DLTensor* data0 = partitioned_input[0];

  std::vector<int64_t> out_data_shape;
  // data0 can be either dim 3 or dim 4 depending on whether it is squeezed
  if (data0->ndim == 4) {
    CHECK_EQ(data0->shape[1], 1) << "4D Input data should have dim 1 at axis 1.";
    out_data_shape = {data0->shape[0], data0->shape[1], data0->shape[2], data0->shape[3]}; // shape (C, 1, G, M)
  } else if (data0->ndim == 3) {
    out_data_shape = {data0->shape[0], data0->shape[1], data0->shape[2]}; // shape (C, G, M)
  } else {
    CHECK_EQ(data0->ndim, 2) << "Input data should be 2,3 or 4-D."; // shape (C x G, M)
    out_data_shape = {data0->shape[0], data0->shape[1]}; // shape (C x G, M)
  }
  CHECK_EQ(mask0->ndim, 1) << "Input indices_locations should be 2-D."; // shape (G x LE)

  call->out = TensorValue::Assemble(data0->device, DType(DTypeCode::kFloat(), 32), out_data_shape);
  call->device = data0->device;
}

void MoeEncodeDxDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeDxArgs>();
  CHECK(args != nullptr);
  DLTensor* dy = args->dy;                                // [E, C, M]
  DLTensor* indices_locations = args->indices_locations;  // [2, S]
  CHECK_EQ(dy->ndim, 3) << "Input dy should be 3-D.";
  CHECK(CheckDTypeFloat32Or16(DType(dy->dtype))) << "moe_encode_dx only implemented for float32 or float16.";

  std::vector<int64_t> dy_shape(dy->shape, dy->shape + dy->ndim);
  std::vector<int64_t> indices_shape(indices_locations->shape, indices_locations->shape + indices_locations->ndim);

  std::vector<int64_t> data_shape = {indices_shape[1], dy_shape[2]}; // [S, M]
  call->out = TensorValue::Assemble(dy->device, dy->dtype, data_shape);
  call->device = dy->device;
}

void MoeEncodeDgDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeDgArgs>();
  CHECK(args != nullptr);
  DLTensor* dy = args->dy;                                // [S]
  DLTensor* indices_locations = args->indices_locations;  // [2, S]
  CHECK_EQ(dy->ndim, 1) << "Input dy should be 1-D.";
  CHECK_EQ(indices_locations->ndim, 2) << "Input indices_locations should be 2-D.";
  CHECK(CheckDTypeFloat32Or16(DType(dy->dtype))) << "moe_encode_dg only implemented for float32 or float16.";

  std::vector<int64_t> dy_shape(dy->shape, dy->shape + dy->ndim);

  std::vector<int64_t> gate_shape = {dy_shape[0], args->n_experts}; // [S, E]

  call->out = TensorValue::Assemble(dy->device, dy->dtype, gate_shape);
  call->device = dy->device;
}

void MoeDecodeDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeDecodeArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;              // [E, C, M]
  DLTensor* gate = args->gate;              // [S]
  CHECK_EQ(data->ndim, 3) << "Input data should be 3-D.";
  CHECK_EQ(gate->ndim, 1) << "Input gate should be 1-D.";
  CHECK(CheckDTypeFloat32Or16(DType(data->dtype))) << "moe_encode_dx only implemented for float32 or float16.";

  std::vector<int64_t> data_shape(data->shape, data->shape + data->ndim);
  std::vector<int64_t> gate_shape(gate->shape, gate->shape + gate->ndim);

  std::vector<int64_t> out_shape = {gate_shape[0], data_shape[2]}; // [S, M]
  call->out = TensorValue::Assemble(data->device, data->dtype, out_shape);
  call->device = data->device;
}

void MoeDecodeDxDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeDecodeDxArgs>();
  CHECK(args != nullptr);
  DLTensor* dy = args->dy;  // [S, M]
  CHECK_EQ(dy->ndim, 2) << "Input dy should be 2-D.";
  CHECK(CheckDTypeFloat32Or16(DType(dy->dtype))) << "moe_decode_dx only implemented for float32 or float16.";

  std::vector<int64_t> dy_shape(dy->shape, dy->shape + dy->ndim);
  std::vector<int64_t> data_shape = {args->n_experts, args->capacity, dy_shape[1]}; // [E, C, M]

  call->out = TensorValue::Assemble(dy->device, dy->dtype, data_shape);
  call->device = dy->device;
}

void MoeDecodeDgDeclare(const CallValues& call) {
  const auto* args = call->args.as<MoeDecodeDgArgs>();
  CHECK(args != nullptr);
  DLTensor* dy = args->dy;  // [S, M]
  CHECK_EQ(dy->ndim, 2) << "Input dy should be 2-D.";
  CHECK(CheckDTypeFloat32Or16(DType(dy->dtype))) << "moe_decode_dx only implemented for float32 or float16.";

  std::vector<int64_t> dy_shape(dy->shape, dy->shape + dy->ndim);

  std::vector<int64_t> gate_shape = {dy_shape[0]}; // [S]
  call->out = TensorValue::Assemble(dy->device, dy->dtype, gate_shape);
  call->device = dy->device;
}

void SparseExpertMatmulNtDeclare(const CallValues& call) {
  const auto* args = call->args.as<SparseExpertMatmulNtArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  DLTensor* weight = args->weight;
  DLTensor* nelements = args->nelements_per_gpu;
  int nelements_stride = args->nelements_stride;
  CHECK_EQ(data->ndim, 3) << "Input data should be 3-D.";
  CHECK_EQ(weight->ndim, 2) << "Input weight should be 2-D.";
  CHECK_EQ(nelements->ndim, 1) << "Input nelements should be 1-D.";
  std::vector<int64_t> data_shape(data->shape, data->shape + data->ndim); // [C, G, M]
  std::vector<int64_t> weight_shape(weight->shape, weight->shape + weight->ndim); // [N, M]
  std::vector<int64_t> nelements_shape(nelements->shape, nelements->shape + nelements->ndim); // [G*nelements_stride]
  CHECK_EQ(data_shape[1], nelements_shape[0] / nelements_stride) << "Input data and nelements should have matching shape in dim 1 and 0 after stride.";
  CHECK_EQ(weight_shape[1], data_shape[2]) << "Input weight and nelements should have matching shapes in dim 1 and 2.";

  std::vector<int64_t> out_shape = {data_shape[0], data_shape[1], weight_shape[0]};
  call->out = TensorValue::Assemble(data->device, data->dtype, out_shape);
}

RAF_OP_DECLARE("raf.op.moe_encode_batch_prioritized", MoeEncodeBatchPrioritizedDeclare)
  .set_attr<TOpPattern>("TOpPattern", kOpaque)
  .set_num_inputs(2)
  .add_argument("data", "Tensor", "The reshaped input tensor.")
  .add_argument("gate", "Tensor", "The gate tensor.");
RAF_OP_DECLARE("raf.op.moe_encode", MoeEncodeDeclare)
  .set_attr<TOpPattern>("TOpPattern", kOpaque)
  .set_num_inputs(3)
  .add_argument("data", "Tensor", "The reshaped input tensor.")
  .add_argument("gate", "Tensor", "The gate tensor.")
  .add_argument("used_capacity", "Tensor", "Tensor containing used capacity for each expert.");
RAF_OP_DECLARE("raf.op.moe_merge_masks", MoeMergeMasksDeclare)
  .set_attr<TOpPattern>("TOpPattern", kOpaque)
  .set_num_inputs(1)
  .add_argument("indices_locations", "TensorTuple", "partitioned indices and locations.");
RAF_OP_DECLARE("raf.op.moe_redispatch", MoeRedispatchDeclare).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.moe_redispatch_expert_input", MoeRedispatchExpertInputDeclare).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.moe_encode_dx", MoeEncodeDxDeclare).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.moe_encode_dg", MoeEncodeDgDeclare).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.moe_decode", MoeDecodeDeclare)
  .set_attr<TOpPattern>("TOpPattern", kOpaque)
  .set_num_inputs(3)
  .add_argument("data", "Tensor", "The expert output tensor.")
  .add_argument("gate", "Tensor", "The 1D gate tensor.")
  .add_argument("indices_locations", "Tensor", "The 2D indices and locations tensor.");
RAF_OP_DECLARE("raf.op.moe_decode_dx", MoeDecodeDxDeclare).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.moe_decode_dg", MoeDecodeDgDeclare).set_attr<TOpPattern>("TOpPattern", kOpaque);
RAF_OP_DECLARE("raf.op.sparse_expert_matmul_nt", SparseExpertMatmulNtDeclare)
  .set_attr<TOpPattern>("TOpPattern", kOpaque)
  .set_num_inputs(3)
  .add_argument("data", "Tensor", "The reshaped input tensor.")
  .add_argument("weight", "Tensor", "The weight tensor.")
  .add_argument("nelements_per_row", "Tensor", "Tensor containing number of elements per row.");

}  // namespace declare
}  // namespace op
}  // namespace raf
