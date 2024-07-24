/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/moe.cc
 * \brief Typing of MoE operators
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "../schema/moe.h"
#include "./utils.h"

namespace raf {
namespace op {
namespace type {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
using tvm::tir::as_const_int;

inline std::vector<int64_t> GetShapeFromTensorType(TensorType type) {
  std::vector<int64_t> result = {};
  for (auto axis : Downcast<TensorType>(type)->shape) {
    result.push_back(*as_const_int(axis));
  }
  return result;
}

Type MoeEncodeInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType gate = Downcast<TensorType>(GetType(args->gate));
  CHECK_EQ(data->shape.size(), 2) << "ValueError: Input data should be 2-D";
  CHECK_EQ(gate->shape.size(), 2) << "ValueError: Input gate should be 2-D";

  float capacity_factor = args->capacity_factor;

  std::vector<int64_t> data_shape = GetShapeFromTensorType(data);
  std::vector<int64_t> gate_shape = GetShapeFromTensorType(gate);

  CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0.";
  int dim_S = data_shape[0];
  int dim_E = gate_shape[1];
  int dim_M = data_shape[1];

  int capacity = static_cast<int>(capacity_factor * ((dim_S + dim_E - 1) / dim_E));

  CHECK_EQ(data->dtype.code(), kDLFloat) << "Input data should be float32 or float16.";
  int data_bits = data->dtype.bits();

  Array<Type> ret;
  Array<PrimExpr> odata_shape = {Integer(dim_E), Integer(capacity), Integer(dim_M)};
  Array<PrimExpr> ogate_shape = {Integer(dim_S)};
  Array<PrimExpr> omask_shape = {Integer(2), Integer(dim_S)};
  Array<PrimExpr> ocapacity_shape = {Integer(dim_E)};
  ret.push_back(TensorType(ogate_shape, DataType::Float(data_bits))); // gates_s
  ret.push_back(TensorType(omask_shape, DataType::Int(32))); // indices_locations
  ret.push_back(TensorType(ocapacity_shape, DataType::Int(32))); // used_capacity
  ret.push_back(TensorType(ocapacity_shape, DataType::UInt(64))); // n_elems_per_expert
  ret.push_back(TensorType(odata_shape, data->dtype)); // dispatched_input
  return TupleType(ret);
};

Type MoeEncodeBatchPrioritizedInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeBatchPrioritizedArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType gate = Downcast<TensorType>(GetType(args->gate));
  CHECK_EQ(data->shape.size(), 2) << "ValueError: Input data should be 2-D";
  CHECK_EQ(gate->shape.size(), 2) << "ValueError: Input gate should be 2-D";

  float capacity_factor = args->capacity_factor;

  int n_partitions = args->n_partitions;
  int partition_id = args->partition_id;
  CHECK_GT(n_partitions, 0) << "n_partitions should be greater than 0.";
  CHECK_LT(partition_id, n_partitions) << "partition_id should be less than n_partitions.";

  std::vector<int64_t> data_shape = GetShapeFromTensorType(data);
  std::vector<int64_t> gate_shape = GetShapeFromTensorType(gate);

  CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0.";
  int orig_dim_S = data_shape[0];
  int dim_S = orig_dim_S / n_partitions;
  int dim_E = gate_shape[1];
  int dim_M = data_shape[1];

  int capacity = static_cast<int>(capacity_factor * ((dim_S + dim_E - 1) / dim_E));

  CHECK_EQ(data->dtype.code(), kDLFloat) << "Input data should be float32 or float16.";
  int data_bits = data->dtype.bits();

  Array<Type> ret;
  Array<PrimExpr> odata_shape = {Integer(dim_E), Integer(capacity), Integer(dim_M)};
  Array<PrimExpr> ogate_shape = {Integer(dim_S)};
  Array<PrimExpr> omask_shape = {Integer(2), Integer(dim_S)};
  Array<PrimExpr> ocapacity_shape = {Integer(dim_E)};
  ret.push_back(TensorType(ogate_shape, DataType::Float(data_bits))); // gates_s
  ret.push_back(TensorType(omask_shape, DataType::Int(32))); // indices_locations
  ret.push_back(TensorType(ocapacity_shape, DataType::Int(32))); // used_capacity
  ret.push_back(TensorType(ocapacity_shape, DataType::UInt(64))); // n_elems_per_expert
  ret.push_back(TensorType(odata_shape, data->dtype)); // dispatched_input
  return TupleType(ret);
};

Type MoeMergeMasksInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeMergeMasksArgs>();
  CHECK(args != nullptr);
  TensorType in_indices_locations = Downcast<TensorType>(GetType(args->indices_locations[0]));
  CHECK_EQ(in_indices_locations->shape.size(), 2) << "Input locations should be 2-D."; // shape (2, S)

  std::vector<int64_t> indices_locations_shape = GetShapeFromTensorType(in_indices_locations);
  int dim_S = indices_locations_shape[1];
  int n_partitions = args->indices_locations.size();

  Array<PrimExpr> omask_shape = {Integer(2), Integer(dim_S * n_partitions)};

  return TensorType(omask_shape, DataType::Int(32));
};

Type MoeRedispatchInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeRedispatchArgs>();
  CHECK(args != nullptr);
  TensorType in_data = Downcast<TensorType>(GetType(args->data[0]));
  TensorType in_indices_locations = Downcast<TensorType>(GetType(args->indices_locations[0]));
  std::vector<int64_t> data_shape = GetShapeFromTensorType(in_data);
  Array<PrimExpr> odata_shape;
  if(data_shape.size() == 3) {
    // shape (E, C, M)
    odata_shape = {Integer(data_shape[0]), Integer(data_shape[1]), Integer(data_shape[2])};
  } else {
    CHECK_EQ(data_shape.size(), 4) << "Input data should be 3 or 4-D.";
    // shape (G, LE, C, M)
    odata_shape = {Integer(data_shape[0]), Integer(data_shape[1]), Integer(data_shape[2]), Integer(data_shape[3])};
  }
  CHECK_EQ(in_indices_locations->shape.size(), 2) << "Input locations should be 2-D."; // shape (2, S)
  return TensorType(odata_shape, DataType(in_data->dtype));
}

Type MoeRedispatchExpertInputInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeRedispatchExpertInputArgs>();
  CHECK(args != nullptr);
  TensorType in_data = Downcast<TensorType>(GetType(args->data[0]));
  TensorType in_recv_cnts = Downcast<TensorType>(GetType(args->recv_cnts[0]));

  std::vector<int64_t> data_shape = GetShapeFromTensorType(in_data);
  Array<PrimExpr> odata_shape;
  if (data_shape.size() == 4) {
    // shape (C, 1, G, M)
    CHECK_EQ(data_shape[1], 1) << "4D Input data should have dim(1) = 1.";
    odata_shape = {Integer(data_shape[0]), Integer(data_shape[1]), Integer(data_shape[2]), Integer(data_shape[3])};
  } else if (data_shape.size() == 3) {
    // shape (C, G, M)
    odata_shape = {Integer(data_shape[0]), Integer(data_shape[1]), Integer(data_shape[2])};
  } else {
    CHECK_EQ(data_shape.size(), 2) << "Input data should be 2,3 or 4-D.";
    // shape (C x G, M)
    odata_shape = {Integer(data_shape[0]), Integer(data_shape[1])};
  }
  CHECK_EQ(in_recv_cnts->shape.size(), 1) << "Input recv_cnts should be 1-D."; // shape (G x LE)

  return TensorType(odata_shape, DataType(in_data->dtype));
}

Type MoeEncodeDxInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));                                 // [E, C, M]
  TensorType indices_locations = Downcast<TensorType>(GetType(args->indices_locations));   // [2, S]

  CHECK_EQ(dy->shape.size(), 3) << "Input dy should be 3-D.";

  std::vector<int64_t> dy_shape = GetShapeFromTensorType(dy);
  std::vector<int64_t> mask_shape = GetShapeFromTensorType(indices_locations);

  int dim_S = mask_shape[1];
  int dim_M = dy_shape[2];

  Array<PrimExpr> out_shape = {Integer(dim_S), Integer(dim_M)};
  return TensorType(out_shape, DataType(dy->dtype));
};

Type MoeEncodeDgInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeEncodeDgArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));      // [S]
  TensorType indices_locations = Downcast<TensorType>(GetType(args->indices_locations));   // [2, S]
  CHECK_EQ(dy->shape.size(), 1) << "Input dy should be 1-D.";
  CHECK_EQ(indices_locations->shape.size(), 2) << "Input indices_locations should be 2-D.";

  std::vector<int64_t> dy_shape = GetShapeFromTensorType(dy);

  Array<PrimExpr> out_shape = {Integer(dy_shape[0]), Integer(args->n_experts)}; // [S, E]
  return TensorType(out_shape, DataType(dy->dtype));
};

Type MoeDecodeInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeDecodeArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));   // [E, C, M]
  TensorType gate = Downcast<TensorType>(GetType(args->gate));   // [S]

  CHECK_EQ(data->shape.size(), 3) << "Input data should be 3-D.";
  CHECK_EQ(gate->shape.size(), 1) << "Input gate should be 1-D.";

  std::vector<int64_t> data_shape = GetShapeFromTensorType(data);
  std::vector<int64_t> gate_shape = GetShapeFromTensorType(gate);

  int dim_S = gate_shape[0];
  int dim_M = data_shape[2];

  Array<PrimExpr> out_shape = {Integer(dim_S), Integer(dim_M)};
  return TensorType(out_shape, DataType(data->dtype));
};

Type MoeDecodeDxInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeDecodeDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));   // [S, M]
  CHECK_EQ(dy->shape.size(), 2) << "Input dy should be 2-D.";

  std::vector<int64_t> dy_shape = GetShapeFromTensorType(dy);
  Array<PrimExpr> out_shape = {Integer(args->n_experts), Integer(args->capacity), Integer(dy_shape[1])};
  return TensorType(out_shape, DataType(dy->dtype));
};

Type MoeDecodeDgInfer(const CallValues& call) {
  const auto* args = call->args.as<MoeDecodeDgArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));   // [S, M]
  CHECK_EQ(dy->shape.size(), 2) << "Input dy should be 2-D.";

  std::vector<int64_t> dy_shape = GetShapeFromTensorType(dy);

  Array<PrimExpr> gate_shape = {Integer(dy_shape[0])}; // [S]
  return TensorType(gate_shape, DataType(dy->dtype));
};

Type SparseExpertMatmulNtInfer(const CallValues& call) {
  const auto* args = call->args.as<SparseExpertMatmulNtArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));                    // [C, G, M]
  TensorType weight = Downcast<TensorType>(GetType(args->weight));                // [N, M]
  TensorType nelements = Downcast<TensorType>(GetType(args->nelements_per_gpu));  // [G]
  int nelements_stride = args->nelements_stride;

  CHECK_EQ(data->shape.size(), 3) << "Input data should be 3-D.";
  CHECK_EQ(weight->shape.size(), 2) << "Input weight should be 2-D.";
  CHECK_EQ(nelements->shape.size(), 1) << "Input nelements should be 1-D.";
  std::vector<int64_t> data_shape = GetShapeFromTensorType(data);
  std::vector<int64_t> weight_shape = GetShapeFromTensorType(weight);
  std::vector<int64_t> nelements_shape = GetShapeFromTensorType(nelements);
  CHECK_EQ(data_shape[1], nelements_shape[0] / nelements_stride) << "Input data and nelements should have matching shape in dim 1 and 0 after stride.";
  CHECK_EQ(weight_shape[1], data_shape[2]) << "Input weight and nelements should have matching shapes in dim 1 and 2.";

  Array<PrimExpr> out_shape = {Integer(data_shape[0]), Integer(data_shape[1]), Integer(weight_shape[0])}; // [C, G, N]
  return TensorType(out_shape, DataType(data->dtype));
}

// Also register relay type for MoeEncode
template<int NumInputs>
bool MoeEncodeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // [data, gate, used_capacity, out] for MoeEncode
  // [data, gate, out] for MoeEncodeBatchPrioritized
  CHECK_EQ(types.size(), NumInputs);
  if(types[0].as<IncompleteTypeNode>()) {
    return false;
  }
  TensorType data_type = Downcast<TensorType>(types[0]);
  TensorType gate_type = Downcast<TensorType>(types[1]);
  CHECK_EQ(data_type->shape.size(), 2) << "ValueError: Input data should be 2-D";
  CHECK_EQ(gate_type->shape.size(), 2) << "ValueError: Input gate should be 2-D";

  float capacity_factor = 1.0; // TODO: remove hardcode by adding an attrs

  std::vector<int64_t> data_shape = GetShapeFromTensorType(data_type);
  std::vector<int64_t> gate_shape = GetShapeFromTensorType(gate_type);

  CHECK_EQ(data_shape[0], gate_shape[0]) << "Input data and gate should have matching shape in dim 0.";
  int dim_S = data_shape[0];
  int dim_E = gate_shape[1];
  int dim_M = data_shape[1];

  int capacity = static_cast<int>(capacity_factor * ((dim_S + dim_E - 1) / dim_E));
  int input_bits = data_type->dtype.bits();

  Array<Type> ret;
  Array<PrimExpr> odata_shape = {Integer(dim_E), Integer(capacity), Integer(dim_M)};
  Array<PrimExpr> ogate_shape = {Integer(dim_S)};
  Array<PrimExpr> omask_shape = {Integer(2), Integer(dim_S)};
  Array<PrimExpr> ocapacity_shape = {Integer(dim_E)};
  ret.push_back(TensorType(ogate_shape, DataType::Float(input_bits))); // gates_s
  ret.push_back(TensorType(omask_shape, DataType::Int(32))); // indices_locations
  ret.push_back(TensorType(ocapacity_shape, DataType::Int(32))); // used_capacity
  ret.push_back(TensorType(ocapacity_shape, DataType::UInt(64))); // elements_per_expert
  ret.push_back(TensorType(odata_shape, data_type->dtype)); // dispatched_input

  auto tt = TupleType(ret);
  reporter->Assign(types[NumInputs-1], tt);
  return true;
}



// Also register relay type for MoeDecode
bool MoeDecodeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4); // [data, gate, indices_locations, out]
  if(types[0].as<IncompleteTypeNode>()) {
    return false;
  }
  TensorType data = Downcast<TensorType>(types[0]);   // [E, C, M]
  TensorType gate = Downcast<TensorType>(types[1]);   // [S]

  CHECK_EQ(data->shape.size(), 3) << "Input data should be 3-D.";
  CHECK_EQ(gate->shape.size(), 1) << "Input gate should be 1-D.";

  std::vector<int64_t> data_shape = GetShapeFromTensorType(data);
  std::vector<int64_t> gate_shape = GetShapeFromTensorType(gate);

  int dim_S = gate_shape[0];
  int dim_M = data_shape[2];

  Array<PrimExpr> out_shape = {Integer(dim_S), Integer(dim_M)};

  auto tt = TensorType(out_shape, DataType(data->dtype));
  reporter->Assign(types[3], tt);
  return true;
}

RAF_OP_TYPE("raf.op.moe_encode", "MoeEncodeInfer", MoeEncodeInfer);
RAF_OP_TYPE("raf.op.moe_encode_batch_prioritized", "MoeEncodeBatchPrioritizedInfer", MoeEncodeBatchPrioritizedInfer);
RAF_OP_TYPE("raf.op.moe_merge_masks", "MoeMergeMasksInfer", MoeMergeMasksInfer);
RAF_OP_TYPE("raf.op.moe_redispatch", "MoeRedispatchInfer", MoeRedispatchInfer);
RAF_OP_TYPE("raf.op.moe_redispatch_expert_input", "MoeRedispatchExpertInputInfer", MoeRedispatchExpertInputInfer);
RAF_OP_TYPE("raf.op.moe_encode_dx", "MoeEncodeDxInfer", MoeEncodeDxInfer);
RAF_OP_TYPE("raf.op.moe_encode_dg", "MoeEncodeDgInfer", MoeEncodeDgInfer);
RAF_OP_TYPE("raf.op.moe_decode", "MoeDecodeInfer", MoeDecodeInfer);
RAF_OP_TYPE("raf.op.moe_decode_dx", "MoeDecodeDxInfer", MoeDecodeDxInfer);
RAF_OP_TYPE("raf.op.moe_decode_dg", "MoeDecodeDgInfer", MoeDecodeDgInfer);
RAF_OP_TYPE("raf.op.sparse_expert_matmul_nt", "SparseExpertMatmulNtInfer", SparseExpertMatmulNtInfer);

// TODO: RAF's type infer function will overwrite the op type of the op. Add type rel after calling RAF_OP_TYPE
// solves the Relay type infer problem during tracing (why this will not cause a problem in RAF?)
RELAY_REGISTER_OP("raf.op.moe_encode").add_type_rel("MoeEncodeRel", MoeEncodeRel<4>);
RELAY_REGISTER_OP("raf.op.moe_encode_batch_prioritized").add_type_rel("MoeEncodeBatchPrioritizedRel", MoeEncodeRel<3>);
RELAY_REGISTER_OP("raf.op.moe_decode").add_type_rel("MoeDecodeRel", MoeDecodeRel);

}  // namespace type
}  // namespace op
}  // namespace raf
