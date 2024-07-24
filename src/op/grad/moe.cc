/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/grad/moe.cc
 * \brief Declaration of MoE gradients
 */
#include "./grad_utils.h"
#include "raf/pass.h"
#include "raf/ir.h"

namespace raf {
namespace op {
namespace grad {

using namespace raf::ir;

inline std::vector<int64_t> GetShapeVecFromTensorType(const TensorType& ttype) {
  std::vector<int64_t> shape;
  for (auto axis : ttype->shape) {
    auto node = axis.as<ir::IntImmNode>();
    CHECK(node != nullptr) << "Axis " << axis << " is not IntImmNode";
    shape.push_back((int64_t)node->value);
  }
  return shape;
}

Array<Expr> MoeEncodeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  using namespace raf::value;
  static auto op_dx = Op::Get("raf.op.moe_encode_dx");
  static auto op_dg = Op::Get("raf.op.moe_encode_dg");
  // MoeEncode args: [reshaped_input, gate]
  // MoeEncode output tuple: [gates_s, indices_locations, accum_used_capacity, elements_per_expert, dispatched_input]
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& gate = call->args[1];
  TensorType gate_type;
  Expr dim_E;
  if (gate->checked_type_.defined()) {
    gate_type = Downcast<TensorType>(gate->checked_type());
    auto gate_shape = GetShapeVecFromTensorType(gate_type);
    dim_E = MakeConstant(ScalarValue::make(gate_shape[1]));
  } else {
    dim_E = TupleGetItem(GetShape(gate), 1);
  }

  const Expr& dy_gate_s = TupleGetItem(dy, 0);
  const Expr& dy_input = TupleGetItem(dy, 4);
  const Expr& indices_locations = TupleGetItem(y, 1);

  return {Call(op_dx, {dy_input, indices_locations}), Call(op_dg, {dy_gate_s, indices_locations, dim_E}), ir::NullValue<ir::Expr>()};
}

RAF_OP_GRAD("raf.op.moe_encode", MoeEncodeGrad);
RAF_OP_GRAD("raf.op.moe_encode_batch_prioritized", MoeEncodeGrad);

Array<Expr> MoeDecodeGrad(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                          const Expr& dy) {
  using namespace raf::value;
  static auto op_dx = Op::Get("raf.op.moe_decode_dx");
  static auto op_dg = Op::Get("raf.op.moe_decode_dg");
  static auto zeros_like = Op::Get("raf.op.zeros_like");
  // MoeDecode args: [expert_outputs, gates_s, indices_locations]
  // MoeDecode output: combined_output
  const CallNode* call = orig_call.as<CallNode>();
  CHECK(call != nullptr);
  const Expr& expert_outputs = call->args[0]; // [E, C, M]
  Expr dim_E;
  Expr dim_C;
  if (expert_outputs->checked_type_.defined()) {
    auto exp_out_t = Downcast<TensorType>(expert_outputs->checked_type());
    auto exp_out_t_vec = GetShapeVecFromTensorType(exp_out_t);
    int E = exp_out_t_vec[0];
    int C = exp_out_t_vec[1];
    dim_E = MakeConstant(ScalarValue::make(E));
    dim_C = MakeConstant(ScalarValue::make(C));
  } else {
    auto exp_out_shape = GetShape(expert_outputs);
    dim_E = TupleGetItem(exp_out_shape, 0);
    dim_C = TupleGetItem(exp_out_shape, 1);
  }
  const Expr& gate_s = call->args[1];
  const Expr& indices_locations = call->args[2];

  return {Call(op_dx, {dy, gate_s, indices_locations, dim_E, dim_C}),
          Call(op_dg, {dy, expert_outputs, indices_locations}),
          ir::NullValue<ir::Expr>()};
}

RAF_OP_GRAD("raf.op.moe_decode", MoeDecodeGrad);

}  // namespace grad
}  // namespace op
}  // namespace raf
