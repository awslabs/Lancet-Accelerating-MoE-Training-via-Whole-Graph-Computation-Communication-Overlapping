/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/tvm_fusion.cc
 * \brief Implementation of tvm dispatch for fused functions
 */
#include "./tvm_fusion.h"
namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::value;
using namespace raf::ir;

Expr Cast2TVMDialect::VisitExpr(const Expr& expr) {
  auto ret = ExprMutator::VisitExpr(expr);
  if (expr->checked_type_.defined()) {
    ret->checked_type_ = expr->checked_type_;
  }
  return ret;
}

Expr Cast2TVMDialect::VisitExpr_(const OpNode* node) {
  auto op = GetRef<Op>(node);
  auto base_op = IsDialectOp(op) ? GetBaseOp(op) : op;
  auto tvm_op = OpDialect::Lower(base_op, "tvm");
  if (tvm_op.defined()) {
    return tvm_op;
  }
  // No TVM op registered for this base op, just return the original op
  return op;
}

Meta2TVM::Meta2TVM(const CallValues& call, const DevType& dev_type)
    : func_(Downcast<ClosureValue>(call->callee)->func),
      call_values_getter_(call),
      device_type_(dev_type) {
}

Expr Meta2TVM::operator()() {
  call_values_getter_();
  Expr ret = VisitExpr(func_);
  return ret;
}

Expr Meta2TVM::VisitExpr(const Expr& expr) {
  auto ret = ExprMutator::VisitExpr(expr);
  ret->checked_type_ = expr->checked_type();
  return ret;
}

Expr Meta2TVM::VisitExpr_(const VarNode* node) {
  input_.insert(GetRef<Var>(node));
  return GetRef<Var>(node);
}

Expr Meta2TVM::VisitExpr_(const CallNode* node) {
  CallValues op_call_values = call_values_getter_.call_values.at(GetRef<Call>(node));
  const Op& op = Downcast<Op>(node->op);
  ICHECK_EQ(GetDialect(op), "tvm")
      << "Encountered a non-TVM op in fused TVM closure: " << op->name;
  auto farg_indices = GetOpAttr<FRAFArgIndices>(op, "FRAFArgIndices");
  auto fattr = GetOpAttr<FRAFAttr>(op, "FRAFAttr");
  Attrs op_tvm_attr = fattr(op_call_values);
  Array<IntImm> arg_indices = farg_indices(op_call_values);
  std::vector<Expr> inputs;
  for (const auto& i : arg_indices) {
    Expr arg = VisitExpr(node->args[i->value]);
    inputs.push_back(arg);
    if (const auto* vn = arg.as<VarNode>()) {
      input_.insert(GetRef<Var>(vn));
    }
  }
  return Call(op, inputs, op_tvm_attr);
}

Expr Meta2TVM::VisitExpr_(const FunctionNode* node) {
  Expr new_body = VisitExpr(node->body);
  std::vector<Var> new_params;
  size_t num = node->params.size();
  for (size_t i = 0; i < num; ++i) {
    const Var& param = node->params[i];
    if (input_.find(param) != input_.end()) {
      // param is a tensor input
      new_params.push_back(param);
      arg_indices.push_back(i);
    }
  }
  func_name = call_values_getter_.readable_name_stream.str();
  return Function(Array<Var>(new_params), new_body, node->ret_type, {});
}

PackedFunc CompileFunc(const op::CallValues& call) {
  tvm::relay::tec::TECompiler te_compiler;
  Device dev = call->device;
  tvm::Target target = dev.tvm_target();
  CHECK(dev.device_type() == DevType::kCPU() || dev.device_type() == DevType::kCUDA())
      << "NotImplementedError: target is not supported " << dev.device_type().c_str();
  Meta2TVM meta_to_tvm(call, dev.device_type());
  Function func = Downcast<Function>(meta_to_tvm());
  // TODO(@hzfan): add cache for raf
  te_compiler->Clear();
  try {
    return te_compiler->JIT(tvm::relay::tec::CCacheKey(func, target));
  } catch (const dmlc::Error& e) {
    if (!AllowJitFailure()) {
      LOG(FATAL) << "Failed to build a fused op " <<
        TruncateName(GetUniqueName(meta_to_tvm.func_name)) << ": " << e.what();
    }
  }
}

OpEnv* FusedFuncBuild(const op::CallValues& call) {
  tvm::relay::tec::TECompiler te_compiler;
  auto env = std::make_unique<TVMOpEnv>();
  Device dev = call->device;
  tvm::Target target = dev.tvm_target();
  CHECK(dev.device_type() == DevType::kCPU() || dev.device_type() == DevType::kCUDA())
      << "NotImplementedError: target is not supported " << dev.device_type().c_str();
  Meta2TVM meta_to_tvm(call, dev.device_type());
  Function func = Downcast<Function>(meta_to_tvm());
  // TODO(@hzfan): add cache for raf
  te_compiler->Clear();
  env->env_name = TruncateName(GetUniqueName(meta_to_tvm.func_name));
  try {
    env->f = te_compiler->JIT(tvm::relay::tec::CCacheKey(func, target));
  } catch (const dmlc::Error& e) {
    if (!AllowJitFailure()) {
      LOG(FATAL) << "Failed to build a fused op " << env->env_name << ": " << e.what();
    }
  }
  env->arg_indices = meta_to_tvm.arg_indices;
  Array<Value> args = GetListArgs(call->args);
  for (const int& i : env->arg_indices) {
    GetDLTensor(args[i], &env->inputs);
  }
  GetDLTensor(call->out, &env->outputs);
  return env.release();
}

/*!
 * \brief Calculate the total computation GFLOPS required by a function.
 * \param call The call values, which callee is a ClosureValue that includes the target function.
 * \param param_types The type of function parameters.
 * \param ret_type The function return type.
 * \param device The device.
 * \return The calculated GFLOPS.
 */
float CalcFuncGFLOPS(const op::CallValues& call, const Array<Type>& param_types,
                     const Type& ret_type, const Device& device) {
  tvm::relay::tec::TECompiler compiler;
  // Create a new call value and cast ops in callee to TVM dialect
  auto new_call = op::CallValues::make();
  auto callee = Downcast<ClosureValue>(call->callee)->func;
  callee = Downcast<Function>(Cast2TVMDialect().Mutate(callee));
  new_call->callee = ClosureValue::make({}, callee);
  new_call->args = call->args;
  new_call->out = call->out;
  new_call->device = call->device;

  Meta2TVM meta_to_tvm(new_call, device.device_type());
  Function tvm_func = Downcast<Function>(meta_to_tvm());
  tvm::Target target = device.tvm_target();

  auto cache_key = tvm::relay::tec::CCacheKey(tvm_func, target);
  try {
    auto tensors = compiler->Lower(cache_key, "mod_calc_flops")->outputs;
    auto dag = tvm::auto_scheduler::ComputeDAG(tensors);
    return dag->flop_ct / 1e9;
  } catch (dmlc::Error& e) {
    LOG(WARNING) << "Failed to create ComputeDAG for " << raf::ir::AsText(tvm_func) << "\n"
                 << e.what();
  }
  return -1;
}

RAF_OP_ENV_MAKER("raf.op.tvm._fused_op", FusedFuncBuild);

} // namespace tvm_dialect
} // namespace op
} // namespace raf
