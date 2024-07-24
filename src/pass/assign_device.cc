/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file assign_device.cc
 * \brief Assign the target device to init and constant ops.
 */
#include "raf/op.h"
#include "raf/ir.h"

#include "raf/pass.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/op_attr_types.h"
#include "../op/schema/init.h"
#include "../op/schema/transform.h"
#include "../op/schema/nn.h"

namespace raf {
namespace pass {
namespace assign_device {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::op::schema;
using namespace raf::value;
template <typename T>
using VarMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

raf::Device GetDeviceFromConstExpr(const Expr& expr) {
  static auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  auto device_name_node = expr.as<ir::ConstantNode>();
  CHECK(device_name_node) << "Device name node " << expr << " is not constant.";
  auto device_name_string_obj = device_name_node->value.as<StringValueObj>();
  CHECK(device_name_string_obj);
  std::string device_name_str = device_name_string_obj->value;
  return Device(static_cast<tvm::Device>((*str2dev)(device_name_str)));
}

/*!
 * \brief The helper function to mutate a call node to be in the target device.
 * \param node The input call node.
 * \param args The mutated args.
 * \param target_device_str A string of the target device.
 * \param device_arg_idx The argument index of the target device of the op.
 * \param default_vals The default values of the op. This array must have the same length of the
 * arguments. If an argument is required and does not have a default value, an undefined Expr has
 * to be provided as a placeholder.
 *
 * \return A mutated call node with the target device.
 */
Expr AssignDeviceHelper(const CallNode* node, const Array<Expr> args, std::string target_device_str,
                        size_t device_arg_idx, Array<Expr> default_vals) {
  Array<Expr> new_args;

  // Get the device of the current node. If not specified, the default is always CPU.
  static const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  Device call_device;
  if (device_arg_idx >= args.size()) {
    call_device = Device(static_cast<tvm::Device>((*str2dev)("cpu")));
  } else {
    if(args[device_arg_idx]->IsInstance<VarNode>()) {
      // the device argument corresponds to a var, ignore
      return GetRef<Call>(node);
    }
    call_device = GetDeviceFromConstExpr(args[device_arg_idx]);
  }

  // Get the target device.
  Device target_device = Device(static_cast<tvm::Device>((*str2dev)(target_device_str)));

  // Current node is not on the desired device, adjust the device argument.
  if (target_device.device_type() != call_device.device_type() || target_device.device_id() != call_device.device_id()) {
    for (auto arg_idx = 0; arg_idx < default_vals.size(); ++arg_idx) {
      if (!default_vals[arg_idx].defined()) {
        // Do nothing with required arguments.
        new_args.push_back(args[arg_idx]);
      } else if (arg_idx >= args.size() || arg_idx == device_arg_idx) {
        // Make up the default argument value.
        new_args.push_back(default_vals[arg_idx]);
      } else {
        // Optional argument is specified.
        new_args.push_back(args[arg_idx]);
      }
    }
    CHECK_EQ(new_args.size(), default_vals.size());
  } else {
    new_args = args;
  }
  return Call(node->op, new_args);
}

Expr AssignDeviceFullOp(const CallNode* node, const Array<Expr> args,
                        std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* fill value */
      Expr(),                                            /* shape */
      MakeConstant(StringValue::make("int")),            /* target dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 3, default_vals);
}

Expr AssignDeviceOneHotOp(const CallNode* node, const Array<Expr> args,
                          std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* indices */
      Expr(),                                            /* on_value */
      Expr(),                                            /* off_value */
      Expr(),                                            /* depth */
      MakeConstant(ScalarValue::make(-1)),               /* axis */
      MakeConstant(StringValue::make("int")),            /* dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 6, default_vals);
}

Expr AssignDeviceInitOp(const CallNode* node, const Array<Expr> args,
                        std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* shape */
      MakeConstant(StringValue::make("int")),            /* dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 2, default_vals);
}

Expr AssignDeviceArangeOp(const CallNode* node, const Array<Expr> args,
                          std::string target_device_str) {
  Array<Expr> default_vals = {
      Expr(),                                            /* start */
      Expr(),                                            /* stop */
      Expr(),                                            /* step */
      MakeConstant(StringValue::make("float")),          /* dtype */
      MakeConstant(StringValue::make(target_device_str)) /* device */
  };
  return AssignDeviceHelper(node, args, target_device_str, 4, default_vals);
}

int CheckDeviceVarHelper(const CallNode* node, const Array<Expr> args, std::string target_device_str,
                        size_t device_arg_idx, const VarMap<Expr>& params_map) {
  // Get the device of the current node. If not specified, the default is always CPU.
  static const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  CHECK(device_arg_idx < args.size()) << "Default value should have already been populated.";
  if(args[device_arg_idx]->IsInstance<VarNode>()) {
    auto device_var = Downcast<Var>(args[device_arg_idx]);
    CHECK(params_map.count(device_var)) << "Cannot find device var in params map.";
    auto arg = params_map.at(device_var);
    Device call_device = GetDeviceFromConstExpr(arg);
    Device target_device = Device(static_cast<tvm::Device>((*str2dev)(target_device_str)));
    if(target_device.device_type() != call_device.device_type() ||
       target_device.device_id() != call_device.device_id()) {
      return device_arg_idx;
    }
  }
  return -1;
}

typedef Expr (*AssignDeviceOpFuncType)(const CallNode* node, const Array<Expr> args,
                                       std::string target_device);
std::unordered_map<String, AssignDeviceOpFuncType> fmap = {
    {"raf.op.full", &AssignDeviceFullOp},
    {"raf.op.one_hot", &AssignDeviceOneHotOp},
    {"raf.op.zeros", &AssignDeviceInitOp},
    {"raf.op.ones", &AssignDeviceInitOp},
    {"raf.op.arange", &AssignDeviceArangeOp}};

std::unordered_map<String, int> fmap_idx = {
    {"raf.op.full", 3},
    {"raf.op.one_hot", 6},
    {"raf.op.zeros", 2},
    {"raf.op.ones", 2},
    {"raf.op.arange", 4}};

Expr MutateConstant(const RelayConstantNode* node, const std::string& device_str) {
  auto value = Downcast<Value>(ConstantExtractValue(GetRef<Constant>(node)));

  // Only focus on constant tensor.
  if (value.as<TensorValueObj>()) {
    DLTensor* dlt = value;

    const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
    tvm::Device target_tvm_ctx = (*str2dev)(device_str);
    Device target_device = Device(target_tvm_ctx);

    // Do nothing if the constant is already on the target device.
    if (target_tvm_ctx.device_type == dlt->device.device_type) {
      return GetRef<Expr>(node);
    }

    std::vector<int64_t> shape;
    DType dtype = DType(DLDataType(dlt->dtype));
    for (auto i = 0; i < dlt->ndim; ++i) {
      shape.push_back(dlt->shape[i]);
    }

    auto array = tvm::runtime::NDArray::Empty(shape, dtype, target_device);

    // Move tensor to the target device.
    array.CopyFrom(dlt);
    auto tv = TensorValue::Assemble(target_device, dtype, shape);
    tv->tensor = std::move(array);
    return MakeConstant(tv);
  }
  return GetRef<Expr>(node);
}

class AssignDevicePrimitiveHelper : public ExprMutator {
public:
  AssignDevicePrimitiveHelper(std::string device) : device_str_(device) {}

  std::pair<Expr, Array<Expr>> Run(const Function& func, const Array<Expr> args) {
    for(int i=0; i < func->params.size(); i++) {
      auto param = func->params[i];
      param2arg_[param] = args[i];
    }
    auto new_body = this->Mutate(func->body);
    auto new_func = WithFields(func, func->params, new_body);
    Array<Expr> new_args;
    for(auto arg: args) {
      if(arg_to_mutate_.count(arg)) {
        new_args.push_back(MakeConstant(StringValue::make(device_str_)));
      } else {
        new_args.push_back(arg);
      }
    }
    return {new_func, new_args};
  }

  Expr VisitExpr_(const RelayConstantNode* node) final {
    return MutateConstant(node, device_str_);
  }

  Expr VisitExpr_(const CallNode* node) final {
    if (node->op.as<OpNode>() == nullptr) {
      return ExprMutator::VisitExpr_(node);
    }

    const Op& node_op = Downcast<Op>(node->op);
    CHECK(node_op.defined());
    auto base_op = op::IsDialectOp(node_op) ? op::GetBaseOp(node_op) : node_op;

    if (fmap_idx.count(base_op->name) != 0) {
      Array<Expr> visited_args;
      for (auto arg : node->args) {
        visited_args.push_back(this->Mutate(arg));
      }
      auto new_call = Downcast<Call>((*fmap[base_op->name])(node, visited_args, device_str_));
      int var_mutate_idx = CheckDeviceVarHelper(new_call.as<CallNode>(), new_call->args, device_str_,
                                                fmap_idx[base_op->name], param2arg_);
      if(var_mutate_idx != -1) {
        auto param_to_mutate = Downcast<Var>(new_call->args[var_mutate_idx]);
        CHECK(param2arg_.count(param_to_mutate));
        arg_to_mutate_[param2arg_.at(param_to_mutate)] = true;
      }
      return new_call;
    }
    return ExprMutator::VisitExpr_(node);
  }

private:
  /*! \brief The target device string. */
  std::string device_str_;
  VarMap<Expr> param2arg_;
  ExprMap<bool> arg_to_mutate_;
};

class DeviceAssigner : public ExprMutator {
 public:
  DeviceAssigner(std::string device) : device_str_(device){};

  Expr VisitExpr_(const RelayConstantNode* node) final {
    return MutateConstant(node, device_str_);
  }

  Expr VisitExpr_(const CallNode* node) final {
    if (auto func_node = node->op.as<FunctionNode>()) {
      if(func_node->HasNonzeroAttr(attr::kPrimitive)) {
        // encountered fused primitive function
        Expr new_func;
        Array<Expr> new_args;
        std::tie(new_func, new_args) = AssignDevicePrimitiveHelper(device_str_).Run(GetRef<Function>(func_node), node->args);
        return Call(new_func, new_args);
      }
    }
    
    if (node->op.as<OpNode>() == nullptr) {
      return ExprMutator::VisitExpr_(node);
    }

    const Op& node_op = Downcast<Op>(node->op);
    CHECK(node_op.defined());
    auto base_op = op::IsDialectOp(node_op) ? op::GetBaseOp(node_op) : node_op;

    if (fmap.count(base_op->name) != 0) {
      Array<Expr> visited_args;
      for (auto arg : node->args) {
        visited_args.push_back(this->Mutate(arg));
      }
      return (*fmap[base_op->name])(node, visited_args, device_str_);
    }
    return ExprMutator::VisitExpr_(node);
  }

  Expr VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      this->memo_[GetRef<Expr>(op)] = Let(var, value, body);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

 private:
  /*! \brief The target device string. */
  std::string device_str_;
};
}  // namespace assign_device

Pass AssignDevice(std::string device) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto assigner = assign_device::DeviceAssigner(device);
    return Downcast<Function>(assigner.Mutate(f));
  };
  return CreateRAFFunctionPass(pass_func, 0, "AssignDevice", {});
}

Expr AssignDeviceForExpr(Expr expr, std::string device) {
  auto assigner = assign_device::DeviceAssigner(device);
  return assigner.Mutate(expr);
}

RAF_REGISTER_GLOBAL("raf.pass_.AssignDevice").set_body_typed(AssignDevice);

}  // namespace pass
}  // namespace raf
