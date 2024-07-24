/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file enforce_sync.cc
 * \brief Enforce synchronization between ops in multiple streams.
 */
#include "raf/device.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/dist_context.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/stream_pool.h"
#include "./common.h"
#include "./../common/shape_utils.h"
#include "./let_list.h"

namespace raf {
namespace pass {
namespace fix_fp16 {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using raf::distributed::DistContext;
using namespace raf::analysis;
using stream_pool::StreamTagEnum;
using common::shape_utils::BytesCompactTensor;

using OpSet = std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

static const std::unordered_set<std::string> never_cast_ops = {
    "raf.op.tanh", "raf.op.tvm.tanh", "raf.op.tanh_dx", "raf.op.tvm.tanh_dx",
};

class NeverCastOpFinder : public ExprVisitor {
public:
  void VisitExpr_(const CallNode* op) {
    if (const OpNode* op_node = op->op.as<OpNode>()) {
      if (never_cast_ops.count(op_node->name)) {
        has_never_cast_ops = true;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  bool has_never_cast_ops = false;
};

class FuncParamCaster : public ExprMutator {
public:
  Expr VisitExpr_(const VarNode* op) {
    if (op->type_annotation.as<TensorTypeNode>() == nullptr) {
      return ExprMutator::VisitExpr_(op);
    }
    auto input_type = Downcast<TensorType>(op->type_annotation);
    if (input_type->dtype.code() == kDLFloat && input_type->dtype.bits() == 16) {
      // replace with fp32
      return Var(op->name_hint(), TensorType(input_type->shape, DataType::Float(32)), op->span);
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }
};

class FP16Fixer : public ExprMutator {
public:
  Expr VisitExpr_(const CallNode* op) {
    static const Op& cast_op = Op::Get("raf.op.cast");
    // we only care about calls which contains the never cast ops
    bool has_never_cast_ops = false;
    if (const OpNode* op_node = op->op.as<OpNode>()) {
      // call to a builtin op
      has_never_cast_ops = never_cast_ops.count(op_node->name);
    } else {
      // call to a function
      NeverCastOpFinder finder;
      finder.VisitExpr(op->op);
      has_never_cast_ops = finder.has_never_cast_ops;
      if (has_never_cast_ops) {
        LOG(INFO) << "Found call to a never cast op: " << op->op;
      }
    }
    if (!has_never_cast_ops) {
      return ExprMutator::VisitExpr_(op);
    }
    // we have a call to a never cast op, we need to cast the inputs to fp32
    bool casted = false;
    Array<Expr> new_args;
    for (int i=0; i<op->args.size(); i++) {
      if (op->args[i]->checked_type().as<TensorTypeNode>() == nullptr) {
        // not a tensor, we don't care for now.
        new_args.push_back(op->args[i]);
        continue;
      }
      if (op->args[i]->checked_type() == IncompleteType()) {
        // incomplete type, we don't care for now.
        LOG(FATAL) << "Encountered incomplete type for op " << op << " arg " << i;
      }
      auto input_type = Downcast<TensorType>(op->args[i]->checked_type());
      if (input_type->dtype.code() == kDLFloat && input_type->dtype.bits() == 16) {
        // need to cast
        auto casted_input = Call(cast_op, {op->args[i], MakeConstant(StringValue::make("float32"))});
        new_args.push_back(casted_input);
        casted = true;
      }
    }
    if (casted) {
      if (op->op->IsInstance<FunctionNode>()) {
        // also need to modify func params
        auto func = Downcast<Function>(op->op);
        FuncParamCaster caster;
        func = Downcast<Function>(caster.Mutate(func));
        return Call(cast_op, {Call(func, new_args, op->attrs, op->type_args), MakeConstant(StringValue::make("float16"))});
      }
      return Call(cast_op, {Call(op->op, new_args, op->attrs, op->type_args), MakeConstant(StringValue::make("float16"))});
    } else {
      return ExprMutator::VisitExpr_(op);
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr Mutate(const Expr& expr) {
    return ExprMutator::Mutate(expr);
  }
};

}  // namespace fix_fp16

Pass FixFP16() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(fix_fp16::FP16Fixer().Mutate(f));
  };
  Pass func_pass = CreateRAFFunctionPass(pass_func, 0, "FixFP16", {});
  PassInfo pass_info(0, "FixFP16", {});
  return RAFSequential({InferType(), func_pass, }, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.FixFP16").set_body_typed(FixFP16);

}  // namespace pass
}  // namespace raf
