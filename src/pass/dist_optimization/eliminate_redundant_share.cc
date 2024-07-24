/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file eliminate_redundant_share.cc
 * \brief A pass for eliminating redundant share variables of MoE parameters.
 */
#include <vector>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "../let_list.h"
#include "../common.h"
#include "../../common/shape_utils.h"

namespace raf {
namespace pass {
namespace eliminate_redundant_share {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::pass;

template <class T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

Expr GetShare(Expr expr) {
  if (expr->IsInstance<ExtendedVarNode>()) {
    return expr.as<ExtendedVarNode>()->may_share;
  } else {
    return Var();
  }
}

Expr SubstituteArgsIfShare(Expr expr, const ExprMap<Expr>& expr_map) {
  if (expr->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr);
    Array<Expr> args = {};
    for (auto arg : call->args) {
      auto share = GetShare(arg);
      if (expr_map.count(share)) {
        args.push_back(expr_map.at(share));
      } else {
        args.push_back(arg);
      }
    }
    return Call(call->op, args);
  } else if (expr->IsInstance<TupleNode>()) {
    Tuple tuple = Downcast<Tuple>(expr);
    Array<Expr> fields = {};
    for (auto field : tuple->fields) {
      auto share = GetShare(field);
      if (expr_map.count(share)) {
        fields.push_back(expr_map.at(share));
      } else {
        fields.push_back(field);
      }
    }
    return Tuple(fields);
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    TupleGetItem tgi = Downcast<TupleGetItem>(expr);
    Expr tuple = tgi->tuple;
    auto share = GetShare(tuple);
    if (expr_map.count(share)) {
      tuple = expr_map.at(share);
    }
    return TupleGetItem(tuple, tgi->index);
  } else {
    return expr;
  }
}

class RedundantSharesEliminator : public ExprVisitor {
 public:
  RedundantSharesEliminator(const FunctionNode* func) : func_(func) {
    ell_ = ExplicitLetList::make(func->body);
    for (int i = 0; i < ell_->vars.size(); ++i) {
      auto var = ell_->vars[i];
      auto share = GetShare(var);
      if (share.defined() && !orig_first_share_.count(share)) {
        // record the first share variable.
        orig_first_share_[share] = var;
      }
    }
  }

  Function Eliminate() {
    std::unordered_set<int> indices_to_delete = {};
    for (int i = 0; i < ell_->exprs.size(); ++i) {
      auto share = GetShare(ell_->vars[i]);
      if (orig_first_share_.count(share) && ell_->vars[i] != orig_first_share_.at(share)) {
        // delete share varaible if it is not the first one.
        indices_to_delete.insert(i);
        continue;
      }
      ell_->exprs[i] = SubstituteArgsIfShare(ell_->exprs[i], orig_first_share_);
    }
    std::vector<Var> vars = {};
    std::vector<Expr> exprs = {};
    for (int i = 0; i < ell_->exprs.size(); ++i) {
      if (!indices_to_delete.count(i)) {
        vars.push_back(ell_->vars[i]);
        exprs.push_back(ell_->exprs[i]);
      }
    }
    ell_->vars = vars;
    ell_->exprs = exprs;
    return Downcast<Function>(pass::InferType(Function(func_->params, ell_->AsExpr(), {}, {})));
  }

 private:
  std::unique_ptr<ExplicitLetList> ell_;
  const FunctionNode* func_ = nullptr;
  ExprMap<Expr> orig_first_share_ = {};
};

} // namespace eliminate_redundant_share

Pass EliminateRedundantShare() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return eliminate_redundant_share::RedundantSharesEliminator(f.operator->()).Eliminate();
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 0, "EliminateRedundantShare", {});
  PassInfo pass_info(0, "EliminateRedundantShare", {});
  return RAFSequential({func_pass}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.EliminateRedundantShare").set_body_typed(EliminateRedundantShare);

Function EliminateRudundantShare(const FunctionNode* func) {
  return eliminate_redundant_share::RedundantSharesEliminator(func).Eliminate();
}

} // namespace pass
} // namespace raf