/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file extract_binding.cc
 * \brief Extracting a relay body from frontend defined binding
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/binding.h"

namespace raf {
namespace pass {
namespace rename_vars {

using namespace raf::ir;

struct RenameVarsMutator : public ExprMutator {
 public:
  explicit RenameVarsMutator(const Map<String, Var>& named_vars): in_main_(false) {
    for (const auto& iter : named_vars) {
      const auto* var = iter.second.as<ExtendedVarNode>();
      var_map_.Set(iter.second,
                   raf::ir::MakeVar(iter.first, iter.second->type_annotation, var->may_share));
    }
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map_.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const FunctionNode* node) final {
    if (!in_main_) {
      in_main_ = true;
      return ExprMutator::VisitExpr_(node);
    }
    // in closure, mutate return and body only
    for (auto param : node->params) {
      var_map_.Set(param, param);
    }
    auto ret_type = this->VisitType(node->ret_type);
    auto body = this->Mutate(node->body);

    if (ret_type.same_as(node->ret_type) && body.same_as(node->body)) {
      return GetRef<Expr>(node);
    } else {
      return Function(node->params, body, ret_type, node->type_params, node->attrs, node->span);
    }
  }

  Expr VisitExpr_(const LetNode* node) final {
    auto pre_visit = [this](const LetNode* node) {
      const Var& var = node->var;
      CHECK_EQ(var_map_.count(var), 0) << "IR is malformed: cannot bind var twice";
      const auto* vn = var.as<ExtendedVarNode>();
      Var may_share = vn->may_share;
      Var new_var =
          raf::ir::MakeVar("a" + std::to_string(++num_bound_var_), var->type_annotation,
                           may_share.defined() ? Downcast<Var>(var_map_.at(may_share)) : may_share);
      var_map_.Set(var, new_var);
      this->Mutate(node->value);
    };
    auto post_visit = [this](const LetNode* node) {
      Var var = Downcast<Var>(node->var);
      Expr value = this->Mutate(node->value);
      Expr body = this->Mutate(node->body);

      auto expr = GetRef<Expr>(node);
      if (var.same_as(node->var) && value.same_as(node->value) && body.same_as(node->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(Downcast<Var>(var_map_[var]), value, body);
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

 private:
  /*! \brief The counter of bound variables. */
  int num_bound_var_ = 0;
  bool in_main_ = false;
  /*! \brief Map from original var to the renamed var. */
  Map<Var, Expr> var_map_;
};

Expr RenameVars(Expr expr, Map<String, Var> named_vars) {
  return RenameVarsMutator(named_vars).Mutate(expr);
}

RAF_REGISTER_GLOBAL("raf.pass_.RenameVars").set_body_typed(RenameVars);
}  // namespace rename_vars
}  // namespace pass
}  // namespace raf
