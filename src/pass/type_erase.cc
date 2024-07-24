/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file type_erase.cc
 * \brief The type erase pass erases the checked type and function return type.
 */

#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/pass_manager.h"

namespace raf {
namespace pass {
namespace type_erase {

using namespace raf::op;
using namespace raf::value;

class TypeEraser : public ExprMutator {
 public:
  Expr VisitExpr(const Expr& expr) final {
    auto ret = ExprMutator::VisitExpr(expr);
    ret->checked_type_ = Type();
    return ret;
  }

  Expr VisitExpr_(const LetNode* op) override {
    // avoids stack overflow
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

  Expr VisitExpr_(const FunctionNode* node) final {
    Array<Var> new_params;
    if(in_closure_) {
      for(auto expr: node->params) {
        new_params.push_back(Downcast<Var>(this->Mutate(expr)));
      }
    } else {
      // preserve the type for main function inputs
      for(auto expr: node->params) {
        new_params.push_back(expr);
      }
    }
    in_closure_ ++;
    auto new_body = this->Mutate(node->body);
    in_closure_ --;
    return Function(new_params, new_body, Type(), {}, node->attrs, node->span);
  }
  int in_closure_ = 0;
};

}  // namespace type_erase

Expr EraseType(const Expr& expr, bool is_main) {
  auto eraser = type_erase::TypeEraser();
  if(!is_main) {
    eraser.in_closure_++;
  }
  return eraser.Mutate(expr);
}

Pass EraseType() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(type_erase::TypeEraser().Mutate(f));
  };
  return CreateRAFFunctionPass(pass_func, 1, "EraseType", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.EraseType").set_body_typed([]() { return EraseType(); });

}  // namespace pass
}  // namespace raf
