/*!
 * Copyright (c) 2020 by Contributors
 * \file apply_backward.cc
 * \brief apply backward graph in the forward pass
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace apply_backward {

using namespace raf::ir;

class ApplyBackwardFunc : public ExprVisitor {
 public:
  void VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* let) {
      if (!in_closure_ && let->body.get() == let->var.get()) {
        // The return stmt in original function
        if (auto tup = let->value.as<TupleNode>()) {
          old_ret_ = tup->fields;
        }
      } else if (let->value.as<FunctionNode>()) {
        // The backward graph closure
        closure_var_ = let->var.get();
        closure_var = let->var;
        closure_expr_ = Downcast<Function>(let->value);
        ell_.exprs.push_back(let->value);
        ell_.vars.push_back(let->var);
      } else if (!in_closure_) {
        ell_.exprs.push_back(let->value);
        ell_.vars.push_back(let->var);
      }
      this->VisitExpr(let->var);
      this->VisitExpr(let->value);
    };
    auto post_visit = [this](const LetNode* let) {
      this->VisitExpr(let->body);
      this->visit_counter_[let] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const FunctionNode* func_node) final {
    closure_params_ = func_node->params;
    in_closure_ = true;
    VisitExpr(func_node->body);
    in_closure_ = false;
  }

  Function Apply(Function func) {
    VisitExpr(func->body);
    if (closure_var_ == nullptr) {
      // No closure found in the function
      return func;
    }

    // Check if the closure is in the return tuple
    bool return_closure = false;
    Array<Expr> ret_tup = old_ret_;
    for (auto it = ret_tup.begin(); it != ret_tup.end(); ++it) {
      if ((*it).get() == closure_var_) {
        ret_tup.erase(it);
        return_closure = true;
        break;
      }
    }
    if (!return_closure) {
      return func;
    }
    // Append the closure params into the function params
    auto new_params = func->params;
    Array<Expr> closure_params_as_call_args;
    for (auto closure_param : closure_params_) {
      auto new_param = MakeVar("closure_"+closure_param->name_hint(), closure_param->type_annotation);
      new_params.push_back(new_param);
      closure_params_as_call_args.push_back(new_param);
    }
    auto gradient = MakeVar("gradient", {});
    ell_.vars.push_back(gradient);
    ell_.exprs.push_back(Call(closure_var, closure_params_as_call_args));
    ret_tup.push_back(gradient);
    // Add an extra return stmt
    Var ret_var = MakeVar("ret", {});
    ell_.vars.push_back(ret_var);
    ell_.exprs.push_back(Tuple(ret_tup));
    ell_.ret = ret_var;
    return Function(new_params, ell_.AsExpr(), {}, {});
  }

 private:
  /*! \brief Original return value for function */
  Array<Expr> old_ret_;
  /*! \brief Closure parameters */
  Array<Var> closure_params_;
  /*! \brief Pointers to store the var for closure */
  const VarNode* closure_var_ = nullptr;
  Expr closure_var;
  Function closure_expr_;
  /*! \brief ExplicitLetList to rebuild the function */
  ExplicitLetList ell_;
  /*! \brief Indicate whether it is in closure */
  bool in_closure_ = false;
};
}  // namespace apply_backward

Pass ApplyBackward() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return apply_backward::ApplyBackwardFunc().Apply(f);
  };
  return CreateRAFFunctionPass(pass_func, 1, "ApplyBackward", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ApplyBackward").set_body_typed(ApplyBackward);

}  // namespace pass
}  // namespace raf
