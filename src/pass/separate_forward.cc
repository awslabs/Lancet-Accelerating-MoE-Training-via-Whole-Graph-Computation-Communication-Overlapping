/*!
 * Copyright (c) 2020 by Contributors
 * \file separate_forward.cc
 * \brief separate out the fw part of the graph returning all used intermediate output
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace separate_forward {

using namespace raf::ir;

class SeparateForwardFunc : public ExprVisitor {
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
        // we eliminate the backward closure here
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

    // record original input params
    auto orig_input_params = func->params;
    std::set<Var> orig_input_params_set;
    for (auto p: orig_input_params) orig_input_params_set.insert(p);

    // find the set of forward activation used by bw closure
    Array<Var> fw_output_used;
    auto free_vars = ::tvm::relay::FreeVars(closure_expr_);
    for (auto x: free_vars) {
      if (!orig_input_params_set.count(x)) {
        fw_output_used.push_back(x);
      }
    }
    for(auto& var: fw_output_used) {
      ret_tup.push_back(var);
    }

    // Add an extra return stmt
    Var ret_var = MakeVar("ret", {});
    ell_.vars.push_back(ret_var);
    ell_.exprs.push_back(Tuple(ret_tup));
    ell_.ret = ret_var;
    return Function(func->params, ell_.AsExpr(), {}, {});
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
}  // namespace separate_forward

Pass SeparateForward() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return separate_forward::SeparateForwardFunc().Apply(f);
  };
  return CreateRAFFunctionPass(pass_func, 1, "SeparateForward", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.SeparateForward").set_body_typed(SeparateForward);

}  // namespace pass
}  // namespace raf
