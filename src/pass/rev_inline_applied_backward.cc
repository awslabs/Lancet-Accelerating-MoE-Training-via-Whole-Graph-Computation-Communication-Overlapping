/*!
 * Copyright (c) 2020 by Contributors
 * \file rev_inline_applied_backward.cc
 * \brief inlining backward graph in the forward pass
 */
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace rev_inline_applied_backward {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::analysis;
using Node = DependencyGraph::Node;
using LinkNode = tvm::support::LinkNode<Node*>;
using NodeExprMap = std::unordered_map<const Node*, Expr>;
using VarExprMap = tvm::runtime::Map<Var, Expr>;
using ExprVarMap = tvm::runtime::Map<Expr, Var>;
using VarIdxMap = std::unordered_map<Var, int>;

void RotateInplace(ExplicitLetList& ell, int k) {
  std::function<int(int, int)> calc_gcd = [&](int a, int b) {return b == 0 ? a : calc_gcd(b, a%b);};
  int n = ell.vars.size();
  int gcd = calc_gcd(n, k);
  for (int offset = 0; offset < gcd; offset++) {
    // rotate one partition
    Var tmp_var = ell.vars[offset];
    Expr tmp_expr = ell.exprs[offset];
    int current_idx = offset;
    while (true) {
        int next_idx = current_idx + k;
        if (next_idx>=n) next_idx -= n;
        if (next_idx == offset)
            break;
        ell.vars[current_idx] = ell.vars[next_idx];
        ell.exprs[current_idx] = ell.exprs[next_idx];
        current_idx = next_idx;
    }
    ell.vars[current_idx] = tmp_var;
    ell.exprs[current_idx] = tmp_expr;
  }
}

void ReplaceVars(ExplicitLetList& ell, const Map<Var, Var>& var_var_map, const std::set<Var>* filter = nullptr) {
  for(size_t i=0; i < ell.vars.size(); i++) {
    if(filter == nullptr || filter->count(ell.vars[i])) {
      // modify input args for Call, Tuple, TupleGetItem
      if(auto call_node = ell.exprs[i].as<CallNode>()) {
        Array<Expr> args;
        for(auto& arg: call_node->args) {
          if(arg.as<VarNode>()) {
            auto arg_var = Downcast<Var>(arg);
            if(var_var_map.count(arg_var)) {
              args.push_back(var_var_map.at(arg_var));
            } else {
              args.push_back(arg_var);
            }
          } else {
            args.push_back(arg);
          }
        }
        ell.exprs[i] = Call(call_node->op, args, call_node->attrs, call_node->type_args);
      } else if (auto tuple_node = ell.exprs[i].as<TupleNode>()) {
        Array<Expr> fields;
        for(auto& arg: tuple_node->fields) {
          if(arg.as<VarNode>()) {
            auto arg_var = Downcast<Var>(arg);
            if(var_var_map.count(arg_var)) {
              fields.push_back(var_var_map.at(arg_var));
            } else {
              fields.push_back(arg_var);
            }
          } else {
            fields.push_back(arg);
          }
        }
        ell.exprs[i] = Tuple(fields);
      } else if (auto tgi_node = ell.exprs[i].as<TupleGetItemNode>()) {
        if(tgi_node->tuple.as<VarNode>()) {
          auto tuple = Downcast<Var>(tgi_node->tuple);
          if(var_var_map.count(tuple)) {
            ell.exprs[i] = TupleGetItem(var_var_map.at(tuple), tgi_node->index);
          }
        }
      }
    }
  }
}

void PutLineToFront(ExplicitLetList& ell, const Var& var) {
  int var_index = 0;
  for(size_t i=0; i < ell.vars.size(); i++) {
    if (ell.vars[i] == var) {
      var_index = i;
      break;
    }
  }
  Expr expr = ell.exprs[var_index];
  for(int i = var_index; i > 0; i--) {
    ell.vars[i] = ell.vars[i-1];
    ell.exprs[i] = ell.exprs[i-1];
  }
  ell.vars[0] = var;
  ell.exprs[0] = expr;
}

class RevInlineAppliedBackwardFunc : public ExprVisitor {
 public:
  void VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* let) {
      if (let->body.get() == let->var.get()) {
        if (!in_closure_) {
          // The return stmt in original function
          if (auto tup = let->value.as<TupleNode>()) {
            old_ret_ = tup->fields;
          }
        } else {
          // the return stmt in the grad closure
          if (auto tup = let->value.as<TupleNode>()) {
            closure_ret_ = tup->fields;
          }
        }
      } else if (let->value.as<FunctionNode>()) {
        // The backward graph closure
        closure_var_ = let->var.get();
        closure_expr_ = let->value;
      } else if (let->value.as<CallNode>() && let->value.as<CallNode>()->op.as<VarNode>()) {
        // closure call
        CHECK(let->value.as<CallNode>()->op.as<VarNode>() == closure_var_);
        for(auto& arg: let->value.as<CallNode>()->args) {
          if (func_params_set_.find(Downcast<Var>(arg)) == func_params_set_.end()) {
            CHECK(let->value.as<CallNode>()->args.size() == 1);
            has_closure_args_tuple_ = true;
            closure_arg_var_ = Downcast<Var>(arg);
          }
          closure_call_args_.push_back(Downcast<Var>(arg));
        }
        closure_call_var_ = let->var;
        ell_.exprs.push_back(let->value);
        ell_.vars.push_back(let->var);
      } else {
        if (in_closure_) {
          if(!first_bw_var_.defined()) {
            first_bw_var_ = let->var;
          }
          last_bw_var_ = let->var;
          bw_vars_.insert(let->var);
          bw_exprs_.insert(let->value);
        } else {
          if(last_bw_var_.defined()) {
            // passed bw closure
            update_vars_.insert(let->var);
            update_exprs_.insert(let->value);
          }
        }
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

  Function Inline(Function func) {
    for (int i=0; i < func->params.size(); i++) {
      func_params_set_.insert(func->params[i]);
    }
    VisitExpr(func->body);

    if (closure_var_ == nullptr) {
      // No closure found in the function
      return func;
    }

    Arena arena;
    DependencyGraph dfg = CreateDependencyGraph(&arena, func->body, false);
    NodeExprMap node_expr;
    for (auto& it : dfg.expr_node) {
      node_expr[it.second] = it.first;
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

    Var intermediate_tuple_var;
    Expr intermediate_tuple_expr;
    int intermediate_tuple_idx = -1;

    Node* closure_call_node = dfg.expr_node[closure_call_var_];
    if (closure_call_node->parents.head) {
      LinkNode* current = closure_call_node->parents.head;
      while(current != nullptr) {
        auto parent_node = current->value;
        auto parent_expr = node_expr[parent_node];
        if(parent_expr.as<TupleNode>()) {
          intermediate_tuple_expr = parent_expr;
          break;
        }
        current = current->next;
      }
    }

    VarExprMap var_expr;
    ExprVarMap expr_var;

    int first_bw_idx = -1;
    int closure_call_index = -1;
    
    
    for(size_t i=0; i<ell_.vars.size(); i++) {
      var_expr.Set(ell_.vars[i], ell_.exprs[i]);
      expr_var.Set(ell_.exprs[i], ell_.vars[i]);
      if (ell_.vars[i] == first_bw_var_) first_bw_idx = i;
      if (ell_.exprs[i] == intermediate_tuple_expr) {
        intermediate_tuple_var = ell_.vars[i];
        intermediate_tuple_idx = i;
      }
      if (ell_.vars[i] == closure_call_var_) closure_call_index = i;
    }

    bool has_second_tuple = false;
    if(LinkNode* tgi_link_node = dfg.expr_node[intermediate_tuple_var]->parents.head) {
      while(tgi_link_node && !node_expr.count(tgi_link_node->value)) {
        tgi_link_node = tgi_link_node->next;
      }
      if(tgi_link_node) {
        auto tgi_var_node = dfg.expr_node[expr_var[node_expr[tgi_link_node->value]]];
        for(LinkNode* tuple_link_node = tgi_var_node->parents.head; tuple_link_node!=nullptr; tuple_link_node=tuple_link_node->next) {
          if(node_expr.count(tuple_link_node->value)) {
            if(auto tuple_node = node_expr[tuple_link_node->value].as<TupleNode>()) {
              intermediate_tuple_expr = node_expr[tuple_link_node->value];
              has_second_tuple = true;
              break;
            }
          }
        }
      }
    }
    if (has_second_tuple) {
      for(size_t i=0; i<ell_.vars.size(); i++) {
        if (ell_.exprs[i] == intermediate_tuple_expr) {
          intermediate_tuple_var = ell_.vars[i];
          intermediate_tuple_idx = i;
        }
      }
    }

    CHECK_NE(intermediate_tuple_idx, -1);

    // erase closure call
    ell_.vars.erase(ell_.vars.begin() + closure_call_index);
    ell_.exprs.erase(ell_.exprs.begin() + closure_call_index);

    // erase intermediate tuple
    if (intermediate_tuple_idx > closure_call_index) {
      intermediate_tuple_idx --;
    }
    ell_.vars.erase(ell_.vars.begin() + intermediate_tuple_idx);
    ell_.exprs.erase(ell_.exprs.begin() + intermediate_tuple_idx);
    if(has_second_tuple) {
      // also remove the first tuple and intermediate tgis
      // assume they are consecutive in ANF
      for(int i=0;i<3;i++) {
        intermediate_tuple_idx --;
        ell_.vars.erase(ell_.vars.begin() + intermediate_tuple_idx);
        ell_.exprs.erase(ell_.exprs.begin() + intermediate_tuple_idx);
      }
    }

    // find weight update ops and their depended fw ops
    Array<Var> weight_updates;
    Array<Expr> weight_update_exprs;
    Array<Var> weight_vars;
    std::set<Var> depended_forward_vars; 

    for(int i=1; i<old_ret_.size(); i++) {
      weight_updates.push_back(Downcast<Var>(old_ret_[i]));
      auto weight_update_expr = var_expr.at(weight_updates.back());
      auto weight_update_callnode = weight_update_expr.as<CallNode>();
      CHECK(weight_update_callnode && weight_update_callnode->op.as<OpNode>());
      // check the weight update node is an update op
      auto weight_update_op = Downcast<Op>(weight_update_callnode->op);
      auto binary_u_fschema_index =
        Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex")[weight_update_op];
      CHECK(op::IsUpdateOp(weight_update_op));
      weight_vars.push_back(Downcast<Var>(weight_update_callnode->args[binary_u_fschema_index("out")]));
      weight_update_exprs.push_back(weight_update_expr);
    }

    for(int i=0; i<weight_updates.size(); i++) {
      Node* weight_var = dfg.expr_node[weight_vars[i]];
      if (weight_var->parents.head) {
        LinkNode* current = weight_var->parents.head;
        while(current != nullptr) {
          auto parent_node = current->value;
          auto parent_expr = node_expr[parent_node];
          if(!bw_exprs_.count(parent_expr) && !update_exprs_.count(parent_expr) && parent_expr != weight_update_exprs[i]) {
            depended_forward_vars.insert(expr_var[parent_expr]);
          }
          current = current->next;
        }
      }
    }

    // Start modifying graph.
    // 1. Remap GetTupleItem vars, since the original tuple is removed
    // 2. Reorder FW -> BW -> UPDATE to BW -> UPDATE -> FW
    // 3. Disconnect FW -> BW
    //    1. Create a new variable for each `fw_output_used`
    //    2. Add the newly created variables to function parameters
    //    3. Replace original vars in BW exprs with new ones.
    // 4. Reconnect BW -> FW by replacing the weight vars in 
    //    `depended_forward_vars` with weight update outputs.
    // 5. Replace closure param `closure_params_` to input dy (`closure_call_args_`)
    // 6. Create new return 

    // 1
    std::set<Expr> tgis_to_remove;
    Map<Var, Var> grad_tgi;
    CHECK(intermediate_tuple_expr.as<TupleNode>());
    Node* int_tuple_node = dfg.expr_node[intermediate_tuple_var];
    if (int_tuple_node->parents.head) {
      LinkNode* current = int_tuple_node->parents.head;
      while(current != nullptr) {
        auto parent_node = current->value;
        auto parent_expr = node_expr[parent_node];
        // parent_tgi: tgis for the (loss, gradients) tuple node
        if(auto parent_tgi = parent_expr.as<TupleGetItemNode>()) {
          if (parent_tgi->index == 1) {
            // tgi for gradient tuple
            tgis_to_remove.insert(parent_expr);
            // gradient_tuple_node: node in dfg corresponding to the var of gradient tgi
            Node* gradient_tuple_node = dfg.expr_node[expr_var[parent_expr]];
            if (gradient_tuple_node->parents.head) {
              LinkNode* p_current = gradient_tuple_node->parents.head;
              while(p_current != nullptr) {
                // p_parent: tgi for getting gradient from (gradients) tuple
                auto p_parent_node = p_current->value;
                auto p_parent_expr = node_expr[p_parent_node];
                if(auto p_parent_tgi = p_parent_expr.as<TupleGetItemNode>()) {
                  tgis_to_remove.insert(p_parent_expr);
                  grad_tgi.Set(expr_var[p_parent_expr], Downcast<Var>(closure_ret_[p_parent_tgi->index]));
                }
                p_current = p_current->next;
              }
            }
          } else {
            CHECK_EQ(parent_tgi->index, 0);
            tgis_to_remove.insert(parent_expr);
          }
        }
        current = current->next;
      }
    }
    // remove tgis
    for(size_t i=0; i<ell_.vars.size(); ) {
      if(tgis_to_remove.count(ell_.exprs[i])) {
        ell_.vars.erase(ell_.vars.begin() + i);
        ell_.exprs.erase(ell_.exprs.begin() + i);
        continue; // don't increment i
      }
      i++;
    }
    // replace tgi variables
    ReplaceVars(ell_, grad_tgi);

    // 2
    RotateInplace(ell_, first_bw_idx);

    // 3.1 & 3.2
    Array<Var> new_fw_activations;
    Map<Var, Var> activation_map;
    for(auto& var: fw_output_used) {
      Type var_type = var->type_annotation;
      if(var->checked_type_.defined()) {
        var_type = var->checked_type();
      }
      auto new_var = MakeVar("extra_input_"+var->name_hint(), var_type);
      new_fw_activations.push_back(new_var);
      orig_input_params.push_back(new_var);
      activation_map.Set(var, new_var);
    }

    // 3.3
    ReplaceVars(ell_, activation_map, &bw_vars_);

    // 4
    Map<Var, Var> update_fw_map;
    for(size_t i=0; i<weight_updates.size(); i++) {
      update_fw_map.Set(weight_vars[i], weight_updates[i]);
    }
    ReplaceVars(ell_, update_fw_map, &depended_forward_vars);

    // 5
    Map<Var, Var> closure_param_map;
    CHECK_EQ(closure_params_.size(), closure_call_args_.size());
    for(size_t i=0; i<closure_params_.size(); i++) {
      closure_param_map.Set(closure_params_[i], closure_call_args_[i]);
    }
    ReplaceVars(ell_, closure_param_map);

    // put closure arg var in the beginning
    if (has_closure_args_tuple_) {
      PutLineToFront(ell_, closure_arg_var_);
    }

    // 6
    Var loss = ell_.vars[ell_.vars.size()-1];
    Var new_ret = MakeVar("ret", {});
    Array<Expr> new_ret_expr;
    new_ret_expr.push_back(loss);
    for(auto fw_out: fw_output_used) {
      new_ret_expr.push_back(fw_out);
    }
    ell_.vars.push_back(new_ret);
    ell_.exprs.push_back(Tuple(new_ret_expr));
    ell_.ret = new_ret;

    return Function(orig_input_params, ell_.AsExpr(), {}, {});
  }

 private:
  /*! \brief Original return value for function */
  Array<Expr> old_ret_;
  Array<Expr> closure_ret_;
  /*! \brief Closure parameters */
  Array<Var> closure_params_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> func_params_set_;
  bool has_closure_args_tuple_;
  Var closure_arg_var_;
  /*! \brief Pointers to store the var for closure */
  const VarNode* closure_var_ = nullptr;
  Expr closure_expr_;
  /*! \brief ExplicitLetList to rebuild the function */
  ExplicitLetList ell_;
  std::set<Var> bw_vars_;
  std::set<Expr> bw_exprs_;
  std::set<Var> update_vars_;
  std::set<Expr> update_exprs_;
  Array<Var> closure_call_args_;
  Var closure_call_var_;
  /*! \brief Indicate whether it is in closure */
  bool in_closure_ = false;
  Var first_bw_var_;
  Var last_bw_var_;
};
}  // namespace rev_inline_applied_backward

Pass RevInlineAppliedBackward() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return rev_inline_applied_backward::RevInlineAppliedBackwardFunc().Inline(f);
  };
  return CreateRAFFunctionPass(pass_func, 1, "RevInlineAppliedBackward", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.RevInlineAppliedBackward").set_body_typed(RevInlineAppliedBackward);

}  // namespace pass
}  // namespace raf
