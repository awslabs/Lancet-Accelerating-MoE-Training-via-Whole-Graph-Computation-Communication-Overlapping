/*!
 * Copyright (c) 2021 by Contributors
 * \file tuple_fusion.h
 * \brief Implements transformations to implement tuple based fusion.
 */
#pragma once
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <tvm/relay/expr.h>
#include <relay/transforms/pass_utils.h>
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/dist_context.h"
#include "raf/op_utils.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "./scheduler_common.h"
#include "../common.h"
#include "../let_list.h"
#include "../stream_schedule.h"
#include "raf/stream_pool.h"

namespace raf {
namespace pass {
namespace tuple_fusion {

using namespace raf::ir;
using namespace raf::analysis;
using namespace raf::pass::scheduler_common;
using tvm::relay::WithFields;
using NodeExprMap = std::unordered_map<const Node*, Expr>;
using ExprIdxMap = std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual>;
using VarIdxMap = std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>;

class ClosureMutator : ExprMutator {
public:
  explicit ClosureMutator(const VarMap<Var>& param_map, Var fused_param, int offset):
    param_map_(param_map), fused_param_(fused_param), offset_(offset) {}

  Expr Run(Expr expr) {
    return this->Mutate(expr);
  }

  Expr VisitExpr_(const VarNode* var) {
    auto var_ = GetRef<Var>(var);
    if (param_map_.count(var_)) {
      return param_map_.at(var_);
    } else if (var_map_.count(var_)) {
      return var_map_.at(var_);
    } else {
      Var new_var = MakeVar(var_->name_hint(), var_->type_annotation, {});
      new_var->checked_type_ = var_->checked_type_;
      var_map_[var_] = new_var;
      return new_var;
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* tgi) {
    if (tgi->tuple == fused_param_) {
      auto t = this->Mutate(tgi->tuple);
      return TupleGetItem(t, tgi->index+offset_, tgi->span);
    } else {
      auto t = this->Mutate(tgi->tuple);
      if (tgi->tuple == t) {
        return GetRef<Expr>(tgi);
      } else {
        return TupleGetItem(t, tgi->index, tgi->span);
      }
    }
  }

  Expr VisitExpr_(const TupleNode* tuple) {
    Array<Expr> fields = {};
    for (auto field : tuple->fields) {
      fields.push_back(this->Mutate(field));
    }
    return Tuple(fields);
  }

  Expr VisitExpr_(const CallNode* call) {
    Array<Expr> args = {};
    for (auto arg : call->args) {
      args.push_back(this->Mutate(arg));
    }
    return Call(call->op, args);
  }

private:
  const VarMap<Var>& param_map_;
  // map a var in closure body to updated var.
  VarMap<Var> var_map_;
  Var fused_param_;
  int offset_;
};

class TupleFusor : ExprMutator {
public:
  TupleFusor(Expr e, Expr op_a, Expr op_b): expr_(e), let_list_(ExplicitLetList::make(expr_)) {
    // populate expr -> idx and var -> idx map
    for(int i=0; i<let_list_->vars.size(); i++) {
      expr_idx_[let_list_->exprs[i]] = i;
      var_idx_[let_list_->vars[i]] = i;
    }

    std::tie(a_idx_, a_input_idx_) = getAndValidateInput(op_a);
    std::tie(b_idx_, b_input_idx_) = getAndValidateInput(op_b);

    // use the later input tuple as the fused tuple
    last_input_tuple_idx_ = std::max(a_input_idx_, b_input_idx_);
    auto a_input_tuple = let_list_->exprs[a_input_idx_].as<TupleNode>();
    b_args_offset_ = a_input_tuple->fields.size();

    for(int i=a_idx_; i<let_list_->vars.size(); i++) {
      // function may get reused. we only mutate a function if it takes a fused parameter
      auto expr = let_list_->exprs[i];
      if(auto call_node = expr.as<CallNode>()) {
        auto call_op = call_node->op;
        if(auto call_function = call_op.as<FunctionNode>()) {
          // check if call args contain reference to expr_a or expr_b
          for(int arg_idx = 0; arg_idx < call_node->args.size(); arg_idx++) {
            // arg_idx is referrin to the index of argument in the call node
            auto arg = call_node->args[arg_idx];
            if(auto arg_var_node = arg.as<VarNode>()) {
              auto arg_var = Downcast<Var>(arg);
              if (var_idx_.count(arg_var)) {
                // arg_var_idx is the index (in the let list) of the argument
                int arg_var_idx = var_idx_.at(arg_var);
                if (arg_var_idx == a_idx_) {
                  // no offset is needed
                  func_mutation_map_[i] = {arg_idx, 0};
                } else if (arg_var_idx == b_idx_) {
                  func_mutation_map_[i] = {arg_idx, b_args_offset_};
                }
              }
            }
          }
        }
      }
    }

    fused_name_ = fuseNodeNames(let_list_->vars[a_idx_]->name_hint(), let_list_->vars[b_idx_]->name_hint());
  }

  // During mutation, we need to mutate 
  //  1. the input to op_b, which is a tuple node
  //  2. all TupleGetItems that originally references the output of a and b
  //     for those originally referencing a, we directly point them to b
  //     for those originally referencing b, we need to offset their index
  //  3. all call nodes that originally references the output of a (point them to b)

  Expr VisitExpr_(const VarNode* node) override {
    // for the var node corresponding to op_b, we rename it
    auto var = GetRef<Var>(node);
    if(var_idx_.count(var)) {
      auto new_value = this->Mutate(let_list_->exprs[var_idx_.at(var)]);
      std::string new_name;
      if(var_idx_.at(var) == b_idx_) {
        new_name = fused_name_;
      } else {
        new_name = var->name_hint();
      }
      auto new_var = MakeVar(new_name, new_value->checked_type_);
      new_var->checked_type_ = new_value->checked_type_;
      return new_var;
    }
    return var;
  }

  Expr VisitExpr_(const TupleNode* tuple_node) override {
    auto tuple_expr = GetRef<Tuple>(tuple_node);
    if (expr_idx_.count(tuple_expr) && expr_idx_.at(tuple_expr) == last_input_tuple_idx_) {
      // this should be mutated to the fused tuple
      // get a and b's original input
      auto a_input_tuple = let_list_->exprs[a_input_idx_].as<TupleNode>();
      auto b_input_tuple = let_list_->exprs[b_input_idx_].as<TupleNode>();

      tvm::Array<Expr> fields;
      fields.reserve(a_input_tuple->fields.size() + b_input_tuple->fields.size());
      for (auto field : a_input_tuple->fields) {
        auto new_field = this->Mutate(field);
        fields.push_back(new_field);
        fused_args_type_.push_back(new_field->checked_type());
      }
      for (auto field : b_input_tuple->fields) {
        auto new_field = this->Mutate(field);
        fields.push_back(new_field);
        fused_args_type_.push_back(new_field->checked_type());
      }
      auto new_tuple = WithFields(GetRef<Tuple>(tuple_node), std::move(fields));
      new_tuple->checked_type_ = TupleType(fused_args_type_);
      return new_tuple;
    } else {
      auto new_tuple = ExprMutator::VisitExpr_(tuple_node);
      new_tuple->checked_type_ = tuple_node->checked_type_;
      return new_tuple;
    }
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    return GetRef<Function>(func_node);
  }

  Expr MutateFunction(const FunctionNode* func_node, int mutate_param_idx, int fused_param_offset) {
    // we update the type of function params here
    Array<Var> new_params;
    Array<Type> new_param_types;
    VarMap<Var> mutated_param_map;
    for(int param_idx = 0; param_idx < func_node->params.size(); param_idx++) {
      auto old_param_var = func_node->params[param_idx];
      Type new_type;
      if(param_idx == mutate_param_idx) {
        new_type = TupleType(fused_args_type_);
      } else {
        new_type = old_param_var->type_annotation;
      }
      auto new_var = MakeVar("mutated_" + old_param_var->name_hint(), new_type);
      new_var->checked_type_ = new_type;
      mutated_param_map[old_param_var] = new_var;
      new_params.push_back(new_var);
      new_param_types.push_back(new_type);
    }

    auto new_body = ClosureMutator(
        mutated_param_map,
        func_node->params[mutate_param_idx],
        fused_param_offset).Run(func_node->body);

    auto new_func = Function(new_params, new_body, func_node->ret_type, func_node->type_params, func_node->attrs, func_node->span);
    new_func->checked_type_ = FuncType(new_param_types, func_node->ret_type, func_node->type_params, {});
    return new_func;
  }

  Expr VisitExpr_(const CallNode* c) override {
    // check if this is a function call that need mutation
    auto new_op = c->op;
    auto call = GetRef<Call>(c);
    if(expr_idx_.count(call) && func_mutation_map_.count(expr_idx_.at(call))) {
      auto func_node = call->op.as<FunctionNode>();
      CHECK(func_node);
      int param_idx, offset;
      std::tie(param_idx, offset) = func_mutation_map_.at(expr_idx_.at(call));
      new_op = MutateFunction(func_node, param_idx, offset);
    } else {
      new_op = this->Mutate(c->op);
    }

    tvm::Array<Expr> args;
    args.reserve(c->args.size());
    for (const auto& a : c->args) {
      if(a->IsInstance<VarNode>()) {
        auto var = Downcast<Var>(a);
        if(var_idx_.count(var) && var_idx_.at(var) == a_idx_) {
          // point to b
          args.push_back(this->Mutate(let_list_->vars[b_idx_]));
        } else {
          args.push_back(this->Mutate(var));
        }
      } else {
        args.push_back(this->Mutate(a));
      }
    }
    auto new_call = WithFields(GetRef<Call>(c), std::move(new_op), std::move(args));
    auto orig_call_expr = GetRef<Expr>(c);
    if(expr_idx_.count(orig_call_expr) && expr_idx_.at(orig_call_expr) == b_idx_) {
      new_call->checked_type_ = TupleType(fused_args_type_);
    } else {
      new_call->checked_type_ = c->checked_type_;
    }
    // LOG(INFO) << "New call address: " << new_call.get();
    return new_call;
  }

  Expr VisitExpr_(const TupleGetItemNode* tgi_node) override {
    auto tuple_var = Downcast<Var>(tgi_node->tuple);
    if(var_idx_.count(tuple_var)) {
      int tuple_idx = var_idx_.at(tuple_var);
      if(tuple_idx == a_idx_) {
        // point the tuple to b
        Expr new_b_var = this->Mutate(let_list_->vars[b_idx_]);
        auto new_tgi = WithFields(GetRef<TupleGetItem>(tgi_node), std::move(new_b_var));
        new_tgi->checked_type_ = tgi_node->checked_type_;
        return new_tgi;
      } else if (tuple_idx == b_idx_) {
        // offset the index
        Expr new_b_var = this->Mutate(tgi_node->tuple);
        auto new_tgi = WithFields(GetRef<TupleGetItem>(tgi_node), std::move(new_b_var), Integer(tgi_node->index + b_args_offset_));
        new_tgi->checked_type_ = tgi_node->checked_type_;
        return new_tgi;
      }
    }
    auto new_tgi = ExprMutator::VisitExpr_(tgi_node);
    new_tgi->checked_type_ = tgi_node->checked_type_;
    return new_tgi;
  }

  Expr VisitExpr_(const LetNode* let_node) override {
    auto pre_visit = [this](const LetNode* op) {
      this->Mutate(op->value);
      this->Mutate(op->var);
    };
    auto post_visit = [this](const LetNode* op) {
      auto var = Downcast<Var>(this->Mutate(op->var));
      auto value = this->Mutate(op->value);
      auto body = this->Mutate(op->body);
      CHECK(!encountered_vars.count(var)) << "Found duplicate var " << var << " in the Expr. Corresponding values: " << value << " and " << encountered_vars.at(var);
      encountered_vars[var] = value;
      // LOG(INFO) << "New let value ptr:" << value.get();
      auto new_let = WithFields(GetRef<Let>(op), std::move(var), std::move(value), std::move(body));
      this->memo_[GetRef<Expr>(op)] = new_let;
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

  Expr RunFusion() {
    // LOG(INFO) << "================== Before Fusion";
    // LOG(INFO) << ir::AsText(expr_);
    auto mutated_expr = this->Mutate(expr_);
    // std::unique_ptr<ExplicitLetList> new_expr_let_list_ = ExplicitLetList::make(mutated_expr);
    // for(int i=0; i < new_expr_let_list_->exprs.size(); i++) {
    //     LOG(INFO) << "New mutated_expr before dce value ptrs: " << new_expr_let_list_->exprs[i].get();
    // }
    // LOG(INFO) << "================== Before DeadCodeElimination";
    // LOG(INFO) << ir::AsText(mutated_expr);
    // run dead code elimination and simpifly expr before return
    // LOG(INFO) << "==================";
    auto dead_code_eliminated_expr = DeadCodeElimination(mutated_expr);
    // std::unique_ptr<ExplicitLetList> dce_expr_let_list_ = ExplicitLetList::make(dead_code_eliminated_expr);
    // for(int i=0; i < dce_expr_let_list_->exprs.size(); i++) {
    //     LOG(INFO) << "New mutated_expr after dce value ptrs: " << dce_expr_let_list_->exprs[i].get();
    // }
    // LOG(INFO) << "================== After DeadCodeElimination";
    // LOG(INFO) << ir::AsText(dead_code_eliminated_expr);
    auto type_infered_expr = InferType(dead_code_eliminated_expr);
    return dead_code_eliminated_expr;
  }

  ExprMap<Expr> getMutatedExprMap() {
    ExprMap<Expr> result;
    for(int i=0; i<let_list_->vars.size(); i++) {
      auto orig_expr = let_list_->exprs[i];
      auto new_expr = this->Mutate(orig_expr);
      result[orig_expr] = new_expr;
      // LOG(INFO) << "Mapping expr " << orig_expr.get() << " to " << new_expr.get();
    }
    return result;
  }

  Expr getFusedExpr() {
    return this->Mutate(let_list_->exprs[b_idx_]);
  }

protected:
  // modify the let list using the grouping information in uf_
  std::tuple<int, int> getAndValidateInput(Expr op) {
    CHECK(expr_idx_.count(op)) << "Cannot find the ops to fuse in the Expr.";
    const CallNode* call_node = op.as<CallNode>();
    CHECK(call_node) << "One of the ops is not call nodes.";
    int idx = expr_idx_.at(op);

    auto input_var = Downcast<Var>(call_node->args[0]);
    CHECK(var_idx_.count(input_var)) << "Cannot find op's inputs in the Expr.";
    int input_idx = var_idx_.at(input_var);

    auto input_expr = let_list_->exprs[input_idx];
    CHECK(input_expr->IsInstance<TupleNode>()) << "Input of a op to fuse is not tuple.";
    return std::make_tuple(idx, input_idx);
  }

  Expr expr_;

  ExprMap<Expr> encountered_vars;

  std::unique_ptr<ExplicitLetList> let_list_;
  ExprIdxMap expr_idx_;
  VarIdxMap var_idx_;

  int a_idx_, b_idx_;
  int a_input_idx_, b_input_idx_;
  int b_args_offset_ = -1;
  int last_input_tuple_idx_;
  std::string fused_name_;
  Array<Type> fused_args_type_;

  // this map is used to map function call args to parameters of function nodes
  // key: index of the call expr to mutate, value: argument index to mutate
  std::unordered_map<int, std::pair<int, int>> func_mutation_map_;

  bool in_closure_ = false;
};

}  // namespace tuple_fusion

}  // namespace pass
}  // namespace raf
