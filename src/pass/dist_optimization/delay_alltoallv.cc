/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file delay_alltoallv.cc
 * \brief Delay alltoallv ops to the latest possible location (without
 *        changing communication and computation order) to allow more
 *        overlap
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
#include "./scheduler_utils.h"
#include "../common.h"
#include "../../common/shape_utils.h"
#include "../let_list.h"

namespace raf {
namespace pass {
namespace delay_alltoallv {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using raf::distributed::DistContext;
using namespace raf::analysis;
using scheduler_utils::Exprs;
using stream_pool::StreamTagEnum;
using common::shape_utils::BytesCompactTensor;

using OpSet = std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

struct pair_hash {
  std::size_t operator()(const std::pair<int, int>& v) const {
    return std::hash<std::string>{}(std::to_string(v.first) + "," + std::to_string(v.second));
  }
};

struct pair_equal {
  bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};

using DepSet = std::unordered_set<std::pair<int, int>, pair_hash, pair_equal>;

class AllToAllvDependencyAnalyzer : public ExprVisitor {
 public:
  virtual void VisitExpr_(const VarNode* var) {
    var_idx_map_[GetRef<Var>(var)] = current_idx_;
  }

  virtual void VisitExpr_(const CallNode* call) {
    auto call_expr = GetRef<Expr>(call);
    UpdateDependencyInfo_(call_expr, call->args);
  }

  virtual void VisitExpr_(const TupleNode* tuple) {
    auto tuple_expr = GetRef<Expr>(tuple);
    UpdateDependencyInfo_(tuple_expr, tuple->fields);
  }

  virtual void VisitExpr_(const TupleGetItemNode* tuple_get_item) {
    auto tuple_get_item_expr = GetRef<Expr>(tuple_get_item);
    UpdateDependencyInfo_(tuple_get_item_expr, {tuple_get_item->tuple});
  }

  virtual void VisitExpr_(const LetNode* op) {
    auto pre_visit = [this](const LetNode* op) {
      Var var = op->var;
      Expr value = op->value;
      expr_idx_map_[value] = current_idx_;
      var_idx_map_[var] = current_idx_;
      idx_expr_map_[current_idx_] = value;
      idx_var_map_[current_idx_] = var;
      this->VisitExpr(var);
      this->VisitExpr(value);
      current_idx_++;
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = op->body;
      this->VisitExpr(body);
      current_idx_--;
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  /*! \brief Analyse the predecessors and successors for fused collectives.
   *
   * \returns true if there is any fused collectives in the expr.
   */
  bool Analyse(const Expr& func_body) {
    VisitExpr(func_body);
    if(alltoallv_indices.empty()) {
      return false;
    }
    // compute the latest location of each alltoallv
    for (const auto& a2av_idx : alltoallv_indices) {
      int first_consumer_idx = INT32_MAX;
      for (const auto& tgi_idx : alltoallv_to_tgis_[a2av_idx]) {
        if (schedule_before_idx_.count(tgi_idx)) {
          first_consumer_idx = std::min(first_consumer_idx, schedule_before_idx_.at(tgi_idx));
        }
      }
      CHECK_NE(first_consumer_idx, INT32_MAX) << "first consumer of alltoallv not found";
      schedule_before_idx_[a2av_idx] = first_consumer_idx;
    }
    // check that we do not change the order of alltoallvs
    for (int i=0; i<alltoallv_indices.size() - 1; i++) {
      if (schedule_before_idx_[alltoallv_indices[i]] > schedule_before_idx_[alltoallv_indices[i+1]]) {
        schedule_before_idx_[alltoallv_indices[i]] = schedule_before_idx_[alltoallv_indices[i+1]];
      }
    }
    // for each expr, find the set of ops to be scheduled before it
    // sort schedule_before_idx_'s keys
    std::vector<int> sorted_indices_to_be_scheduled;
    for (const auto& kv : schedule_before_idx_) {
      sorted_indices_to_be_scheduled.push_back(kv.first);
    }
    std::sort(sorted_indices_to_be_scheduled.begin(), sorted_indices_to_be_scheduled.end());
    for (auto idx: sorted_indices_to_be_scheduled) {
      auto expr = idx_expr_map_.at(idx);
      auto before_idx = schedule_before_idx_.at(idx);
      auto before_expr = idx_expr_map_.at(before_idx);
      expr_to_ops_scheduled_before_[before_expr].push_back(expr);
    }
    return true;
  }

  Exprs GetOpsScheduledBefore(const Expr& expr) {
    if (expr_to_ops_scheduled_before_.count(expr)) {
      return expr_to_ops_scheduled_before_.at(expr);
    } else {
      return {};
    }
  }

  bool NeedsToBeDelayed(const Expr& expr) {
    int idx = GetIndexFromExpr(expr);
    return schedule_before_idx_.count(idx);
  }

  int GetIndexFromVar(const Expr& expr) {
    if(var_idx_map_.count(expr)) {
      return var_idx_map_.at(expr);
    } else {
      return -1;
    }
  }

  int GetIndexFromExpr(const Expr& expr) {
    if(expr_idx_map_.count(expr)) {
      return expr_idx_map_.at(expr);
    } else {
      return -1;
    }
  }

  Expr GetExprFromIndex(int idx) {
    return idx_expr_map_.at(idx);
  }

  Var GetVarFromIndex(int idx) {
    return idx_var_map_.at(idx);
  }

 private:
  void UpdateDependencyInfo_(const Expr& expr, const Array<Expr>& input_args) {
    if (expr->IsInstance<CallNode>() && scheduler_utils::IsOp(expr, Op::Get("raf.op._all_to_allv"))) {
      alltoallv_idx_.insert(current_idx_);
      alltoallv_indices.push_back(current_idx_);
    } else {
      for (const auto& arg : input_args) {
        int arg_idx = GetIndexFromVar(arg);
        if (alltoallv_idx_.count(arg_idx)) {
          // tgi of alltoallv
          alltoallv_to_tgis_[arg_idx].push_back(current_idx_);
          tgis_idx_.insert(current_idx_);
        } else if (tgis_idx_.count(arg_idx)) {
          // update first consumer of tgi
          auto expr = idx_expr_map_.at(arg_idx);
          if (!schedule_before_idx_.count(arg_idx)) {
            schedule_before_idx_[arg_idx] = current_idx_;
          }
        }
      }
    }
  }

  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
  // maps each var to its index
  ExprMap<int> var_idx_map_;
  // maps each expr to its index
  ExprMap<int> expr_idx_map_;
  // index to expr map
  std::unordered_map<int, Expr> idx_expr_map_;
  std::unordered_map<int, Var> idx_var_map_;

  std::vector<int> alltoallv_indices;
  std::unordered_set<int> alltoallv_idx_;
  std::unordered_set<int> tgis_idx_;

  std::unordered_map<int, int> schedule_before_idx_;
  ExprMap<Exprs> expr_to_ops_scheduled_before_;
  std::unordered_map<int, std::vector<int>> alltoallv_to_tgis_;
};


class AllToAllvDelayer : public ExprVisitor {
public:
  explicit AllToAllvDelayer(const FunctionNode* func) : func_(func) {
  }
  void VisitExpr_(const LetNode* op) {
    Var orig_var = op->var;
    Expr orig_value = op->value;
    int orig_value_idx = analyzer_.GetIndexFromExpr(orig_value);
    CHECK_NE(orig_value_idx, -1);
    if (!analyzer_.NeedsToBeDelayed(orig_value)) {
      auto exprs_to_add = analyzer_.GetOpsScheduledBefore(orig_value);
      for (const auto& expr_to_add : exprs_to_add) {
        auto var_to_add = analyzer_.GetVarFromIndex(analyzer_.GetIndexFromExpr(expr_to_add));
        ell_->Push(var_to_add, expr_to_add);
      }
      ell_->Push(orig_var, orig_value);
    }
    ell_->ret = ell_->vars.back();
    VisitExpr(op->body);
  }

  Function Run() {
    if (!analyzer_.Analyse(func_->body)) {
      // no alltoallvs found in expr. do nothing.
      return GetRef<Function>(func_);
    }

    ell_ = std::make_unique<ExplicitLetList>();
    VisitExpr(func_->body);

    return Function(func_->params, ell_->AsExpr(), {}, {});
  }
protected:
  const FunctionNode* func_;
  AllToAllvDependencyAnalyzer analyzer_;
  std::unique_ptr<ExplicitLetList> ell_;
};

}  // namespace delay_alltoallv

Function DelayAllToAllvInFunction(Function f) {
  return delay_alltoallv::AllToAllvDelayer(f.operator->()).Run();
}

Pass DelayAllToAllv() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return delay_alltoallv::AllToAllvDelayer(f.operator->()).Run();
  };
  return CreateRAFFunctionPass(pass_func, 0, "DelayAllToAllv", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.DelayAllToAllv").set_body_typed(DelayAllToAllv);

}  // namespace pass
}  // namespace raf
