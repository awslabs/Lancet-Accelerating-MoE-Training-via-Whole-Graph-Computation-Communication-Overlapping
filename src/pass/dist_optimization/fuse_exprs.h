/*!
 * Copyright (c) 2022 by Contributors
 * \file fuse_exprs.h
 * \brief Fuse given exprs.
 */
#pragma once
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <tvm/relay/expr.h>
#include "raf/ir.h"
#include "raf/value.h"
#include "raf/ir_ext.h"
#include "raf/op_utils.h"
#include "raf/analysis.h"
#include "../common.h"
#include "./scheduler_common.h"
#include "./extended_op_profiler.h"
#include "./extended_dfg.h"

namespace raf {
namespace pass {
namespace fuse_exprs {

using namespace raf::ir;
using namespace raf::analysis;
using namespace raf::op;
using namespace raf::value;
using namespace raf::pass;
using namespace raf::pass::extended_dfg;
using namespace raf::pass::extended_op_profiler;
using namespace raf::pass::scheduler_common;

using ExprIdxMap = ExprMap<int>;
using VarIdxMap = VarMap<int>;

class ExprFusor {
 public:
  ExprFusor(ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler);

  void Fuse(Expr expr_a, Expr expr_b);

  Expr GetInputExpr(Expr expr, int n) const;
  Var GetInputVar(Expr expr, int n) const;

  Expr GetExpr(Expr var) const;
  Var GetVar(Expr expr) const;

  Expr GetPrevExpr(Expr expr) const;

  Expr GetLaterExpr(Expr expr_1, Expr expr_2) const;
  Expr GetFormerExpr(Expr expr_1, Expr expr_2) const;

  void AddCompExpr(Var var, Expr expr, Expr expr_to_append);
  void AddCommExpr(Var var, Expr expr, Expr expr_to_append, Expr component_a, Expr component_b);

  void UpdateCompExpr(Expr expr_to_update, Expr new_expr);

  void RemoveExpr(Expr expr);
  void RemoveExprs(Exprs exprs);
  
  void SetParent(Expr expr, Expr parent);
  void SetParents(Expr expr, Exprs parents);

  void SetFusedComm(Var fused_comm_var, Var comm_var_a, Var comm_var_b);
 
  void SetExprToPropagate(Expr expr);
  void SetExprOfSource(Expr expr);

  NodeMap<bool> GetNeedPropagate() const;
  const Node* GetSource() const;

 private:
  void Finalize_();

  void UpdateIdxMap_();

  void IdentifyCommConsumers_(Var comm_var, Exprs& tgi_consumers, ExprSet& other_consumers);
  Expr GetFirstExpr_(const Exprs& expr);

  ScheduledDFG& sched_dfg_;
  ExtendedOpProfiler& op_profiler_;

  NodeMap<bool> need_propagate_;
  const Node* source_;

  std::unique_ptr<ExplicitLetList> let_list_;
  ExprIdxMap expr_idx_;
  ExprIdxMap var_idx_;

  // expr to remove
  ExprSet removed_exprs_;
  // exprs to update, including exprs to fuse and tgis of exprs to fuse.
  ExprMap<Expr> updated_expr_;

  // added expr to expr to append map.
  ExprMap<Exprs> expr_appended_exprs_;
  // added expr to var.
  ExprMap<Var> added_expr_var_;
  // added and updated expr to parents map.
  ExprMap<Exprs> expr_parents_;
};

// FFuse accepts multiple let exprs to create a fused expr.
using FFuse = std::function<void(ExprFusor&, Expr, Expr)>;
using FFuseMap = std::unordered_map<std::string, FFuse>;

std::tuple<const Node*, NodeMap<bool>>
FuseNodes(ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler,
          const Node* node_a, const Node* node_b);

} // namespace fuse_exprs
} // namespace pass
} // namespace raf