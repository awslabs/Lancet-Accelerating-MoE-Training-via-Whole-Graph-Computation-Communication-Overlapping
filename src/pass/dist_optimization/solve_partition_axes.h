/*!
 * \file solve_partition_axes.h
 * \brief Solve finding partition axes using constraint optimization.
 */
#pragma once
#include <vector>
#include <string>
#include "raf/plugins/ortools/constraint_opt.h"
#include "./partition_common.h"

namespace raf {
namespace pass {
namespace solve_partition_axes {

using namespace partition_common;
using Types = std::vector<Type>;

using plugins::constraint_opt::ConstraintOptimizer;
using plugins::constraint_opt::ConstraintType;

// a class to represent a constraint programming model.
// A cell is any constraint, variable, constant or target.
// CellRef is the index to a cell in the model.
class CPModel {
 public:
  explicit CPModel(const ScheduledDFG& sched_dfg, int dp_group_size, int lower_bound = SpecialAxis::kNone);

  // For constructing PartitionEnv
  ExprMap<InOutCellRefs> GetInOutCellRefs() const;
  ExprMap<Exprs> GetFuncExprs() const;
  ExprMap<ExprMap<ExprMap<Index>>> GetConsumerIndex() const;

  void SetThreshold(int threshold);

  Exprs GetFuncExprs(Expr expr) const;
  void SetFuncExprs(Expr expr, Exprs func_exprs);
  ExprMap<ExprMap<Index>> GetConsumerIndex(Expr expr) const;
  void SetConsumerIndex(Expr expr, ExprMap<ExprMap<Index>> arg_consumer_index);
  CellRefs GetInVariableRefs(Expr expr, Index arg_index) const;
  std::vector<CellRefs> GetInVariableCellRefsVec(Expr expr) const;
  InOutCellRefs GetInOutVariableCellRefs(Expr expr) const;
  void SetInOutVariableCellRefs(Expr expr, InOutCellRefs inout_cellrefs);
  CellRefs GetOutVariableCellRefs(Expr expr) const;

  void SetCheckedArgIndices(Expr expr, Indices arg_indices);
  void SetCheckedArgIndices(Expr expr, int n_arg);
  bool ArgIsChecked(Expr expr, Index arg_index) const;
  IndexMap<Index> GetCheckedArgIndiceMap(Expr expr) const;
  ExprMap<IndexMap<Index>> GetCheckedArgIndiceMap() const;

  CellRefs AddInVariable(Expr expr, Expr in);
  CellRefs AddOutVariable(Expr expr);
  CellRefs GetOutVariableOrAddAsExternal(Expr expr);
  CellRef AddEqual(CellRef cell_1, CellRef cell_2, bool is_top);
  CellRef AddNotEqual(CellRef cell_1, CellRef cell_2, bool is_top);
  CellRef AddAnd(CellRef cell_1, CellRef cell_2, bool is_top);
  CellRef AddOr(CellRef cell_1, CellRef cell_2, bool is_top);

  CellRef AddTarget(CellRef ref, bool is_top);
  void AddMaximizeTarget(InOutCellRefs& inout_cellrefs);

  CellRef GetConstantCellRef(int value);

  CellRefs AddMatching(CellRefs cells_1, CellRefs cells_2, Types types_1, Types types_2, bool is_top);
  CellRef AddMatching(CellRef cell_1, CellRef cell_2, const IndexMap<Index>& matching_axes, bool is_top);
  CellRef AddMatching(CellRef cell_1, CellRef cell_2, Type type_1, Type type_2, bool is_top);
  void AddDegenerateAxes(Type tt, CellRefs cells);

  CellRefMap<Index> GetSolution();

 private:
  bool IsValidCellRef_(CellRef ref);
  CellRef AddConstant_(int val);
  CellRef AddVariable_(int upper_bound);
  CellRefs AddVariable_(Type type);
  CellRefs AddTargets_(CellRefs& refs, bool is_top);
  Expr GetAndCheckExpr_(Expr expr) const;

  const ScheduledDFG& sched_dfg_;
  // expr -> input and output CellRefs
  ExprMap<InOutCellRefs> expr_inout_cellrefs_;
  // Since not all arguments require a variable, we need a map to map for
  // expr -> input arg index -> consolidated index in InOutCellRefs's input array
  ExprMap<IndexMap<Index>> expr_arg_idx_to_inoutcellref_index;
  // expr -> input func args -> consumer in func exprs -> input index
  ExprMap<ExprMap<ExprMap<Index>>> expr_arg_consumer_index_;
  // expr -> expanded func exprs
  ExprMap<Exprs> expr_func_exprs_;
  // func var -> expr
  ExprMap<Expr> func_var_expr_;
  // func expr -> var
  ExprMap<Var> func_expr_var_;

  // constant -> cell reference
  ConstantMap<CellRef> const_val_cell_reference_;
  // threshold of partition dimension
  // used to forbid partitioning axes with a dimension less than the threshold.
  int threshold_;

  ConstraintOptimizer cp_optimizer_;
};

// FConstraint adds constraints on partition axes of input arguments and output tensor / tuple.
using FConstraint = std::function<void(CPModel&, Expr)>;
using FConstraintMap = std::unordered_map<std::string, FConstraint>;

bool HasConstraint(const std::string& op);

class CPModelBuilder : public ExprVisitor {
 public:
  explicit CPModelBuilder(const ScheduledDFG& sched_dfg, int dp_group_size);

  CPModel Build(Exprs& exprs);
  bool ModelValid() const;

  void VisitExpr_(const LetNode* let) override;
  void VisitExpr_(const TupleNode* tuple) override;
  void VisitExpr_(const TupleGetItemNode* tgi) override;
  void VisitExpr_(const CallNode* call) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const FunctionNode* func) override;
  void VisitExpr(const Expr& expr);

 private:
  void Setup_(Exprs& exprs);
  void GetReturnValue_();
  void AddTarget_();
  void PrintStack_();

  // Current call stack.
  Exprs stack_;
  CPModel model_;
  const ScheduledDFG& sched_dfg_;
  int dp_group_size_;
  bool model_valid_;
};

struct CPSolution {
 public:
  CPSolution();

  CPSolution(const CPModel& model,
             const ExprMap<InOutAxes>& expr_inout_axes,
             const Exprs& exprs);

  // expr -> calculated input and output partition axes
  ExprMap<InOutCellRefs> expr_inout_axes;
  // expr -> expr in the function called
  ExprMap<Exprs> expr_func_exprs;
  // expr -> input arg index -> input index in expr_inout_axes
  ExprMap<IndexMap<Index>> expr_arg_idx_to_inoutaxes_index;
  // original expr -> partition axes -> partitioned vars and exprs
  ExprMap<ExprMap<ExprMap<Index>>> expr_arg_consumer_index;
  // all exprs to solve partition axes
  Exprs exprs;
};

CPSolution SolvePartitionAxes(const ScheduledDFG& sched_dfg, Exprs& exprs, int dp_group_size);
CPSolution SolvePartitionAxes(const ScheduledDFG& sched_dfg, const Nodes& nodes, int dp_group_size);
CPSolution SolvePartitionAxes(const ScheduledDFG& sched_dfg, const NodeSet& nodes, int dp_group_size);

bool IsAllPartitionable(const CPSolution& solution, const Exprs& exprs);
bool IsAllPartitionable(const CPSolution& solution);

void CheckAllPartitionable(const CPSolution& solution, const Exprs& exprs);
void CheckAllPartitionable(const CPSolution& solution);

std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const Nodes& nodes);
std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const NodeSet& nodes);

std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const CPSolution& solution, const Nodes& nodes, const Node* node, int dp_group_size);
std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const CPSolution& solution, const NodeSet& nodes, const Node* node, int dp_group_size);

std::vector<int> GetPartitionParts(const ScheduledDFG& sched_dfg, const Nodes& nodes, const CPSolution& solution, int max_partition, int dp_group_size);
std::vector<int> GetPartitionParts(const ScheduledDFG& sched_dfg, const NodeSet& nodes, const CPSolution& solution, int max_partition, int dp_group_size);

} // namespace solve_partition_axes
} // namespace pass
} // namespace raf