/*!
 * Copyright (c) 2022 by Contributors
 * \file solve_partition_axes.h
 * \brief Solve finding partition axes using constraint optimization.
 */
#include "solve_partition_axes.h"
#include "partition_exprs.h"

namespace raf {
namespace pass {
namespace solve_partition_axes {

// using plugins::constraint_opt::SolveConstraintOpt;
using partition_exprs::HasFPartition;

// copy of NormalizeAxis in declare_utils.h
inline int NormalizeAxis(int axis, int ndim) {
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return axis < 0 ? axis + ndim : axis;
}

std::string bool2str(bool b) {
  return b ? "true" : "false";
}

Types ExpandTupleType(TupleType tt) {
  Types result;
  for(auto field_type: tt->fields) {
    result.push_back(field_type);
  }
  return result;
}

// class CPModel

CPModel::CPModel(const ScheduledDFG& sched_dfg, int dp_group_size, int lower_bound) :
  sched_dfg_(sched_dfg), threshold_(2), cp_optimizer_(lower_bound) {}

ExprMap<InOutCellRefs> CPModel::GetInOutCellRefs() const {
  return expr_inout_cellrefs_;
}

ExprMap<Exprs> CPModel::GetFuncExprs() const {
  return expr_func_exprs_;
}

ExprMap<ExprMap<ExprMap<Index>>> CPModel::GetConsumerIndex() const {
  return expr_arg_consumer_index_;
}

void CPModel::SetThreshold(int threshold) {
  threshold_ = threshold;
}

Exprs CPModel::GetFuncExprs(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_func_exprs_.count(expr));
  return expr_func_exprs_.at(expr);
}

void CPModel::SetFuncExprs(Expr expr, Exprs func_exprs) {
  expr = GetAndCheckExpr_(expr);
  expr_func_exprs_[expr] = func_exprs;
  for (auto func_expr : func_exprs) {
    auto var = Downcast<Var>(TryGetLetVar(func_expr));
    auto expr = TryGetLetValue(func_expr);
    func_var_expr_[var] = expr;
    func_expr_var_[expr] = var;
  }
}

ExprMap<ExprMap<Index>> CPModel::GetConsumerIndex(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_arg_consumer_index_.count(expr));
  return expr_arg_consumer_index_.at(expr);
}

void CPModel::SetConsumerIndex(Expr expr, ExprMap<ExprMap<Index>> arg_consumer_index) {
  expr = GetAndCheckExpr_(expr);
  expr_arg_consumer_index_[expr] = arg_consumer_index;
}

CellRefs CPModel::GetInVariableRefs(Expr expr, Index arg_index) const {
  expr = GetAndCheckExpr_(expr);
  // check that we have created variables for this expr
  CHECK(expr_inout_cellrefs_.count(expr));
  auto& in_refs = expr_inout_cellrefs_.at(expr).first;
  // convert arg_index to index of the cell in InOutCellRefs
  CHECK(expr_arg_idx_to_inoutcellref_index.count(expr));
  CHECK(expr_arg_idx_to_inoutcellref_index.at(expr).count(arg_index));
  Index cellref_index = expr_arg_idx_to_inoutcellref_index.at(expr).at(arg_index);
  CHECK(in_refs.size() > cellref_index);
  return in_refs.at(cellref_index);
}

std::vector<Indices> CPModel::GetInVariableCellRefsVec(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_inout_cellrefs_.count(expr));
  return expr_inout_cellrefs_.at(expr).first;
}

InOutCellRefs CPModel::GetInOutVariableCellRefs(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_inout_cellrefs_.count(expr));
  return expr_inout_cellrefs_.at(expr);
}

void CPModel::SetInOutVariableCellRefs(Expr expr, InOutCellRefs inout_cellrefs) {
  expr = GetAndCheckExpr_(expr);
  expr_inout_cellrefs_[expr] = inout_cellrefs;
}

CellRefs CPModel::GetOutVariableCellRefs(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_inout_cellrefs_.count(expr)) << "Failed to find expr " << expr << " in expr_inout_cellrefs_";
  return expr_inout_cellrefs_.at(expr).second;
}

void CPModel::SetCheckedArgIndices(Expr expr, Indices arg_indices) {
  expr = GetAndCheckExpr_(expr);
  // Maps index of arg to index of the cell in InOutCellRefs
  for (int i = 0; i < arg_indices.size(); ++i) {
    expr_arg_idx_to_inoutcellref_index[expr][arg_indices[i]] = i;
  }
}

// set the first n_arg to be checked
void CPModel::SetCheckedArgIndices(Expr expr, int n_arg) {
  expr = GetAndCheckExpr_(expr);
  for (int i = 0; i < n_arg; ++i) {
    expr_arg_idx_to_inoutcellref_index[expr][i] = i;
  }
}

bool CPModel::ArgIsChecked(Expr expr, Index index) const {
  expr = GetAndCheckExpr_(expr);
  return expr_arg_idx_to_inoutcellref_index.count(expr) && expr_arg_idx_to_inoutcellref_index.at(expr).count(index);
}

IndexMap<Index> CPModel::GetCheckedArgIndiceMap(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_arg_idx_to_inoutcellref_index.count(expr));
  return expr_arg_idx_to_inoutcellref_index.at(expr);
}

ExprMap<IndexMap<Index>> CPModel::GetCheckedArgIndiceMap() const {
  return expr_arg_idx_to_inoutcellref_index;
}

// create a new input variable (with "in"'s shape) for "expr" and return its cell references.
CellRefs CPModel::AddInVariable(Expr expr, Expr in) {
  expr = GetAndCheckExpr_(expr);
  expr_inout_cellrefs_[expr].first.push_back(AddVariable_(TryGetLetVar(in)->checked_type_));
  return expr_inout_cellrefs_[expr].first.back();
}

// create a new output variable for "expr" and return its index.
CellRefs CPModel::AddOutVariable(Expr expr) {
  expr = GetAndCheckExpr_(expr);
  expr_inout_cellrefs_[expr].second = AddVariable_(TryGetLetVar(expr)->checked_type_);
  return expr_inout_cellrefs_[expr].second;
}

// return the output variable index if it exists, otherwise this expr must
// be an argument of the global func. In this case we create the output variable 
// for "expr" and return its index.
Indices CPModel::GetOutVariableOrAddAsExternal(Expr expr) {
  expr = GetAndCheckExpr_(expr);
  if (!expr_inout_cellrefs_.count(expr)) {
    expr_inout_cellrefs_[expr] = std::make_pair<std::vector<Indices>, Indices>({}, {});
    AddOutVariable(expr);
  }
  return GetOutVariableCellRefs(expr);
}

CellRef CPModel::AddEqual(CellRef cell_1, CellRef cell_2, bool is_top) {
  if (!IsValidCellRef_(cell_1) || !IsValidCellRef_(cell_2)) {
    // When there're invalid cell in cell_1 and cell_2,
    // the equal constraint should always be true.
    return GetConstantCellRef(1);
  }
  return cp_optimizer_.AddConstraint(ConstraintType::kEq, cell_1, cell_2, is_top);
}

CellRef CPModel::AddNotEqual(CellRef cell_1, CellRef cell_2, bool is_top) {
  if (!IsValidCellRef_(cell_1) || !IsValidCellRef_(cell_2)) {
    // When there're invalid index in cell_1 and cell_2,
    // the equal constraint should always be false.
    return GetConstantCellRef(0);
  }
  return cp_optimizer_.AddConstraint(ConstraintType::kNeq, cell_1, cell_2, is_top);
}

CellRef CPModel::AddAnd(CellRef cell_1, CellRef cell_2, bool is_top) {
  if (!IsValidCellRef_(cell_1) || !IsValidCellRef_(cell_2)) {
    if (IsValidCellRef_(cell_1) || IsValidCellRef_(cell_2)) {
      // When only one cell is valid in cell_1 and cell_2,
      // the and constraint should equal the valid variable.
      return cell_1 != -1 ? cell_1 : cell_2;
    } else {
      // When both of cell_1 and cell_2 are invalid,
      // the and constraint should always be true.
      return GetConstantCellRef(1);
    }
  }
  return cp_optimizer_.AddConstraint(ConstraintType::kAnd, cell_1, cell_2, is_top);
}

CellRef CPModel::AddOr(CellRef cell_1, CellRef cell_2, bool is_top) {
  if (!IsValidCellRef_(cell_1) || !IsValidCellRef_(cell_2)) {
    if (IsValidCellRef_(cell_1) || IsValidCellRef_(cell_2)) {
      // When only one cell is valid in cell_1 and cell_2,
      // the and constraint should equal the valid variable.
      return cell_1 != -1 ? cell_1 : cell_2;
    } else {
      // When both of cell_1 and cell_2 are invalid,
      // the or constraint should always be false.
      return GetConstantCellRef(0);
    }
  }
  return cp_optimizer_.AddConstraint(ConstraintType::kOr, cell_1, cell_2, is_top);
}

CellRef CPModel::AddTarget(CellRef ref, bool is_top) {
  if (!IsValidCellRef_(ref)) {
    // Discard invalid index when adding target.
    return -1;
  }
  return cp_optimizer_.AddTarget(ref, is_top);
}

void CPModel::AddMaximizeTarget(InOutCellRefs& inout_cellrefs) {
  CellRefs refs = {};
  for (auto& in_refs : inout_cellrefs.first) {
    refs.insert(refs.end(), in_refs.begin(), in_refs.end());
  }
  auto& out_refs = inout_cellrefs.second;
  refs.insert(refs.end(), out_refs.begin(), out_refs.end());
  AddTargets_(refs, true);
}

CellRef CPModel::GetConstantCellRef(int value) {
  if (!const_val_cell_reference_.count(value)) {
    CellRef ref;
    ref = AddConstant_(value);
    const_val_cell_reference_[value] = ref;
  }
  return const_val_cell_reference_[value];
}

// matching constraint first finds the matching axes between two exprs
// (i.e. axes of the same shape), and asserts that the corresponding axes
// should have the same partition status. This is a convenient way for handling
// ops involving reshape or transpose.
// For example, if we have two exprs with shape [2, 3, 4] and [4, 3, 2],
// the matching axes are 0 -> 2, 1 -> 1, 2 -> 0, and the matching constraint
// will assert that the axis to partition must be one of the matching ones.

CellRefs CPModel::AddMatching(CellRefs cells_1, CellRefs cells_2, Types types_1, Types types_2, bool is_top) {
  CHECK_EQ(cells_1.size(), cells_2.size());
  CHECK_EQ(types_1.size(), types_2.size());
  CHECK_EQ(cells_1.size(), types_1.size());
  CellRefs refs;
  for(int i=0; i< cells_1.size(); i++) {
    refs.push_back(AddMatching(cells_1[i], cells_2[i], types_1[i], types_2[i], is_top));
  }
  return refs;
}

CellRef CPModel::AddMatching(CellRef cell_1, CellRef cell_2, Type type_1, Type type_2, bool is_top) {
  auto matching_axes = FindMatchingAxes(type_1, type_2);
  return AddMatching(cell_1, cell_2, matching_axes, is_top);
}

CellRef CPModel::AddMatching(CellRef cell_1, CellRef cell_2, const IndexMap<Index>& matching_axes, bool is_top) {
  CellRef special_partition_axis_var = GetConstantCellRef(SpecialAxis::kExpert);
  CellRef not_partition_index = GetConstantCellRef(SpecialAxis::kNone);
  CellRef ref = -1;
  if (matching_axes.size() != 0) {
    IndexMap<Index> matching_axes_ = {matching_axes.begin(), matching_axes.end()};
    matching_axes_[-1] = SpecialAxis::kNone;
    CellRefs rules = {};
    // the axes to partition must be one of the matching axes
    for (auto it : matching_axes_) {
      Index arg_constant_index = GetConstantCellRef(it.first);
      Index expr_constant_index = GetConstantCellRef(it.second);
      CellRef arg_rule = AddEqual(cell_1, arg_constant_index, false);
      CellRef expr_rule = AddEqual(cell_2, expr_constant_index, false);
      CellRef and_rule = AddAnd(arg_rule, expr_rule, false);
      rules.push_back(and_rule);
    }
    ref = rules[0];
    for (int i = 1; i < rules.size(); ++i) {
      ref = AddOr(ref, rules[i], false);
    }
  }
  // or both axis to partition is the special axis
  CellRef cell1_spx = AddEqual(cell_1, special_partition_axis_var, false);
  CellRef cell2_spx = AddEqual(cell_2, special_partition_axis_var, false);
  CellRef spx_rule = AddAnd(cell1_spx, cell2_spx, false);
  // or all not partition
  CellRef cell1_not_partition = AddEqual(cell_1, not_partition_index, false);
  CellRef cell2_not_partition = AddEqual(cell_2, not_partition_index, false);
  CellRef not_partition_rule = AddAnd(cell1_not_partition, cell2_not_partition, false);
  // "or" the above cases together
  ref = AddOr(ref, spx_rule, false);
  ref = AddOr(ref, not_partition_rule, is_top);
  return ref;
}

// forbid partitioning on degenerate axes (i.e. axes with size 1)
void CPModel::AddDegenerateAxes(Type tt, CellRefs cells) {
  if (cells.size() == 1) {
    Indices axes = {};
    if (tt->IsInstance<TensorTypeNode>()) {
      axes = FindDegenerateAxes(tt, threshold_);
    } else if (tt->IsInstance<TupleTypeNode>()) {
      axes = FindDegenerateAxes(Downcast<TupleType>(tt)->fields[0], threshold_);
    }
    for (int i = 0; i < axes.size(); ++i) {
      Index axis = axes[i];
      Index axis_const_idx = GetConstantCellRef(axis);
      AddNotEqual(cells[0], axis_const_idx, true);
    }
  } else if (cells.size() > 1) {
    CHECK(tt->IsInstance<TupleTypeNode>());
    CHECK(Downcast<TupleType>(tt)->fields.size() == cells.size());
    for (int i = 0; i < cells.size(); ++i) {
      AddDegenerateAxes(Downcast<TupleType>(tt)->fields[i], {cells[i]});
    }
  } else {
    LOG(WARNING) << "Invalid cells size: " << cells.size();
  }
}

bool CPModel::IsValidCellRef_(CellRef ref) {
  return ref != -1;
}

CellRef CPModel::AddConstant_(int val) {
  return cp_optimizer_.AddConstant(val);
}

CellRef CPModel::AddVariable_(int upper_bound) {
  return cp_optimizer_.AddVariable(upper_bound);
}

CellRefs CPModel::AddVariable_(Type type) {
  CellRefs refs = {};
  if (type->IsInstance<TensorTypeNode>()) {
    // For scalar value, ndim is 0, it would never be partitioned.
    auto ndim = Downcast<TensorType>(type)->shape.size();
    if (ndim == 0) {
      return {-1};
    }
    refs.push_back(AddVariable_(ndim - 1));
  } else if (type->IsInstance<TupleTypeNode>()) {
    auto fields = Downcast<TupleType>(type)->fields;
    for (int i = 0; i < fields.size(); ++i) {
      CellRefs refs_;
      refs_ = AddVariable_(fields[i]);
      refs.insert(refs.end(), refs_.begin(), refs_.end());
    }
  }
  return refs;
}

CellRefs CPModel::AddTargets_(CellRefs& refs, bool is_top) {
  CellRefs target_refs = {};
  for (int i = 0; i < refs.size(); ++i) {
    target_refs.push_back(AddTarget(refs[i], is_top));
  }
  return target_refs;
}

// Input expr can be let, var and value.
// This function extracts value expr from let, or find the corresponding value
// if the input expr is a var.
Expr CPModel::GetAndCheckExpr_(Expr expr) const {
  expr = TryGetLetVar(expr);
  if (expr->IsInstance<VarNode>()) {
    auto var = Downcast<Var>(expr);
    if (sched_dfg_.var_expr_map.count(var)) {
      expr = sched_dfg_.var_expr_map.at(var);
    } else if (func_var_expr_.count(var)) {
      expr = func_var_expr_.at(var);
    }
  }
  return expr;
}

CellRefMap<CellRef> CPModel::GetSolution() {
  return cp_optimizer_.GetSolution();
}

// FConstraints

FConstraint TupleConstraint = [](CPModel& model, Expr expr) {
  Tuple tuple = GetTuple(expr);
  model.SetCheckedArgIndices(expr, tuple->fields.size());
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == tuple->fields.size());
  for (int i = 0; i < tuple->fields.size(); ++i) {
    Expr field = tuple->fields[i];
    CellRefs field_variables = model.GetOutVariableOrAddAsExternal(field);
    CellRefs in_variables = model.AddInVariable(expr, field);
    CHECK(in_variables.size() == field_variables.size() && in_variables.size() == 1);
    // the partition axis of each out variable should match the corresponding in variable.
    model.AddEqual(in_variables[0], out_variables[i], true);
    // connect the argument's output (field variable) with this expr's input variable.
    model.AddEqual(field_variables[0], in_variables[0], true);
  }
  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint TGIConstraint = [](CPModel& model, Expr expr) {
  TupleGetItem tgi = GetTGI(expr);
  CellRefs out_variables = model.AddOutVariable(expr);

  model.SetCheckedArgIndices(expr, 1);
  Expr tuple = tgi->tuple;
  int index = tgi->index;
  CellRefs tuple_variables = model.GetOutVariableOrAddAsExternal(tuple);
  CHECK(tuple_variables.size() > index);
  CellRefs in_variables = model.AddInVariable(expr, tuple);
  CHECK(in_variables.size() == tuple_variables.size());
  model.AddEqual(in_variables[index], out_variables[0], true);
  for (int i=0; i< tuple_variables.size(); ++i) {
    model.AddEqual(tuple_variables[i], in_variables[i], true);
  }

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

// This constraint should not be used.
FConstraint CallConstraint = [](CPModel& model, Expr expr) {
  LOG(FATAL) << "CallConstraint should not be used.";
};

// Matching Constraint requires all inputs and outputs with matching dimensions (see AddMatching in CPModel) to have
// the same partition axis
FConstraint CreateMatchingConstraint(std::string dialect_name, ArgIndicesGetter getter) {
  // All args and return value are tensors or tuples.
  // If input or output contains tuples, all input / output should be tuple with the same field size
  FConstraint matching_constraint = [=](CPModel& model, Expr expr) {
    Call call = GetOpCall(expr, dialect_name);
    CellRefs out_variables = model.AddOutVariable(expr);

    Indices arg_indices = getter(expr);
    auto valid_arg_indices = GetValidIndices(call, arg_indices);
    if (valid_arg_indices.empty()) {
      return;
    }
    model.SetCheckedArgIndices(expr, valid_arg_indices);
    for (auto arg_index : valid_arg_indices) {
      // connect the argument's output with this expr's input variable.
      Expr arg = call->args[arg_index];
      CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
      CellRefs in_variables = model.AddInVariable(expr, arg);
      CHECK(in_variables.size() == arg_variables.size());
      for (int i=0; i< arg_variables.size(); ++i) {
        model.AddEqual(arg_variables[i], in_variables[i], true);
      }
    }
    // check for tuple inputs and add constraint to ensure that all fields have the same axis
    for (int i = 0; i < valid_arg_indices.size(); ++i) {
      Index arg_index = valid_arg_indices[i];
      if(call->args[arg_index]->checked_type_->IsInstance<TupleTypeNode>()) {
        CellRefs in_variables = model.GetInVariableRefs(expr, arg_index);
        Types types = ExpandTupleType(Downcast<TupleType>(call->args[arg_index]->checked_type_));
        CHECK_EQ(in_variables.size(), types.size());
        for(int j=0; j<in_variables.size()-1; j++) {
          model.AddEqual(in_variables[j], in_variables[j+1], true);
        }
      }
    }
    // add constraints between different inputs
    for (int i = 0; i < valid_arg_indices.size() - 1; ++i) {
      Index arg_index_1 = valid_arg_indices[i];
      Index arg_index_2 = valid_arg_indices[i + 1];
      CellRefs in_vars_1 = model.GetInVariableRefs(expr, arg_index_1);
      CellRefs in_vars_2 = model.GetInVariableRefs(expr, arg_index_2);
      Types types_1, types_2;
      if(call->args[arg_index_1]->checked_type_->IsInstance<TupleTypeNode>()) {
        CHECK(call->args[arg_index_2]->checked_type_->IsInstance<TupleTypeNode>()) << "Cannot handle mixed tensor and tuple inputs in default matching constraint.";
        types_1 = ExpandTupleType(Downcast<TupleType>(call->args[arg_index_1]->checked_type_));
        types_2 = ExpandTupleType(Downcast<TupleType>(call->args[arg_index_2]->checked_type_));
      } else {
        types_1.push_back(call->args[arg_index_1]->checked_type_);
        types_2.push_back(call->args[arg_index_2]->checked_type_);
      }
      model.AddMatching(in_vars_1, in_vars_2, types_1, types_2, true);
    }
    if (!valid_arg_indices.empty() && !out_variables.empty()) {
      Index arg_index = valid_arg_indices[0];
      CellRefs in_variables = model.GetInVariableRefs(expr, arg_index);
      Types arg_types, out_types;
      if(call->args[arg_index]->checked_type_->IsInstance<TupleTypeNode>()) {
        auto arg_tt = Downcast<TupleType>(call->args[arg_index]->checked_type_);
        if(arg_tt->fields.size() == 1 && call->checked_type_->IsInstance<TensorTypeNode>()) {
          // special case to handle communication
          arg_types.push_back(arg_tt->fields[0]);
          out_types.push_back(call->checked_type_);
        } else {
          CHECK(call->checked_type_->IsInstance<TupleTypeNode>()) << "Cannot handle unmatching tensor and tuple input and output in default matching constraint.";
          arg_types = ExpandTupleType(arg_tt);
          out_types = ExpandTupleType(Downcast<TupleType>(call->checked_type_));
        }
      } else {
        arg_types.push_back(call->args[arg_index]->checked_type_);
        out_types.push_back(call->checked_type_);
      }
      model.AddMatching(in_variables, out_variables, arg_types, out_types, true);
    }
    Type out_type = TryGetLetVar(expr)->checked_type_;
    model.AddDegenerateAxes(out_type, out_variables);
  };
  return matching_constraint;
}

FConstraint CreateMatchingConstraint(std::string dialect_name, Indices arg_indices) {
  // The matching constraint should match all args in arg_indices
  ArgIndicesGetter getter = [=](const Expr& expr) -> Indices {
    return arg_indices;
  };
  return CreateMatchingConstraint(dialect_name, getter);
}

FConstraint CreateMatchingConstraint(std::string dialect_name) {
  // The matching constraint should match all input args
  ArgIndicesGetter getter = [=](const Expr& expr) -> Indices {
    Call call = GetOpCall(expr, dialect_name);
    Indices arg_indices = {};
    for (int i = 0; i < call->args.size(); ++i) {
      arg_indices.push_back(i);
    }
    return arg_indices;
  };
  return CreateMatchingConstraint(dialect_name, getter);
}

FConstraint CreateMatchingConstraint(std::string dialect_name, int n_arg) {
  // The matching constraint should match first n_arg args
  Indices indices = {};
  for (int i = 0; i < n_arg; ++i) {
    indices.push_back(i);
  }
  return CreateMatchingConstraint(dialect_name, indices);
}

FConstraint CreateBroadCastConstraint(std::string dialect_name) {
  // this constraint allows broadcast between the first two args
  FConstraint broadcast_constraint = [=](CPModel& model, Expr expr) {
    Call call = GetOpCall(expr, dialect_name);
    CellRefs out_variables = model.AddOutVariable(expr);

    Indices arg_indices;
    for(int i=0; i< call->args.size(); i++) {
      arg_indices.push_back(i);
    }
    auto valid_arg_indices = GetValidIndices(call, arg_indices);
    if (valid_arg_indices.empty()) {
      return;
    }
    model.SetCheckedArgIndices(expr, valid_arg_indices);
    for (auto arg_index : valid_arg_indices) {
      Expr arg = call->args[arg_index];
      CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
      CellRefs in_variables = model.AddInVariable(expr, arg);
      CHECK(in_variables.size() == arg_variables.size());
      for (int i = 0; i < in_variables.size(); ++i) {
        model.AddEqual(in_variables[i], arg_variables[i], true);
      }
    }

    CellRef matching_constraint = -1;
    // add constraints between different inputs
    for (int i = 0; i < valid_arg_indices.size() - 1; ++i) {
      Index arg_index_1 = valid_arg_indices[i];
      Index arg_index_2 = valid_arg_indices[i + 1];
      CellRefs vars_1 = model.GetInVariableRefs(expr, arg_index_1);
      CellRefs vars_2 = model.GetInVariableRefs(expr, arg_index_2);
      Types types_1, types_2;
      types_1.push_back(call->args[arg_index_1]->checked_type_);
      types_2.push_back(call->args[arg_index_2]->checked_type_);
      Index matching_constraint_i = model.AddMatching(vars_1[0], vars_2[0], types_1[0], types_2[0], false);
      matching_constraint = model.AddAnd(matching_constraint, matching_constraint_i, false);
    }
    // add broadcast constraint
    std::vector<int64_t> arg0_shape = GetShapeFromTensorType(Downcast<TensorType>(call->args[0]->checked_type_));
    std::vector<int64_t> arg1_shape = GetShapeFromTensorType(Downcast<TensorType>(call->args[1]->checked_type_));

    Index nonbroadcast_input_idx = 0;
    if(arg0_shape.size() == 0 || arg1_shape.size() == 0) {
      // arg0 or arg1 is scalar, only need matching constraint since scalar args are automatically duplicated
      if(arg0_shape.size() == 0) {
        nonbroadcast_input_idx = 1;
      } else {
        nonbroadcast_input_idx = 0;
      }
      CellRef true_constraint = model.AddEqual(model.GetConstantCellRef(0), model.GetConstantCellRef(0), false);
      model.AddAnd(true_constraint, matching_constraint, true);
    } else {
      CellRefs vars_0 = model.GetInVariableRefs(expr, 0);
      CellRefs vars_1 = model.GetInVariableRefs(expr, 1);
      if(arg0_shape.size() < arg1_shape.size()) {
        CHECK_EQ(arg0_shape.size(), 1) << "Cannot handle broadcast with more than one non-broadcasted axis.";
        CHECK_EQ(arg1_shape[arg1_shape.size()-1], arg0_shape[0]) << "Broadcast axis does not match.";
        // not partition on the last axis
        CellRef np_constraint = model.AddEqual(vars_0[0], model.GetConstantCellRef(SpecialAxis::kNone), false);
        CellRef neq_constraint = model.AddNotEqual(vars_1[0], model.GetConstantCellRef(arg1_shape.size()-1), false);
        CellRef np_last_axis_constraint = model.AddAnd(np_constraint, neq_constraint, false);
        // Or all partition on the last axis
        CellRef vars0_constraint = model.AddEqual(vars_0[0], model.GetConstantCellRef(0), false);
        CellRef vars1_constraint = model.AddEqual(vars_1[0], model.GetConstantCellRef(arg1_shape.size()-1), false);
        CellRef and_constraint = model.AddAnd(vars0_constraint, vars1_constraint, false);
        model.AddOr(np_last_axis_constraint, and_constraint, true);
        nonbroadcast_input_idx = 1;
      } else if(arg0_shape.size() > arg1_shape.size()) {
        CHECK_EQ(arg1_shape.size(), 1) << "Cannot handle broadcast with more than one non-broadcasted axis.";
        CHECK_EQ(arg0_shape[arg0_shape.size()-1], arg1_shape[0]) << "Broadcast axis does not match.";
        // not partition on the last axis
        CellRef np_constraint = model.AddEqual(vars_1[0], model.GetConstantCellRef(SpecialAxis::kNone), false);
        CellRef neq_constraint = model.AddNotEqual(vars_0[0], model.GetConstantCellRef(arg0_shape.size()-1), false);
        CellRef np_last_axis_constraint = model.AddAnd(np_constraint, neq_constraint, false);
        // Or all partition on the last axis
        CellRef vars1_constraint = model.AddEqual(vars_1[0], model.GetConstantCellRef(0), false);
        CellRef vars0_constraint = model.AddEqual(vars_0[0], model.GetConstantCellRef(arg0_shape.size()-1), false);
        CellRef and_constraint = model.AddAnd(vars0_constraint, vars1_constraint, false);
        model.AddOr(np_last_axis_constraint, and_constraint, true);
      } else {
        CellRef true_constraint = model.AddEqual(model.GetConstantCellRef(0), model.GetConstantCellRef(0), false);
        model.AddAnd(true_constraint, matching_constraint, true);
      }
    }

    if (!valid_arg_indices.empty() && !out_variables.empty()) {
      Index arg_index = valid_arg_indices[nonbroadcast_input_idx];
      CellRefs in_variables = model.GetInVariableRefs(expr, arg_index);
      Types arg_types, out_types;
      arg_types.push_back(call->args[arg_index]->checked_type_);
      out_types.push_back(call->checked_type_);
      model.AddMatching(in_variables, out_variables, arg_types, out_types, true);
    }
    Type out_type = TryGetLetVar(expr)->checked_type_;
    model.AddDegenerateAxes(out_type, out_variables);
  };
  return broadcast_constraint;
}

FConstraint CreateBatchMatMulConstraint(std::string dialect_name, bool transpose_a, bool transpose_b) {
  FConstraint batch_matmul_constraint = [=](CPModel& model, Expr expr) {
    Call call = GetOpCall(expr, dialect_name);
    CellRefs out_variables = model.AddOutVariable(expr);
    model.SetCheckedArgIndices(expr, 2);
    for (int i = 0; i < 2; ++i) {
      Expr arg = call->args[i];
      CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
      CellRefs in_variables = model.AddInVariable(expr, arg);
      CHECK(in_variables.size() == arg_variables.size());
      for (int j=0; j<in_variables.size(); j++) {
        model.AddEqual(arg_variables[j], in_variables[j], true);
      }
    }

    CellRef var_a = model.GetInVariableRefs(expr, 0)[0];
    CellRef var_b = model.GetInVariableRefs(expr, 1)[0];
    CellRef out_var = out_variables[0];
    Index a_partition_axis = transpose_a ? 2 : 1;
    Index b_partition_axis = transpose_b ? 1 : 2;
    Index batch_axis = 0;
    Index out_axis_if_partition_a = 1;
    Index out_axis_if_partition_b = 2;
    CellRef batch_axis_var = model.GetConstantCellRef(batch_axis);
    CellRef out_axis_if_partition_a_var = model.GetConstantCellRef(out_axis_if_partition_a);
    CellRef out_axis_if_partition_b_var = model.GetConstantCellRef(out_axis_if_partition_b);
    CellRef a_partition_axis_var = model.GetConstantCellRef(a_partition_axis);
    CellRef b_partition_axis_var = model.GetConstantCellRef(b_partition_axis);
    CellRef not_partition_var = model.GetConstantCellRef(SpecialAxis::kNone);

    // case 1: both lhs & rhs partition at batch dim
    CellRef constraint_idx_0 = model.AddEqual(out_var, batch_axis_var, false);
    CellRef constraint_idx_1 = model.AddEqual(var_a, batch_axis_var, false);
    CellRef constraint_idx_2 = model.AddEqual(var_b, batch_axis_var, false);
    CellRef constraint_idx_3 = model.AddAnd(constraint_idx_1, constraint_idx_2, false);
    CellRef constraint_idx_4 = model.AddAnd(constraint_idx_0, constraint_idx_3, false);

    // case 2: partition a, not partition b
    CellRef constraint_idx_9 = model.AddEqual(out_var, out_axis_if_partition_a_var, false);
    CellRef constraint_idx_10 = model.AddEqual(var_a, a_partition_axis_var, false);
    CellRef constraint_idx_11 = model.AddEqual(var_b, not_partition_var, false);
    CellRef constraint_idx_12 = model.AddAnd(constraint_idx_9, constraint_idx_10, false);
    CellRef constraint_idx_13 = model.AddAnd(constraint_idx_11, constraint_idx_12, false);

    // case 3: partition b, not partition a
    CellRef constraint_idx_14 = model.AddEqual(out_var, out_axis_if_partition_b_var, false);
    CellRef constraint_idx_15 = model.AddEqual(var_b, b_partition_axis_var, false);
    CellRef constraint_idx_16 = model.AddEqual(var_a, not_partition_var, false);
    CellRef constraint_idx_17 = model.AddAnd(constraint_idx_14, constraint_idx_15, false);
    CellRef constraint_idx_18 = model.AddAnd(constraint_idx_16, constraint_idx_17, false);

    // case 4: all not partition
    CellRef constraint_idx_19 = model.AddEqual(out_var, not_partition_var, false);
    CellRef constraint_idx_20 = model.AddEqual(var_a, not_partition_var, false);
    CellRef constraint_idx_21 = model.AddEqual(var_b, not_partition_var, false);
    CellRef constraint_idx_22 = model.AddAnd(constraint_idx_19, constraint_idx_20, false);
    CellRef constraint_idx_23 = model.AddAnd(constraint_idx_21, constraint_idx_22, false);

    // OR the cases
    CellRef constraint_idx_24 = model.AddOr(constraint_idx_4, constraint_idx_13, false);
    CellRef constraint_idx_25 = model.AddOr(constraint_idx_24, constraint_idx_18, false);
    CellRef constraint_idx_26 = model.AddOr(constraint_idx_25, constraint_idx_23, true);

    Type out_type = TryGetLetVar(expr)->checked_type_;
    model.AddDegenerateAxes(out_type, out_variables);
  };
  return batch_matmul_constraint;
}

FConstraint CreateMatMulConstraint(std::string dialect_name, bool transpose_a, bool transpose_b) {
  FConstraint matmul_constraint = [=](CPModel& model, Expr expr) {
    Call call = GetOpCall(expr, dialect_name);
    CellRefs out_variables = model.AddOutVariable(expr);
    model.SetCheckedArgIndices(expr, 2);
    for (int i = 0; i < 2; ++i) {
      Expr arg = call->args[i];
      CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
      CellRefs in_variables = model.AddInVariable(expr, arg);
      CHECK(in_variables.size() == arg_variables.size());
      for (int j = 0; j < in_variables.size(); ++j) {
        model.AddEqual(arg_variables[j], in_variables[j], true);
      }
    }

    CellRef var_a = model.GetInVariableRefs(expr, 0)[0];
    CellRef var_b = model.GetInVariableRefs(expr, 1)[0];
    CellRef var_out = out_variables[0];
    Index a_partition_axis = transpose_a ? 1 : 0;
    Index b_partition_axis = transpose_b ? 0 : 1;
    Index out_axis_if_partition_a = 0;
    Index out_axis_if_partition_b = 1;
    CellRef out_axis_if_partition_a_var = model.GetConstantCellRef(out_axis_if_partition_a);
    CellRef out_axis_if_partition_b_var = model.GetConstantCellRef(out_axis_if_partition_b);
    CellRef a_partition_axis_var = model.GetConstantCellRef(a_partition_axis);
    CellRef b_partition_axis_var = model.GetConstantCellRef(b_partition_axis);
    CellRef not_partition_var = model.GetConstantCellRef(SpecialAxis::kNone);
    CellRef exp_partition_var = model.GetConstantCellRef(SpecialAxis::kExpert);

    // case 1: not partition
    CellRef constraint_idx_0 = model.AddEqual(var_out, not_partition_var, false);
    CellRef constraint_idx_1 = model.AddEqual(var_a, not_partition_var, false);
    CellRef constraint_idx_2 = model.AddEqual(var_b, not_partition_var, false);
    CellRef constraint_idx_3 = model.AddAnd(constraint_idx_0, constraint_idx_1, false);
    CellRef constraint_idx_4 = model.AddAnd(constraint_idx_2, constraint_idx_3, false);

    // case 2: partition a, not partition b
    CellRef constraint_idx_5 = model.AddEqual(var_out, out_axis_if_partition_a_var, false);
    CellRef constraint_idx_6 = model.AddEqual(var_a, a_partition_axis_var, false);
    CellRef constraint_idx_7 = model.AddEqual(var_b, not_partition_var, false);
    CellRef constraint_idx_8 = model.AddAnd(constraint_idx_5, constraint_idx_6, false);
    CellRef constraint_idx_9 = model.AddAnd(constraint_idx_7, constraint_idx_8, false);

    // case 3: partition b, not partition a
    CellRef constraint_idx_10 = model.AddEqual(var_out, out_axis_if_partition_b_var, false);
    CellRef constraint_idx_11 = model.AddEqual(var_b, b_partition_axis_var, false);
    CellRef constraint_idx_12 = model.AddEqual(var_a, not_partition_var, false);
    CellRef constraint_idx_13 = model.AddAnd(constraint_idx_10, constraint_idx_11, false);
    CellRef constraint_idx_14 = model.AddAnd(constraint_idx_12, constraint_idx_13, false);

    // case 4: exactly one of a or b is partitoned at expert axis
    CellRef constraint_idx_15 = model.AddEqual(var_out, exp_partition_var, false);

    CellRef constraint_idx_16 = model.AddEqual(var_a, exp_partition_var, false);
    CellRef constraint_idx_17 = model.AddEqual(var_b, not_partition_var, false);
    CellRef constraint_idx_18 = model.AddAnd(constraint_idx_16, constraint_idx_17, false);

    CellRef constraint_idx_19 = model.AddEqual(var_b, exp_partition_var, false);
    CellRef constraint_idx_20 = model.AddEqual(var_a, not_partition_var, false);
    CellRef constraint_idx_21 = model.AddAnd(constraint_idx_19, constraint_idx_20, false);

    CellRef constraint_idx_22 = model.AddOr(constraint_idx_18, constraint_idx_21, false);
    CellRef constraint_idx_23 = model.AddAnd(constraint_idx_15, constraint_idx_22, false);

    // Or all above cases
    CellRef constraint_idx_24 = model.AddOr(constraint_idx_4, constraint_idx_9, false);
    CellRef constraint_idx_25 = model.AddOr(constraint_idx_14, constraint_idx_24, false);
    CellRef constraint_idx_26 = model.AddOr(constraint_idx_23, constraint_idx_25, true);

    Type out_type = TryGetLetVar(expr)->checked_type_;
    model.AddDegenerateAxes(out_type, out_variables);
  };
  return matmul_constraint;
}

// FConstraint TopKConstraint = [](CPModel& model, Expr expr) {
//   Type type = TryGetLetVar(expr)->checked_type_;
//   if (type->IsInstance<TensorTypeNode>()) {
//     FConstraint matching_constraint = CreateMatchingConstraint("raf.op.tvm.topk", 1);
//     matching_constraint(model, expr);
//     return;
//   }

//   Call call = GetOpCall(expr, "raf.op.tvm.topk");
//   std::string call_name = GenerateExprString(call);
//   std::string call_constraint_name = std::string("constraint_") + call_name;
//   Indices out_indices = model.AddOutVariable(expr, call_name);

//   model.SetCheckedArgIndices(expr, 1);
//   std::string arg_name = call_name + "_arg_0";
//   std::string arg_conn_constraint_name = std::string("conn_") + call_constraint_name + "_0";
//   Expr arg = call->args[0];
//   Indices arg_indices = model.GetOutVariableOrAddAsExternal(arg);
//   Indices in_indices = model.AddInVariable(expr, arg, arg_name);
//   CHECK(in_indices.size() == arg_indices.size());
//   model.AddEqual(arg_conn_constraint_name, arg_indices, in_indices);

//   Index arg_index = arg_indices[0];
//   Index out_index = out_indices[0];
//   Type arg_type = call->args[0]->checked_type_;
//   Type expr_type = Downcast<TupleType>(TryGetLetVar(expr)->checked_type_)->fields[0];
//   std::string matching_constraint_name = std::string("matching_arg_rv_") + call_constraint_name;
//   model.AddMatching(matching_constraint_name, arg_index, out_index, arg_type, expr_type, true);
//   for (int i = 0; i < out_indices.size() - 1; ++i) {
//     std::string eq_constraint_name = call_constraint_name + "_" + std::to_string(i);
//     model.AddEqual(eq_constraint_name, out_indices[i], out_indices[i + 1], true);
//   }

//   std::string degenerate_constraint_name = std::string("redundant_") + call_constraint_name;
//   Type out_type = TryGetLetVar(expr)->checked_type_;
//   model.AddDegenerateAxes(degenerate_constraint_name, out_type, out_indices);
// };

FConstraint SplitConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.tvm.split");
  CellRefs out_variables = model.AddOutVariable(expr);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size());
  for(int i = 0; i < in_variables.size(); ++i) {
    model.AddEqual(arg_variables[i], in_variables[i], true);
  }

  CellRef arg_var = arg_variables[0];
  CellRef out_var = out_variables[0];
  Type arg_type = call->args[0]->checked_type_;
  Type expr_type = Downcast<TupleType>(TryGetLetVar(expr)->checked_type_)->fields[0];
  model.AddMatching(arg_var, out_var, arg_type, expr_type, true);
  int ndim = Downcast<TensorType>(arg->checked_type_)->shape.size();
  int axis = NormalizeAxis(GetInt(call->args[2]), ndim);
  CellRef axis_const_var = model.GetConstantCellRef(axis);
  model.AddNotEqual(arg_var, axis_const_var, true);
  model.AddNotEqual(out_var, axis_const_var, true);
  for (int i = 0; i < out_variables.size() - 1; ++i) {
    model.AddEqual(out_variables[i], out_variables[i + 1], true);
  }

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint ConcatConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.tvm.concatenate");
  CellRefs out_variables = model.AddOutVariable(expr);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);  
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size());
  for(int i = 0; i < in_variables.size(); ++i) {
    model.AddEqual(arg_variables[i], in_variables[i], true);
  }

  CellRef in_var = in_variables[0];
  CellRef out_var = out_variables[0];
  Type arg_type = Downcast<TupleType>(arg->checked_type_)->fields[0];
  Type expr_type = TryGetLetVar(expr)->checked_type_;
  model.AddMatching(in_var, out_var, arg_type, expr_type, true);
  for (int i = 0; i < in_variables.size() - 1; ++i) {
    // All the concat tuple fields' partition axes should be equal.
    model.AddEqual(in_variables[i], in_variables[i + 1], true);
  }

  model.AddDegenerateAxes(expr_type, out_variables);
};

FConstraint CreateScanConstraint(std::string dialect_name) {
  FConstraint scan_constraint = [=](CPModel& model, Expr expr) {
    // partition axis should not be the scan axis.
    FConstraint matching_constraint = CreateMatchingConstraint(dialect_name, 1);
    matching_constraint(model, expr);

    Call call = GetOpCall(expr, dialect_name);
    int ndim = Downcast<TensorType>(call->args[0]->checked_type_)->shape.size();
    int axis = NormalizeAxis(GetInt(call->args[1]), ndim);

    CellRef axis_var = model.GetConstantCellRef(axis);
    CellRef out_var = model.AddOutVariable(expr)[0];
    model.AddNotEqual(out_var, axis_var, true);
  };
  return scan_constraint;
}

FConstraint LayerNormConstraint = [](CPModel& model, Expr expr) {
  // To simplify implementation, we only allow the partition axis to be the first (batch) axis.
  Call call = GetOpCall(expr, "raf.op.tvm.layer_norm");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 3);
  CellRefs in_variables;
  for (int i = 0; i < 3; ++i) {
    Expr arg = call->args[i];
    CellRefs arg_vars = model.GetOutVariableOrAddAsExternal(arg);
    CellRefs in_vars = model.AddInVariable(expr, arg);
    CHECK(in_vars.size() == arg_vars.size() && in_vars.size() == 1);
    model.AddEqual(arg_vars[0], in_vars[0], true);
    in_variables.push_back(in_vars[0]);
  }
  CellRef np_axis = model.GetConstantCellRef(SpecialAxis::kNone);
  // only partition x and output (y) at batch axis
  CellRef c1 = model.AddEqual(in_variables[0], model.GetConstantCellRef(0), false);
  CellRef c2 = model.AddEqual(in_variables[1], np_axis, false);
  CellRef c3 = model.AddEqual(in_variables[2], np_axis, false);
  CellRef c4 = model.AddEqual(out_variables[0], model.GetConstantCellRef(0), false);
  CellRef c5 = model.AddAnd(c1, c2, false);
  CellRef c6 = model.AddAnd(c3, c4, false);
  CellRef c7 = model.AddAnd(c5, c6, false);

  // other wise don't partition
  CellRef c8 = model.AddEqual(in_variables[0], np_axis, false);
  CellRef c9 = model.AddEqual(in_variables[1], np_axis, false);
  CellRef c10 = model.AddEqual(in_variables[2], np_axis, false);
  CellRef c11 = model.AddEqual(out_variables[0], np_axis, false);
  CellRef c12 = model.AddAnd(c8, c9, false);
  CellRef c13 = model.AddAnd(c10, c11, false);
  CellRef c14 = model.AddAnd(c12, c13, false);

  model.AddOr(c7, c14, true);

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint LayerNormDxConstraint = [](CPModel& model, Expr expr) {
  // To simplify implementation, we only allow the partition axis to be the first (batch) axis.
  Call call = GetOpCall(expr, "raf.op.tvm.layer_norm_dx");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 3);

  model.SetCheckedArgIndices(expr, 3);
  CellRefs in_variables;
  for (int i = 0; i < 3; ++i) {
    Expr arg = call->args[i];
    CellRefs arg_vars = model.GetOutVariableOrAddAsExternal(arg);
    CellRefs in_vars = model.AddInVariable(expr, arg);
    CHECK(in_vars.size() == arg_vars.size() && in_vars.size() == 1);
    model.AddEqual(arg_vars[0], in_vars[0], true);
    in_variables.push_back(in_vars[0]);
  }
  CellRef np_axis = model.GetConstantCellRef(SpecialAxis::kNone);
  //x, dy and dx should be partitioned at batch axis, others unpartitioned
  // in_variables[0]: x, [1]: scale, [2]: dy
  // out_variables[0]: dx, [1]: dw, [2]: db
  CellRef c1 = model.AddEqual(in_variables[0], model.GetConstantCellRef(0), false);
  CellRef c2 = model.AddEqual(in_variables[1], np_axis, false);
  CellRef c3 = model.AddEqual(in_variables[2], model.GetConstantCellRef(0), false);

  CellRef c4 = model.AddEqual(out_variables[0], model.GetConstantCellRef(0), false);
  CellRef c5 = model.AddEqual(out_variables[1], np_axis, false);
  CellRef c6 = model.AddEqual(out_variables[2], np_axis, false);

  CellRef c7 = model.AddAnd(c1, c2, false);
  CellRef c8 = model.AddAnd(c3, c4, false);
  CellRef c9 = model.AddAnd(c5, c6, false);
  CellRef c10 = model.AddAnd(c7, c8, false);
  CellRef c11 = model.AddAnd(c9, c10, false);

  // otherwise don't partition
  CellRef c12 = model.AddEqual(in_variables[0], np_axis, false);
  CellRef c13 = model.AddEqual(in_variables[1], np_axis, false);
  CellRef c14 = model.AddEqual(in_variables[2], np_axis, false);
  CellRef c15 = model.AddEqual(out_variables[0], np_axis, false);
  CellRef c16 = model.AddEqual(out_variables[1], np_axis, false);
  CellRef c17 = model.AddEqual(out_variables[2], np_axis, false);

  CellRef c18 = model.AddAnd(c12, c13, false);
  CellRef c19 = model.AddAnd(c14, c15, false);
  CellRef c20 = model.AddAnd(c16, c17, false);
  CellRef c21 = model.AddAnd(c18, c19, false);
  CellRef c22 = model.AddAnd(c20, c21, false);

  model.AddOr(c11, c22, true);

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint CreateReduceConstraint(std::string dialect_name) {
  FConstraint reduce_constraint = [=](CPModel& model, Expr expr) {
    FConstraint matching_constraint = CreateMatchingConstraint(dialect_name, 1);
    matching_constraint(model, expr);

    CellRefs out_variables = model.GetOutVariableCellRefs(expr);
    Call call = GetOpCall(expr, dialect_name);
    CellRef in_var = model.GetInVariableRefs(expr, 0)[0];
    if (out_variables[0] == -1) {
      // if sum result is a scalar, then its argument should not be partitioned.
      CellRef not_partition_var = model.GetConstantCellRef(SpecialAxis::kNone);
      model.AddEqual(in_var, not_partition_var, true);
    }
    // should not partition input at reduce axis
    std::vector<int> reduce_axes;
    if (call->args[1].as<ConstantNode>()->value->IsInstance<TupleValueObj>()) {
      auto fields = Downcast<TupleValue>(call->args[1].as<ConstantNode>()->value)->fields;
      for (auto field : fields) {
        reduce_axes.push_back(Downcast<IntValue>(field)->value);
      }
    } else {
      reduce_axes = {GetInt(call->args[1])};
    }
    CHECK(call->args[0]->checked_type_.defined() && call->args[0]->checked_type_->IsInstance<TensorTypeNode>());
    auto input_dim = Downcast<TensorType>(call->args[0]->checked_type_)->shape.size();
    for (auto axis: reduce_axes) {
      auto normalized_axis = NormalizeAxis(axis, input_dim);
      model.AddNotEqual(in_var, model.GetConstantCellRef(normalized_axis), true);
    }
  };
  return reduce_constraint;
}

FConstraint WhereConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.tvm.where");
  CellRefs out_variables = model.AddOutVariable(expr);

  Expr cond = call->args[0];
  Expr x = call->args[1];
  Expr y = call->args[2];
  std::vector<int64_t> cond_shape = GetShapeFromTensorType(Downcast<TensorType>(cond->checked_type_));
  std::vector<int64_t> x_shape = GetShapeFromTensorType(Downcast<TensorType>(x->checked_type_));
  std::vector<int64_t> y_shape = GetShapeFromTensorType(Downcast<TensorType>(y->checked_type_));
  std::vector<int64_t> broadcasted_shape = GetShapeFromTensorType(Downcast<TensorType>(TryGetLetVar(expr)->checked_type_));

  // LOG(INFO) << "[WHERE] cond shape: " << Downcast<TensorType>(cond->checked_type_);
  // LOG(INFO) << "[WHERE] x shape: " << Downcast<TensorType>(x->checked_type_);
  // LOG(INFO) << "[WHERE] y shape: " << Downcast<TensorType>(y->checked_type_);
  // LOG(INFO) << "[WHERE] out shape: " << Downcast<TensorType>(expr->checked_type_);
  std::unordered_set<int> non_broadcast_arg_indices;
  std::unordered_set<int> np_axes;
  std::vector<std::vector<int64_t>> shapes = {cond_shape, x_shape, y_shape};
  for(int i=0; i<3; i++) {
    auto& a_shape = shapes[i];
    bool a_broadcasted = false;
    int a_idx = 0;
    int b_idx = 0;
    std::unordered_set<int> a_nonbroadcast_axes;
    while(a_idx < a_shape.size() && b_idx < broadcasted_shape.size()) {
      if(a_shape[a_idx] == broadcasted_shape[b_idx]) {
        a_nonbroadcast_axes.insert(b_idx);
        a_idx++;
        b_idx++;
      } else if(a_shape[a_idx] == 1) {
        a_broadcasted = true;
        a_idx++;
        b_idx++;
      } else {
        b_idx++;
      }
    }
    if(a_nonbroadcast_axes.size() != broadcasted_shape.size()) {
      a_broadcasted = true;
    }
    if(!a_broadcasted) {
      // LOG(INFO) << "[WHERE] arg " << i << " is NOT broadcasted.";
      non_broadcast_arg_indices.insert(i);
    } else {
      for(auto axis: a_nonbroadcast_axes) {
        np_axes.insert(axis);
      }
      // LOG(INFO) << "[WHERE] arg " << i << " is broadcasted.";
    }
  }

  // for(auto axis: np_axes) {
  //   LOG(INFO) << "[WHERE] np axes: " << axis;
  // }

  if(non_broadcast_arg_indices.size() == 3) {
    // no broadcast, use matching constraint
    FConstraint matching_constraint = CreateMatchingConstraint("raf.op.tvm.where", 3);
    matching_constraint(model, expr);
  } else {
    // some args broadcasted
    Indices arg_indices;
    for(int i=0; i< call->args.size(); i++) {
      arg_indices.push_back(i);
    }
    auto valid_arg_indices = GetValidIndices(call, arg_indices);
    if (valid_arg_indices.empty()) {
      return;
    }
    model.SetCheckedArgIndices(expr, valid_arg_indices);
    for (auto arg_index : valid_arg_indices) {
      Expr arg = call->args[arg_index];
      CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
      CellRefs in_variables = model.AddInVariable(expr, arg);
      CHECK(in_variables.size() == arg_variables.size());
      for(int i=0; i<in_variables.size(); i++) {
        model.AddEqual(arg_variables[i], in_variables[i], true);
      }
    }
    for(auto arg_index: valid_arg_indices) {
      CellRefs arg_variables = model.GetInVariableRefs(expr, arg_index);
      if(non_broadcast_arg_indices.count(arg_index)) {
        // this arg is not broadcasted..
        // connect it with out
        model.AddEqual(arg_variables[0], out_variables[0], true);
      } else {
        // this arg is broadcasted
        model.AddEqual(arg_variables[0], model.GetConstantCellRef(SpecialAxis::kNone), true);
      }
    }
    // add constraint to out
    // partitioned axis != np_axes
    for(auto axis: np_axes) {
      model.AddNotEqual(out_variables[0], model.GetConstantCellRef(axis), true);
    }
  }
};

FConstraint AllgatherConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.nccl._allgather");
  CellRefs out_variables = model.AddOutVariable(expr);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size() && in_variables.size() == out_variables.size());
  for (int i = 0; i < out_variables.size(); ++i) {
    model.AddEqual(arg_variables[i], in_variables[i], true);
    model.AddEqual(out_variables[i], in_variables[i], true);
  }

  int axis = GetInt(call->args[1]);
  CHECK_EQ(axis, 0);
  Index axis_index = model.GetConstantCellRef(axis);
  for (int i = 0; i < in_variables.size(); ++i) {
    model.AddNotEqual(in_variables[i], axis_index, true);
  }

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint BiasAddConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.tvm.bias_add");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 2);
  for (int i = 0; i < 2; ++i) {
    Expr arg = call->args[i];
    CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
    CellRefs in_variables = model.AddInVariable(expr, arg);
    CHECK(in_variables.size() == arg_variables.size());
    for (int i = 0; i < in_variables.size(); ++i) {
      model.AddEqual(arg_variables[i], in_variables[i], true);
    }
  }

  CellRef arg_var = model.GetInVariableRefs(expr, 0)[0];
  model.AddEqual(arg_var, out_variables[0], true);

  // constraint is (arg_0 axis == bias axis && bias_arg axis == 0) ||
  //  (arg_0 axis != bias axis && bias_arg axis == -1)
  Type out_type = TryGetLetVar(expr)->checked_type_;
  int bias_axis = NormalizeAxis(GetInt(call->args[2]), Downcast<TensorType>(out_type)->shape.size());
  CellRef bias_axis_var = model.GetConstantCellRef(bias_axis);
  CellRef not_partition_var = model.GetConstantCellRef(SpecialAxis::kNone);
  CellRef partition_bias_var = model.GetConstantCellRef(0);
  CellRef bias_var = model.GetInVariableRefs(expr, 1)[0];
  CellRef constraint_idx_1 = model.AddEqual(arg_var, bias_axis_var, false);
  CellRef constraint_idx_2 = model.AddEqual(bias_var, partition_bias_var, false);
  CellRef constraint_idx_3 = model.AddAnd(constraint_idx_1, constraint_idx_2, false);
  CellRef constraint_idx_4 = model.AddNotEqual(arg_var, bias_axis_var, false);
  CellRef constraint_idx_5 = model.AddEqual(bias_var, not_partition_var, false);
  CellRef constraint_idx_6 = model.AddAnd(constraint_idx_4, constraint_idx_5, false);
  model.AddOr(constraint_idx_3, constraint_idx_6, true);

  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint SoftmaxDxConstraint = [](CPModel& model, Expr expr) {
  FConstraint matching_constraint = CreateMatchingConstraint("raf.op.tvm.softmax_dx", 3);
  matching_constraint(model, expr);

  auto in_variables = model.GetInVariableCellRefsVec(expr);
  for (int i = 0; i < 2; ++i) {
    // x, y, dy arg's axes should be equal
    model.AddEqual(in_variables[i][0], in_variables[i + 1][0], true);
  }
};

FConstraint ArgmaxConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.tvm.argmax");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size());
  for (int i = 0; i < in_variables.size(); ++i) {
    model.AddEqual(arg_variables[i], in_variables[i], true);
  }

  bool keepdims = GetBool(call->args[2]);
  if (!keepdims) {
    // When keepdims is false, explicitly find matching axes using the reduce axis.
    Indices axes = GetInts<Index>(call->args[1]);
    bool exclude = GetBool(call->args[3]);
    auto matching_axes = FindMatchingAxes(arg->checked_type_, axes, exclude);
    if (matching_axes.empty()) {
      // All axes are reduced. Should not partition anymore.
      CellRef not_partition_var = model.GetConstantCellRef(SpecialAxis::kNone);
      model.AddEqual(in_variables[0], not_partition_var, true);
      model.AddEqual(out_variables[0], not_partition_var, true);
    } else {
      model.AddMatching(in_variables[0], out_variables[0], matching_axes, true);
      Type out_type = TryGetLetVar(expr)->checked_type_;
      model.AddDegenerateAxes(out_type, out_variables);
    }
  } else {
    // Should not partition argmax when keepdims is true because current
    // partitioning method doesn't support the required concatenating method.
    CellRef not_partition_var = model.GetConstantCellRef(SpecialAxis::kNone);
    model.AddEqual(in_variables[0], not_partition_var, true);
    model.AddEqual(out_variables[0], not_partition_var, true);
  }
};

inline IndexMap<Index> FindReshapePartitionAxes(Type type_in, Type type_out) {
  auto shape_in = ArrayToInt(Downcast<TensorType>(type_in)->shape);
  auto shape_out = ArrayToInt(Downcast<TensorType>(type_out)->shape);

  int type_in_idx = 0;
  int type_out_idx = 0;

  IndexMap<Index> result = {};

  int type_in_product = 1;
  int type_out_product = 1;

  while(type_in_idx < shape_in.size() && type_out_idx < shape_out.size()) {
    if(shape_in[type_in_idx] == shape_out[type_out_idx]) {
      // axis with matching dim, ok to partition
      if(shape_in[type_in_idx] != 1) {
        // ignore cases where dim = 1
        result[type_in_idx] = type_out_idx;
      }
    } else {
      // not equal
      // first unmatched axis, allow partition
      while(shape_in[type_in_idx] == 1) {
        type_in_idx ++;
      }
      while(shape_out[type_out_idx] == 1) {
        type_out_idx ++;
      }
      result[type_in_idx] = type_out_idx;
      type_in_product *= shape_in[type_in_idx];
      type_out_product *= shape_out[type_out_idx];
      while(type_in_product != type_out_product) {
        if(type_in_product < type_out_product) {
          type_in_idx ++;
          type_in_product *= shape_in[type_in_idx];
        } else {
          type_out_idx ++;
          type_out_product *= shape_out[type_out_idx];
        }
      }
      type_in_product = 1;
      type_out_product = 1;
    }
    type_in_idx ++;
    type_out_idx ++;
  }
  return result;
}

FConstraint ReshapeConstraint = [](CPModel& model, Expr expr) {
  Call call = GetOpCall(expr, "raf.op.tvm.reshape");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size());
  for (size_t i = 0; i < in_variables.size(); ++i) {
    model.AddEqual(in_variables[i], arg_variables[i], true);
  }

  Type out_type = TryGetLetVar(expr)->checked_type_;

  // partition at matching axes
  auto partitionable_indices_map = FindReshapePartitionAxes(arg->checked_type_, out_type);
  CellRef or_constraints = -1;
  for(auto it: partitionable_indices_map) {
    CellRef constr_in_eq = model.AddEqual(in_variables[0], model.GetConstantCellRef(it.first), false);
    CellRef constr_out_eq = model.AddEqual(out_variables[0], model.GetConstantCellRef(it.second), false);
    CellRef and_constr = model.AddAnd(constr_in_eq, constr_out_eq, false);
    or_constraints = model.AddOr(or_constraints, and_constr, false);
  }
  // or both partition at special axis
  CellRef constr_in_eq = model.AddEqual(in_variables[0], model.GetConstantCellRef(SpecialAxis::kExpert), false);
  CellRef constr_out_eq = model.AddEqual(out_variables[0], model.GetConstantCellRef(SpecialAxis::kExpert), false);
  CellRef spx_constr = model.AddAnd(constr_in_eq, constr_out_eq, false);
  // no partition
  CellRef np_constraint_in = model.AddEqual(in_variables[0], model.GetConstantCellRef(SpecialAxis::kNone), false);
  CellRef np_constraint_out = model.AddEqual(out_variables[0], model.GetConstantCellRef(SpecialAxis::kNone), false);
  CellRef np_and = model.AddAnd(np_constraint_in, np_constraint_out, false);
  or_constraints = model.AddOr(or_constraints, spx_constr, false);
  model.AddOr(or_constraints, np_and, true);
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint MoeEncodeConstraint = [](CPModel& model, Expr expr) {
  // Partition axes:
  // Inputs: data: 0, gate: 0, used_capacity: SpecialAxis::kNone
  // Outputs:
  //  gates_s: 0
  //  indices_locations: SpecialAxis::kIndiceAndLocations
  //  accum_used_capacity: SpecialAxis::kNone (not partitioned)
  //  elements_per_expert: SpecialAxis::kNone (not partitioned)
  //  dispatched_inputs: SpecialAxis:kExpert (not partitioned, partially filled)
  Call call = GetOpCall(expr, "raf.op.cuda.moe_encode");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK_EQ(out_variables.size(), 5);

  model.SetCheckedArgIndices(expr, 3);
  for (int i = 0; i < 3; ++i) {
    Expr arg = call->args[i];
    CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
    CellRefs in_variables = model.AddInVariable(expr, arg);
    CHECK(in_variables.size() == arg_variables.size());
    for (size_t j = 0; j < in_variables.size(); ++j) {
      model.AddEqual(in_variables[j], arg_variables[j], true);
    }
  }

  CellRef d_var = model.GetInVariableRefs(expr, 0)[0]; // data
  CellRef g_var = model.GetInVariableRefs(expr, 1)[0]; // gate
  CellRef uc_var = model.GetInVariableRefs(expr, 2)[0]; // used_capacity
  CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
  CellRef s_var = model.GetConstantCellRef(0); // dim S
  CellRef ind_loc_var = model.GetConstantCellRef(SpecialAxis::kIndiceAndLocations);
  CellRef exp_var = model.GetConstantCellRef(SpecialAxis::kExpert);
  // output is [gate_s, indices_locations, accum_used_capacity, elements_per_expert, dispatched_inputs]
  CellRefs vars_to_partition =   {d_var, g_var, uc_var, out_variables[0], out_variables[1], out_variables[2], out_variables[3], out_variables[4]};
  CellRefs partition_axis_vars = {s_var, s_var, np_var,     s_var,         ind_loc_var,        np_var,             np_var,          exp_var};
  CellRef p_rule = -1;
  for (int i = 0; i < vars_to_partition.size(); ++i) {
    CellRef rule_var = model.AddEqual(vars_to_partition[i], partition_axis_vars[i], false);
    p_rule = model.AddAnd(p_rule, rule_var, false);
  }
  CellRef np_rule = -1;
  for (int i = 0; i < vars_to_partition.size(); ++i) {
    CellRef rule_var = model.AddEqual(vars_to_partition[i], np_var, false);
    np_rule = model.AddAnd(np_rule, rule_var, false);
  }
  model.AddOr(p_rule, np_rule, true);

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint MoeEncodeBPRConstraint = [](CPModel& model, Expr expr) {
  // Partition axes:
  // Inputs: data: SpecialAxis::kNone, gate: SpecialAxis::kNone
  // Outputs:
  //  gates_s: 0
  //  indices_locations: SpecialAxis::kIndiceAndLocations
  //  accum_used_capacity: SpecialAxis::kNone (not partitioned)
  //  elements_per_expert: SpecialAxis::kNone (not partitioned)
  //  dispatched_inputs: SpecialAxis:kExpert (not partitioned, partially filled)
  Call call = GetOpCall(expr, "raf.op.cuda.moe_encode_batch_prioritized");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK_EQ(out_variables.size(), 5);

  model.SetCheckedArgIndices(expr, 2);
  for (int i = 0; i < 2; ++i) {
    Expr arg = call->args[i];
    CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
    CellRefs in_variables = model.AddInVariable(expr, arg);
    CHECK(in_variables.size() == arg_variables.size());
    for (size_t j = 0; j < in_variables.size(); ++j) {
      model.AddEqual(in_variables[j], arg_variables[j], true);
    }
  }

  CellRef d_var = model.GetInVariableRefs(expr, 0)[0]; // data
  CellRef g_var = model.GetInVariableRefs(expr, 1)[0]; // gate
  CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
  CellRef s_var = model.GetConstantCellRef(0); // dim S
  CellRef ind_loc_var = model.GetConstantCellRef(SpecialAxis::kIndiceAndLocations);
  CellRef exp_var = model.GetConstantCellRef(SpecialAxis::kExpert);
  // output is [gate_s, indices_locations, accum_used_capacity, elements_per_expert, dispatched_inputs]
  CellRefs vars_to_partition =   {d_var,  g_var, out_variables[0], out_variables[1], out_variables[2], out_variables[3], out_variables[4]};
  CellRefs partition_axis_vars = {np_var, np_var,     s_var,         ind_loc_var,        np_var,             np_var,          exp_var};
  CellRef p_rule = -1;
  for (int i = 0; i < vars_to_partition.size(); ++i) {
    CellRef rule_var = model.AddEqual(vars_to_partition[i], partition_axis_vars[i], false);
    p_rule = model.AddAnd(p_rule, rule_var, false);
  }
  CellRef np_rule = -1;
  for (int i = 0; i < vars_to_partition.size(); ++i) {
    CellRef rule_var = model.AddEqual(vars_to_partition[i], np_var, false);
    np_rule = model.AddAnd(np_rule, rule_var, false);
  }
  model.AddOr(p_rule, np_rule, true);

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint MoeDecodeConstraint = [](CPModel& model, Expr expr) {
  // Partition axes:
  // Inputs:
  //  data: SpecialAxis:kExpert
  //  gate: 0
  //  indices_locations: SpecialAxis::kIndiceAndLocations
  // Outputs:
  //  dispatched_inputs: 0
  Call call = GetOpCall(expr, "raf.op.cuda.moe_decode");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 3);
  for (int i = 0; i < 3; ++i) {
    Expr arg = call->args[i];
    CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
    CellRefs in_variables = model.AddInVariable(expr, arg);
    CHECK(in_variables.size() == arg_variables.size());
    for (size_t j = 0; j < in_variables.size(); ++j) {
      model.AddEqual(in_variables[j], arg_variables[j], true);
    }
  }

  CellRef d_var = model.GetInVariableRefs(expr, 0)[0]; // data
  CellRef g_var = model.GetInVariableRefs(expr, 1)[0]; // gate
  CellRef i_var = model.GetInVariableRefs(expr, 2)[0]; // indices_locations
  CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
  CellRef s_var = model.GetConstantCellRef(0); // dim S
  CellRef ind_loc_var = model.GetConstantCellRef(SpecialAxis::kIndiceAndLocations);
  CellRef exp_var = model.GetConstantCellRef(SpecialAxis::kExpert);
  CellRefs variables_to_partition = {d_var,   g_var,    i_var,  out_variables[0]};
  CellRefs partition_axis_vars =    {exp_var, s_var, ind_loc_var,    s_var};
  CellRef p_rule = -1;
  for (int i = 0; i < 4; ++i) {
    CellRef rule_var = model.AddEqual(variables_to_partition[i], partition_axis_vars[i], false);
    p_rule = model.AddAnd(p_rule, rule_var, false);
  }
  CellRef np_rule = -1;
  for (int i = 0; i < 4; ++i) {
    CellRef rule_var = model.AddEqual(variables_to_partition[i], np_var, false);
    np_rule = model.AddAnd(np_rule, rule_var, false);
  }
  model.AddOr(p_rule, np_rule, true);

  Type out_type = TryGetLetVar(expr)->checked_type_;
  model.AddDegenerateAxes(out_type, out_variables);
};

FConstraint AllToAllConstraint = [](CPModel& model, Expr expr) {
  // Partition axes:
  // Inputs:
  //  data: SpecialAxis:kExpert // any axis except the first
  // Outputs:
  //  dispatched_inputs: SpecialAxis:kExpert // any axis except the first
  Call call = GetOpCall(expr, "raf.op.nccl._all_to_all");
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size());
  for (size_t j = 0; j < in_variables.size(); ++j) {
    model.AddEqual(in_variables[j], arg_variables[j], true);
  }
  TupleType arg_type = Downcast<TupleType>(arg->checked_type_);
  Type out_type = call->checked_type_;
  Types arg_type_fields = ExpandTupleType(arg_type);
  CHECK_EQ(arg_type_fields.size(), 1);
  model.AddMatching(out_variables[0], in_variables[0], out_type, arg_type_fields[0], true);
  // additional constaint: not partitioned on the first axis
  CellRef first_axis_var = model.GetConstantCellRef(0);
  model.AddNotEqual(in_variables[0], first_axis_var, true);
  model.AddNotEqual(out_variables[0], first_axis_var, true);
};

FConstraint SoftmaxConstraint = [](CPModel& model, Expr expr) {
  Call call = GetCall(expr);;
  CellRefs out_variables = model.AddOutVariable(expr);
  CHECK(out_variables.size() == 1);

  model.SetCheckedArgIndices(expr, 1);
  Expr arg = call->args[0];
  CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
  CellRefs in_variables = model.AddInVariable(expr, arg);
  CHECK(in_variables.size() == arg_variables.size());
  for (size_t j = 0; j < in_variables.size(); ++j) {
    model.AddEqual(in_variables[j], arg_variables[j], true);
  }
  TensorType arg_type = Downcast<TensorType>(arg->checked_type_);
  Type out_type = call->checked_type_;
  model.AddMatching(out_variables[0], in_variables[0], out_type, arg_type, true);
  // additional constaint: not partitioned on reduction axis
  int axis = NormalizeAxis(GetInt(call->args[1]), arg_type->shape.size());
  CellRef reduction_axis_var = model.GetConstantCellRef(axis);
  model.AddNotEqual(in_variables[0], reduction_axis_var, true);
  model.AddNotEqual(out_variables[0], reduction_axis_var, true);
};

// FConstraint MoeEncodeDxConstraint = [](CPModel& model, Expr expr) {
//   Call call = GetOpCall(expr, "raf.op.cuda.moe_encode_dx");
//   CellRefs out_variables = model.AddOutVariable(expr);
//   CHECK(out_variables.size() == 1);

//   model.SetCheckedArgIndices(expr, 3);
//   for (int i = 0; i < 3; ++i) {
//     Expr arg = call->args[i];
//     CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
//     CellRefs in_variables = model.AddInVariable(expr, arg);
//     CHECK(in_variables.size() == arg_variables.size());
//     for (size_t j = 0; j < in_variables.size(); ++j) {
//       model.AddEqual(in_variables[j], arg_variables[j], true);
//     }
//   }

//   CellRef y_var = model.GetInVariableRefs(expr, 0)[0]; // dy
//   CellRef i_var = model.GetInVariableRefs(expr, 1)[0]; // indices
//   CellRef l_var = model.GetInVariableRefs(expr, 2)[0]; // location
//   CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
//   CellRef s_var = model.GetConstantCellRef(0); // dim S
//   CellRef c_var = model.GetConstantCellRef(1); // dim C
//   CellRefs variables_to_partition = {y_var, i_var, l_var, out_variables[0]};
//   CellRefs partition_axis_vars = {c_var, s_var, s_var, s_var};
//   CellRef p_rule = -1;
//   for (int i = 0; i < 4; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], partition_axis_vars[i], false);
//     p_rule = model.AddAnd(p_rule, rule_var, false);
//   }
//   CellRef np_rule = -1;
//   for (int i = 0; i < 4; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], np_var, false);
//     np_rule = model.AddAnd(np_rule, rule_var, false);
//   }
//   model.AddOr(p_rule, np_rule, true);

//   Type out_type = TryGetLetVar(expr)->checked_type_;
//   model.AddDegenerateAxes(out_type, out_variables);
// };

// FConstraint MoeDecodeDxConstraint = [](CPModel& model, Expr expr) {
//   Call call = GetOpCall(expr, "raf.op.cuda.moe_decode_dx");
//   CellRefs out_variables = model.AddOutVariable(expr);
//   CHECK(out_variables.size() == 1);

//   model.SetCheckedArgIndices(expr, 5);
//   for (int i = 0; i < 5; ++i) {
//     Expr arg = call->args[i];
//     CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
//     CellRefs in_variables = model.AddInVariable(expr, arg);
//     CHECK(in_variables.size() == arg_variables.size());
//     for (size_t j = 0; j < in_variables.size(); ++j) {
//       model.AddEqual(in_variables[j], arg_variables[j], true);
//     }
//   }

//   CellRef y_var = model.GetInVariableRefs(expr, 0)[0]; // dy
//   CellRef d_var = model.GetInVariableRefs(expr, 1)[0]; // data
//   CellRef g_var = model.GetInVariableRefs(expr, 2)[0]; // gate
//   CellRef i_var = model.GetInVariableRefs(expr, 3)[0]; // indices
//   CellRef l_var = model.GetInVariableRefs(expr, 4)[0]; // location
//   CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
//   CellRef s_var = model.GetConstantCellRef(0); // dim S
//   CellRef c_var = model.GetConstantCellRef(1); // dim C
//   CellRefs variables_to_partition = {y_var, d_var, g_var, i_var, l_var, out_variables[0]};
//   CellRefs partition_axis_vars = {s_var, c_var, s_var, s_var, s_var, c_var};
//   CellRef p_rule = -1;
//   for (int i = 0; i < 6; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], partition_axis_vars[i], false);
//     p_rule = model.AddAnd(p_rule, rule_var, false);
//   }
//   CellRef np_rule = -1;
//   for (int i = 0; i < 6; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], np_var, false);
//     np_rule = model.AddAnd(np_rule, rule_var, false);
//   }
//   model.AddOr(p_rule, np_rule, true);

//   Type out_type = TryGetLetVar(expr)->checked_type_;
//   model.AddDegenerateAxes(out_type, out_variables);
// };

// FConstraint MoeEncodeDgConstraint = [](CPModel& model, Expr expr) {
//   Call call = GetOpCall(expr, "raf.op.cuda.moe_encode_dg");
//   CellRefs out_variables = model.AddOutVariable(expr);
//   CHECK(out_variables.size() == 1);

//   model.SetCheckedArgIndices(expr, 3);
//   for (int i = 0; i < 3; ++i) {
//     Expr arg = call->args[i];
//     CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
//     CellRefs in_variables = model.AddInVariable(expr, arg);
//     CHECK(in_variables.size() == arg_variables.size());
//     for (size_t j = 0; j < in_variables.size(); ++j) {
//       model.AddEqual(in_variables[j], arg_variables[j], true);
//     }
//   }

//   CellRef d_var = model.GetInVariableRefs(expr, 0)[0]; // dy
//   CellRef g_var = model.GetInVariableRefs(expr, 1)[0]; // gate
//   CellRef i_var = model.GetInVariableRefs(expr, 2)[0]; // indices
//   CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
//   CellRef s_var = model.GetConstantCellRef(0); // dim S
//   CellRef c_var = model.GetConstantCellRef(1); // dim C
//   CellRefs variables_to_partition = {d_var, g_var, i_var, out_variables[0]};
//   CellRefs partition_axis_vars = {s_var, s_var, s_var, s_var};
//   CellRef p_rule = -1;
//   for (int i = 0; i < 4; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], partition_axis_vars[i], false);
//     p_rule = model.AddAnd(p_rule, rule_var, false);
//   }
//   CellRef np_rule = -1;
//   for (int i = 0; i < 4; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], np_var, false);
//     np_rule = model.AddAnd(np_rule, rule_var, false);
//   }
//   model.AddOr(p_rule, np_rule, true);

//   Type out_type = TryGetLetVar(expr)->checked_type_;
//   model.AddDegenerateAxes(out_type, out_variables);
// };

// FConstraint MoeDecodeDgConstraint = [](CPModel& model, Expr expr) {
//   Call call = GetOpCall(expr, "raf.op.cuda.moe_decode_dg");
//   CellRefs out_variables = model.AddOutVariable(expr);
//   CHECK(out_variables.size() == 1);

//   model.SetCheckedArgIndices(expr, 4);
//   for (int i = 0; i < 4; ++i) {
//     Expr arg = call->args[i];
//     CellRefs arg_variables = model.GetOutVariableOrAddAsExternal(arg);
//     CellRefs in_variables = model.AddInVariable(expr, arg);
//     CHECK(in_variables.size() == arg_variables.size());
//     for (size_t j = 0; j < in_variables.size(); ++j) {
//       model.AddEqual(in_variables[j], arg_variables[j], true);
//     }
//   }

//   CellRef y_var = model.GetInVariableRefs(expr, 0)[0]; // dy
//   CellRef d_var = model.GetInVariableRefs(expr, 1)[0]; // data
//   CellRef i_var = model.GetInVariableRefs(expr, 2)[0]; // indices
//   CellRef l_var = model.GetInVariableRefs(expr, 3)[0]; // location
//   CellRef np_var = model.GetConstantCellRef(SpecialAxis::kNone);
//   CellRef s_var = model.GetConstantCellRef(0); // dim S
//   CellRef c_var = model.GetConstantCellRef(1); // dim C
//   CellRefs variables_to_partition = {y_var, d_var, i_var, l_var, out_variables[0]};
//   CellRefs partition_axis_vars = {s_var, c_var, s_var, s_var, c_var};
//   CellRef p_rule = -1;
//   for (int i = 0; i < 5; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], partition_axis_vars[i], false);
//     p_rule = model.AddAnd(p_rule, rule_var, false);
//   }
//   CellRef np_rule = -1;
//   for (int i = 0; i < 5; ++i) {
//     CellRef rule_var = model.AddEqual(variables_to_partition[i], np_var, false);
//     np_rule = model.AddAnd(np_rule, rule_var, false);
//   }
//   model.AddOr(p_rule, np_rule, true);

//   Type out_type = TryGetLetVar(expr)->checked_type_;
//   model.AddDegenerateAxes(out_type, out_variables);
// };

FConstraintMap op_constraints = {
  {"raf.op.tvm.add", CreateBroadCastConstraint("raf.op.tvm.add")}, // Var arg
  {"raf.op.tvm.subtract", CreateBroadCastConstraint("raf.op.tvm.subtract")},
  {"raf.op.tvm.multiply", CreateMatchingConstraint("raf.op.tvm.multiply", 2)},
  {"raf.op.tvm.divide", CreateMatchingConstraint("raf.op.tvm.divide", 2)},
  {"raf.op.tvm.cast", CreateMatchingConstraint("raf.op.tvm.cast", 1)},
  {"raf.op.tvm.squeeze", CreateMatchingConstraint("raf.op.tvm.squeeze", 1)},
  {"raf.op.tvm.scatter", CreateMatchingConstraint("raf.op.tvm.scatter", 3)},
  {"raf.op.tvm.layer_norm", CreateMatchingConstraint("raf.op.tvm.layer_norm", 1)},
  {"raf.op.cudnn.tanh_dx", CreateMatchingConstraint("raf.op.cudnn.tanh_dx", 3)}, // UnaryDx
  {"raf.op.tvm.tanh_dx", CreateMatchingConstraint("raf.op.tvm.tanh_dx", 3)},
  {"raf.op.tvm.relu_dx", CreateMatchingConstraint("raf.op.tvm.relu_dx", 3)},
  {"raf.op.cudnn.relu_dx", CreateMatchingConstraint("raf.op.cudnn.relu_dx", 3)},
  {"raf.op.tvm.gelu_dx", CreateMatchingConstraint("raf.op.tvm.gelu_dx", 3)},
  {"raf.op.tvm.softmax_dx", CreateMatchingConstraint("raf.op.tvm.softmax_dx", 3)}, // DeclareGeneralDx
  {"raf.op.tvm.power", CreateMatchingConstraint("raf.op.tvm.power", 2)},
  {"raf.op.tvm.transpose", CreateMatchingConstraint("raf.op.tvm.transpose", 1)},
  {"raf.op.tvm.transpose_dx", CreateMatchingConstraint("raf.op.tvm.transpose_dx", 1)},
  {"raf.op.tvm.one_hot", CreateMatchingConstraint("raf.op.tvm.one_hot", 3)},
  {"raf.op.tvm.expand_dims", CreateMatchingConstraint("raf.op.tvm.expand_dims", 1)},
  {"raf.op.tvm.embedding", CreateMatchingConstraint("raf.op.tvm.embedding", 2)},
  {"raf.op.cudnn.tanh", CreateMatchingConstraint("raf.op.cudnn.tanh", 1)},
  {"raf.op.cudnn.relu", CreateMatchingConstraint("raf.op.cudnn.relu", 1)},
  {"raf.op.tvm.relu", CreateMatchingConstraint("raf.op.tvm.relu", 1)},
  {"raf.op.tvm.gelu", CreateMatchingConstraint("raf.op.tvm.gelu", 1)},
  {"raf.op.nccl._allreduce", CreateMatchingConstraint("raf.op.nccl._allreduce", 1)},
  {"raf.op.nccl._reduce_scatter", CreateMatchingConstraint("raf.op.nccl._reduce_scatter", 1)},
  {"raf.op.cublas.batch_matmul", CreateBatchMatMulConstraint("raf.op.cublas.batch_matmul", false, false)},
  {"raf.op.cublas.batch_matmul_tn", CreateBatchMatMulConstraint("raf.op.cublas.batch_matmul_tn", true, false)},
  {"raf.op.cublas.batch_matmul_nt", CreateBatchMatMulConstraint("raf.op.cublas.batch_matmul_nt", false, true)},
  {"raf.op.cublas.batch_matmul_tt", CreateBatchMatMulConstraint("raf.op.cublas.batch_matmul_tt", true, true)},
  {"raf.op.tvm.cumsum", CreateScanConstraint("raf.op.tvm.cumsum")},
  // {"raf.op.tvm.prod", CreateScanConstraint("raf.op.tvm.prod")},
  {"raf.op.cublas.matmul", CreateMatMulConstraint("raf.op.cublas.matmul", false, false)},
  {"raf.op.cublas.matmul_tn", CreateMatMulConstraint("raf.op.cublas.matmul_tn", true, false)},
  {"raf.op.cublas.matmul_nt", CreateMatMulConstraint("raf.op.cublas.matmul_nt", false, true)},
  {"raf.op.cublas.matmul_tt", CreateMatMulConstraint("raf.op.cublas.matmul_tt", true, true)},
  {"raf.op.tvm.sum", CreateReduceConstraint("raf.op.tvm.sum")},
  {"raf.op.tvm.prod", CreateReduceConstraint("raf.op.tvm.prod")},
  {"raf.op.cudnn.softmax", SoftmaxConstraint},
  {"raf.op.tvm.softmax", SoftmaxConstraint},
  {"raf.op.tvm.layer_norm", LayerNormConstraint},
  {"raf.op.tvm.where", WhereConstraint},
  {"raf.op.tvm.split", SplitConstraint},
  {"raf.op.tvm.bias_add", BiasAddConstraint},
  {"raf.op.tvm.concatenate", ConcatConstraint},
  {"raf.op.tvm.argmax", ArgmaxConstraint},
  {"raf.op.tvm.layer_norm_dx", LayerNormDxConstraint},
  {"raf.op.tvm.reshape", ReshapeConstraint},
  {"raf.op.tvm.softmax_dx", SoftmaxDxConstraint},
  // {"raf.op.tvm.topk", TopKConstraint},
  {"raf.op.nccl._allgather", AllgatherConstraint},
  {"raf.op.nccl._all_to_all", AllToAllConstraint},
  {"raf.op.cuda.moe_encode", MoeEncodeConstraint},
  {"raf.op.cuda.moe_encode_batch_prioritized", MoeEncodeBPRConstraint},
  {"raf.op.cuda.moe_decode", MoeDecodeConstraint},
  // {"raf.op.cuda.moe_encode_dx", MoeEncodeDxConstraint},
  // {"raf.op.cuda.moe_decode_dx", MoeDecodeDxConstraint},
  // {"raf.op.cuda.moe_encode_dg", MoeEncodeDgConstraint},
  // {"raf.op.cuda.moe_decode_dg", MoeDecodeDgConstraint},
};

bool HasConstraint(const std::string& op) {
  if (op_constraints.count(op)) {
    return true;
  } else if (base_dialect_op.count(op)) {
    return HasConstraint(base_dialect_op.at(op));
  } else {
    return false;
  }
}

FConstraint GetConstraint(const std::string& op) {
  if (op_constraints.count(op)) {
    return op_constraints.at(op);
  } else if (base_dialect_op.count(op)) {
    return GetConstraint(base_dialect_op.at(op));
  } else {
    return CallConstraint;
  }
}

// CPModelBuilder

CPModelBuilder::CPModelBuilder(const ScheduledDFG& sched_dfg, int dp_group_size) :
  sched_dfg_(sched_dfg), dp_group_size_(dp_group_size), model_(sched_dfg, dp_group_size), model_valid_(true) {}

CPModel CPModelBuilder::Build(Exprs& exprs) {
  // Build a series of operations for the solver from expressions in topological order.
  Setup_(exprs);
  for (auto expr : exprs) {
    stack_.back() = expr;
    VisitExpr(expr);
  }
  return model_;
}

bool CPModelBuilder::ModelValid() const {
  return model_valid_;
}

void CPModelBuilder::VisitExpr_(const LetNode* let) {
  if (!model_valid_) {
    return;
  }
  VisitExpr(let->value);
}

void CPModelBuilder::VisitExpr_(const TupleNode* tuple) {
  if (!model_valid_) {
    return;
  }
  TupleConstraint(model_, stack_.back());
  AddTarget_();
}

void CPModelBuilder::VisitExpr_(const TupleGetItemNode* tgi) {
  if (!model_valid_) {
    return;
  }
  TGIConstraint(model_, stack_.back());
  AddTarget_();
}

void CPModelBuilder::VisitExpr_(const CallNode* call) {
  if (!model_valid_) {
    return;
  }
  VisitExpr(call->op);
}

void CPModelBuilder::VisitExpr_(const OpNode* op) {
  static std::unordered_set<std::string> warning_printed;
  if (!HasConstraint(op->name)) {
    model_valid_ = false;
    if (!warning_printed.count(op->name)) {
      LOG(WARNING) << "Missing constraint for op " << op->name << ".";
      warning_printed.insert(op->name);
    }
    return;
  }
  FConstraint op_constraint = GetConstraint(op->name);
  op_constraint(model_, stack_.back());
  AddTarget_();
}

void CPModelBuilder::VisitExpr_(const FunctionNode* func) {
  Expr caller = stack_.back();
  Exprs func_exprs = UnfoldAndUpdateFunction(caller, func);
  model_.SetFuncExprs(caller, func_exprs);
  stack_.push_back(Expr{nullptr});
  for (auto expr : func_exprs) {
    stack_.back() = expr;
    VisitExpr(expr);
  }
  stack_.pop_back();
  if (!model_valid_) {
    return;
  }
  GetReturnValue_();
}

void CPModelBuilder::VisitExpr(const Expr& expr) {
  if (!model_valid_) {
    return;
  }
  // Always allow visiting op and function node.
  if (expr->IsInstance<FunctionNode>() || expr->IsInstance<OpNode>()) {
    ExprFunctor<void(const Expr& e)>::VisitExpr(expr);
  } else {
    ExprVisitor::VisitExpr(expr);
  }
}

void CPModelBuilder::Setup_(Exprs& exprs) {
  stack_.clear();
  stack_.push_back(Expr{nullptr});
  // if (!ContainsAllToAll(exprs)) {
  //   model_.SetThreshold(2);
  // }
}

void CPModelBuilder::GetReturnValue_() {
  Expr caller = stack_.back();
  auto args = Downcast<Call>(caller)->args;
  Exprs func_exprs = model_.GetFuncExprs(caller);
  Expr ret = func_exprs.back();
  Indices out_variables = model_.GetOutVariableCellRefs(ret);
  auto arg_consumer_index = ConsumptionInfoCollector(args).Collect(func_exprs);
  std::vector<Indices> in_indices_vec = {};
  for (auto arg : args) {
    // Try get non-trivial input indices of consumers as the input indices of this arg.
    // If all consumers' input indices are {}, then arg's input indices would also be {}.
    // Getting in indices of args is only for knowing its corresponding input variables.
    // Targets have already been added when visiting func exprs.
    Indices var_indices = {-1};
    for (auto it : arg_consumer_index.at(arg)) {
      if (model_.ArgIsChecked(it.first, it.second)) {
        var_indices = model_.GetInVariableRefs(it.first, it.second);
        break;
      }
    }
    in_indices_vec.push_back(var_indices);
  }
  InOutCellRefs inout_cellrefs = std::make_pair(in_indices_vec, out_variables);
  model_.SetInOutVariableCellRefs(stack_.back(), inout_cellrefs);
  model_.SetConsumerIndex(caller, arg_consumer_index);
}

void CPModelBuilder::AddTarget_() {
  InOutCellRefs inout_indices = model_.GetInOutVariableCellRefs(stack_.back());
  model_.AddMaximizeTarget(inout_indices);
}

void CPModelBuilder::PrintStack_() {
  // For debug.
  LOG(INFO) << "Calling stack:";
  for (int i = 0; i < stack_.size(); ++i) {
    auto in_indices_vec = model_.GetInVariableCellRefsVec(stack_[i]);
    auto out_variables = model_.GetOutVariableCellRefs(stack_[i]);
    LOG(INFO) << "Depth " << i << ", expr " << stack_[i] << ", in indices " <<
      PrintIndicesVec(in_indices_vec) << ", out indices " << PrintIndices(out_variables);
  }
}

// Util functions

std::vector<Indices> GetInAxesVec(const ExprMap<InOutCellRefs>& expr_inout_indices,
                                  const IndexMap<Index>& index_axis, Expr expr) {
  std::vector<Indices> axes_vec = {};
  CHECK(expr_inout_indices.count(expr));
  for (auto in_indices : expr_inout_indices.at(expr).first) {
    axes_vec.push_back(Indices{});
    for (auto index : in_indices) {
      if (index == -1) {
        axes_vec.back().push_back(SpecialAxis::kNone);  
      } else {
        axes_vec.back().push_back(index_axis.at(index));
      }
    }
  }
  return axes_vec;
}

Indices GetOutAxes(const ExprMap<InOutCellRefs>& expr_inout_indices,
                   const IndexMap<Index>& index_axis, Expr expr) {
  Indices axes = {};
  CHECK(expr_inout_indices.count(expr));
  for (auto index : expr_inout_indices.at(expr).second) {
    if (index == -1) {
      axes.push_back(SpecialAxis::kNone);
    } else {
      axes.push_back(index_axis.at(index));
    }
  }
  return axes;
}

ExprMap<InOutAxes> ParseSolution(const CPModel& model,
                                    const CellRefMap<Index>& solution) {
  auto expr_inout_cellrefs = model.GetInOutCellRefs();
  ExprMap<InOutAxes> expr_inout_axes = {};
  for (auto& it : expr_inout_cellrefs) {
    auto in_axes_vec = GetInAxesVec(expr_inout_cellrefs, solution, it.first);
    auto out_axes = GetOutAxes(expr_inout_cellrefs, solution, it.first);
    expr_inout_axes[it.first] = std::make_pair(in_axes_vec, out_axes);
  }
  return expr_inout_axes;
}

CellRefMap<Index> ParseSolutionAxesFromStr(std::string solution) {
  CellRefMap<Index> index_axis = {};
  std::stringstream ss(solution);
  std::string line;
  while (std::getline(ss, line)) {
    std::stringstream ss_(line);
    Index index, axis;
    ss_ >> index >> axis;
    index_axis[index] = axis;
  }
  return index_axis;
}

ExprMap<InOutAxes> ParseSolution(const CPModel& model,
                                    std::string solution) {
  auto index_axis = ParseSolutionAxesFromStr(solution);
  return ParseSolution(model, index_axis);
}

Indices GetProducers(ExprMap<Index>& var_index, Expr expr) {
  Indices producers = {};
  auto expr_ = TryGetLetValue(expr);
  if (expr_->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr_);
    for (auto arg : call->args) {
      if (var_index.count(arg)) {
        producers.push_back(var_index.at(arg));
      }
    }
  } else if (expr_->IsInstance<TupleNode>()) {
    Tuple tuple = Downcast<Tuple>(expr_);
    for (auto field : tuple->fields) {
      if (var_index.count(field)) {
        producers.push_back(var_index.at(field));
      }
    }
  } else if (expr_->IsInstance<TupleGetItemNode>()) {
    TupleGetItem tgi = Downcast<TupleGetItem>(expr_);
    auto tuple = tgi->tuple;
    if (var_index.count(tuple)) {
      producers.push_back(var_index.at(tuple));
    }
  } else {
    LOG(WARNING) << "Not supported expr: " << expr->GetTypeKey();
  }
  return producers;
}

// CPSolution

CPSolution::CPSolution(const CPModel& model,
                       const ExprMap<InOutCellRefs>& expr_inout_axes,
                       const Exprs& exprs) :
    expr_inout_axes(expr_inout_axes),
    expr_func_exprs(model.GetFuncExprs()),
    expr_arg_idx_to_inoutaxes_index(model.GetCheckedArgIndiceMap()),
    expr_arg_consumer_index(model.GetConsumerIndex()),
    exprs(exprs) {}

CPSolution::CPSolution() :
    expr_inout_axes({}),
    expr_func_exprs({}),
    expr_arg_idx_to_inoutaxes_index({}),
    expr_arg_consumer_index({}),
    exprs({}) {}

static bool debug_print_ = false;
  
void DebugPrintSolvedAxes(CPModel& model, const CPSolution& solution, const Exprs& exprs) {
  if(debug_print_) {
    auto expr_inout_indices = model.GetInOutCellRefs();
    auto expr_inout_axes = solution.expr_inout_axes;
    for (auto expr_ : exprs) {
      CHECK(expr_inout_axes.count(expr_)) << "Expr " << expr_ << " not in expr_inout_axes.";
      auto inout_axes = expr_inout_axes.at(expr_);
      LOG(INFO) << "[PrintSolvedAxes]: " << expr_ << " has input axes: " << PrintIndicesVec(inout_axes.first);
      LOG(INFO) << "[PrintSolvedAxes]: " << expr_ << " has output axes: " << PrintIndices(inout_axes.second);
    }
  }
}

CPSolution SolvePartitionAxes(const ScheduledDFG& sched_dfg, Exprs& exprs, int dp_group_size) {
  if(debug_print_) {
    LOG(INFO) << " ";
    LOG(INFO) << "------------------------------";
    LOG(INFO) << "\tSolve partition axes for:";
    for(auto expr: exprs) {
      LOG(INFO) << "\t\t -> " << expr;
    }
    LOG(INFO) << "------------------------------";
    LOG(INFO) << " ";
  }
  try {
    auto mb_start = std::chrono::system_clock::now();
    auto cp_model_builder = CPModelBuilder(sched_dfg, dp_group_size);
    CPModel model = cp_model_builder.Build(exprs);
    if (!cp_model_builder.ModelValid()) {
      return CPSolution();
    }
    auto mb_end = std::chrono::system_clock::now();
    auto mb_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(mb_end - mb_start).count();
    // LOG(INFO) << "CPModelBuilder took " << mb_elapsed_time / 1000.0 << " ms.";
    auto scp_start = std::chrono::system_clock::now();
    auto solution = model.GetSolution();
    auto scp_end = std::chrono::system_clock::now();
    auto scp_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(scp_end - scp_start).count();
    // LOG(INFO) << "SolveConstraintOpt took " << scp_elapsed_time / 1000.0 << " ms.";
    if (solution.empty()) {
      return CPSolution();
    }
    auto expr_inout_axes = ParseSolution(model, solution);
    auto cp_solution = CPSolution(model, expr_inout_axes, exprs);
    DebugPrintSolvedAxes(model, cp_solution, exprs);
    return CPSolution(model, expr_inout_axes, exprs);
  } catch (std::exception& e) {
    LOG(INFO) << " ";
    LOG(INFO) << "------------------------------";
    LOG(INFO) << "\tSolve partition axes for:";
    for(auto expr: exprs) {
      LOG(INFO) << "\t\t -> " << expr;
    }
    LOG(INFO) << "------------------------------";
    LOG(INFO) << " ";
    LOG(FATAL) << "Failed to build CPModel: " << e.what();
    return CPSolution();
  }
}

CPSolution SolvePartitionAxes(const ScheduledDFG& sched_dfg, const Nodes& nodes, int dp_group_size) {
  Exprs exprs = {};
  auto topo_order = GetTopologicalOrder(sched_dfg.dfg, nodes);
  for (auto node : topo_order) {
    exprs.push_back(sched_dfg.dfg.getExprFromNode(node));
  }
  return SolvePartitionAxes(sched_dfg, exprs, dp_group_size);
}

CPSolution SolvePartitionAxes(const ScheduledDFG& sched_dfg, const NodeSet& nodes, int dp_group_size) {
  Nodes nodes_ = {nodes.begin(), nodes.end()};
  return SolvePartitionAxes(sched_dfg, nodes_, dp_group_size);
}

class FunctionChecker : ExprVisitor {
public:
  FunctionChecker(std::unordered_set<std::string>& invalid_op_cache) : cache_(invalid_op_cache) {}

  bool Check(Expr expr) {
    this->VisitExpr(expr);
    return check_passed_;
  }

  void VisitExpr_(const OpNode* op) {
    if(cache_.count(op->name)) {
      check_passed_ = false;
    } else {
      if(!HasConstraint(op->name) || !HasFPartition(op->name)) {
        LOG(WARNING) << "Op " << op->name << " 's partition rule or constraint is not registered, ignoring.";
        cache_.insert(op->name);
        check_passed_ = false;
      }
    }
  }

private:
  std::unordered_set<std::string>& cache_;
  bool check_passed_ = true;
};

bool IsOpSupported(const ScheduledDFG& sched_dfg, const Node* node) {
  static std::unordered_set<std::string> unpartitionable_op_set_ = {};
  auto expr = sched_dfg.dfg.getExprFromNode(node);
  CHECK(expr.defined()) << "Unable to get expr from node when checking IsOpSupported.";
  if(auto call_node = expr.as<CallNode>()) {
    if(auto call_op = call_node->op.as<OpNode>()) {
      if(unpartitionable_op_set_.count(call_op->name)) {
        return false;
      } else {
        if(!HasConstraint(call_op->name) || !HasFPartition(call_op->name)) {
          LOG(WARNING) << "Op " << call_op->name << "'s partition rule or constraint is not registered, ignoring.";
          unpartitionable_op_set_.insert(call_op->name);
          return false;
        }
      }
    } else if (auto func_node = call_node->op.as<FunctionNode>()) {
      if(!FunctionChecker(unpartitionable_op_set_).Check(func_node->body)) {
        return false;
      }
    }
  }
  return true;
}

bool IsAllOpsSupported(const ScheduledDFG& sched_dfg, const Nodes& nodes) {
  if (nodes.empty()) {
    return false;
  }
  for(auto node: nodes) {
    if(!IsOpSupported(sched_dfg, node)) {
      return false;
    }
  }
  return true;
}

bool IsAllPartitionable(const CPSolution& solution, const Exprs& exprs) {
  if (solution.exprs.empty()) {
    return false;
  }
  for (auto expr : exprs) {
    expr = TryGetLetValue(expr);
    CHECK(solution.expr_inout_axes.count(expr)) << "Expr " << expr << " not in solution.expr_inout_axes.";
    Indices out_axes = solution.expr_inout_axes.at(expr).second;
    if (AllEqual(out_axes, SpecialAxis::kNone)) {
      return false;
    }
    if (solution.expr_func_exprs.count(expr)) {
      // When expr is an function, check whether all expr in its body is partitioned.
      const Exprs& func_exprs = solution.expr_func_exprs.at(expr);
      if (!IsAllPartitionable(solution, func_exprs)) {
        return false;
      }
    }
  }
  return true;
}

bool IsAllPartitionable(const CPSolution& solution) {
  return IsAllPartitionable(solution, solution.exprs);
}

void CheckAllPartitionable(const CPSolution& solution, const Exprs& exprs) {
  for (auto expr : exprs) {
    expr = TryGetLetValue(expr);
    CHECK(solution.expr_inout_axes.count(expr)) << "Expr " << expr << " not in solution.expr_inout_axes.";
    Indices out_axes = solution.expr_inout_axes.at(expr).second;
    if (AllEqual(out_axes, SpecialAxis::kNone)) {
      LOG(FATAL) << "Found not partitionable expr: " << expr;
    }
    if (solution.expr_func_exprs.count(expr)) {
      // When expr is an function, check whether all expr in its body is partitioned.
      const Exprs& func_exprs = solution.expr_func_exprs.at(expr);
      try {
        CheckAllPartitionable(solution, func_exprs);
      } catch (...) {
        LOG(FATAL) << "Found not partitionable func: " << expr;
      }
    }
  }
}

void CheckAllPartitionable(const CPSolution& solution) {
  return CheckAllPartitionable(solution, solution.exprs);
}

std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const Nodes& nodes, int dp_group_size) {
  if(!IsAllOpsSupported(sched_dfg, nodes)) {
    return std::make_pair(false, CPSolution());
  }
  CPSolution solution = SolvePartitionAxes(sched_dfg, nodes, dp_group_size);
  auto result = IsAllPartitionable(solution);
  // LOG(INFO) << "Result: " << result << "-----------------------------------------";
  // auto end = std::chrono::system_clock::now();
  // auto elapsed_time = 
  //   std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  // LOG(INFO) << "Check is all partitionable takes " << elapsed_time / 1000.0f << " ms.";
  return std::make_pair(result, solution);
}

std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const NodeSet& nodes, int dp_group_size) {
  Nodes nodes_ = {nodes.begin(), nodes.end()};
  return IsAllPartitionable(sched_dfg, nodes_, dp_group_size);
}

void AddSolvedAxes(CPModel& model, const Indices& indices, const Indices& axes) {
  CHECK(indices.size() == axes.size());
  for (int i = 0; i < indices.size(); ++i) {
    auto axis_index = model.GetConstantCellRef(axes[i]);
    Index solved_rule_idx = model.AddEqual(indices[i], axis_index, false);
    auto not_partition_index = model.GetConstantCellRef(SpecialAxis::kNone);
    Index np_rule_idx = model.AddEqual(indices[i], not_partition_index, false);
    model.AddOr(solved_rule_idx, np_rule_idx, true);
  }
}

void AddSolvedAxesVec(CPModel& model, const std::vector<Indices>& indices_vec, 
                      const std::vector<Indices>& axes_vec) {
  CHECK_EQ(indices_vec.size(), axes_vec.size());
  for (int i = 0; i < indices_vec.size(); ++i) {
    AddSolvedAxes(model, indices_vec[i], axes_vec[i]);
  }
}

static const std::string node_name_to_print_ = "dontprint";
// static const std::string node_name_to_print_ = "tvm.where";
// static const std::string node_name_to_print_ = "batch_matmul";
// static const std::string node_name_to_print_ = "fn (%p0: Tensor[(32, 128, 768), float32]";

void PrintSolvedAxes(CPModel& model, const CPSolution& solution, const Exprs& exprs, Expr expr, const ScheduledDFG& sched_dfg, const Node* node) {
  if(debug_print_ || sched_dfg.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
    auto expr_inout_indices = model.GetInOutCellRefs();
    auto expr_inout_axes = solution.expr_inout_axes;
    for (auto expr_ : exprs) {
      if (expr_ == expr) {
        continue;
      }
      CHECK(expr_inout_axes.count(expr_)) << "Expr " << expr_ << " not in expr_inout_axes.";
      auto inout_axes = expr_inout_axes.at(expr_);
      LOG(INFO) << "[PrintSolvedAxes]: " << expr_ << " has input axes: " << PrintIndicesVec(inout_axes.first);
      LOG(INFO) << "[PrintSolvedAxes]: " << expr_ << " has output axes: " << PrintIndices(inout_axes.second);
    }
  }
}

void AddSolvedAxes(CPModel& model, const CPSolution& solution, const Exprs& exprs, Expr expr) {
  auto expr_inout_indices = model.GetInOutCellRefs();
  auto expr_inout_axes = solution.expr_inout_axes;
  for (auto expr_ : exprs) {
    if (expr_ == expr) {
      continue;
    }
    CHECK(expr_inout_indices.count(expr_)) << "Expr " << expr_ << " not in expr_inout_indices.";
    auto inout_indices = expr_inout_indices.at(expr_);
    CHECK(expr_inout_axes.count(expr_)) << "Expr " << expr_ << " not in expr_inout_axes.";
    auto inout_axes = expr_inout_axes.at(expr_);
    AddSolvedAxesVec(model, inout_indices.first, inout_axes.first);
    AddSolvedAxes(model, inout_indices.second, inout_axes.second);
  }
}

void MergeCPSolution(const ExtendedDFG& dfg, CPSolution& result, const CPSolution& reference, Expr expr) {
  CHECK(reference.expr_inout_axes.count(expr)) << "Cannot find expr " << expr << "'s axis in reference CPSolution.";
  result.expr_inout_axes[expr] = reference.expr_inout_axes.at(expr);
  if(reference.expr_arg_idx_to_inoutaxes_index.count(expr)) {
    // function call do not have expr_arg_idx_to_inoutaxes_index set
    result.expr_arg_idx_to_inoutaxes_index[expr] = reference.expr_arg_idx_to_inoutaxes_index.at(expr);
  }
  if (reference.expr_func_exprs.count(expr)) {
    result.expr_func_exprs[expr] = reference.expr_func_exprs.at(expr);
    // if this is a function call, we also need to merge expr_inout_axes and expr_arg_idx_to_inoutaxes_index for function exprs
    for(auto expr_: reference.expr_func_exprs.at(expr)) {
      expr_ = TryGetLetValue(expr_);
      CHECK(reference.expr_inout_axes.count(expr_)) << "Cannot find function exprs " << expr_ << "'s axis in reference CPSolution.";
      if(reference.expr_inout_axes.count(expr_)) {
        result.expr_inout_axes[expr_] = reference.expr_inout_axes.at(expr_);
      }
      if(reference.expr_arg_idx_to_inoutaxes_index.count(expr_)) {
        result.expr_arg_idx_to_inoutaxes_index[expr_] = reference.expr_arg_idx_to_inoutaxes_index.at(expr_);
      }
    }
  }
  if (reference.expr_arg_consumer_index.count(expr)) {
    result.expr_arg_consumer_index[expr] = reference.expr_arg_consumer_index.at(expr);
  }
  NodeSet result_node_set;
  for(auto expr_: result.exprs) {
    auto node = dfg.getNodeFromExpr(expr_);
    CHECK(node) << "Unable to get node for expr " << expr_ << " in MergeCPSolution.";
    result_node_set.insert(node);
  }
  result_node_set.insert(dfg.getNodeFromExpr(expr));
  Exprs result_exprs;
  for(auto node: GetTopologicalOrder(dfg, result_node_set)) {
    auto expr = dfg.getExprFromNode(node);
    CHECK(expr.defined()) << "Unable to get expr for node " << dfg.getNodeNameOrDefault(node) << " in MergeCPSolution.";
    result_exprs.push_back(expr);
  }
  result.exprs = result_exprs;
}

std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const CPSolution& cp_solution,
                                               const Nodes& nodes, const Node* node, int dp_group_size) {
  if(!IsOpSupported(sched_dfg, node) || !IsAllOpsSupported(sched_dfg, nodes)) {
    return std::make_pair(false, cp_solution);
  }
  if(nodes.size() < 2) {
    // nodes do not have a valid axes yet, directly solve
    Nodes tmp_nodes = nodes;
    tmp_nodes.push_back(node);
    return IsAllPartitionable(sched_dfg, tmp_nodes, dp_group_size);
  }
  Nodes nodes_ = {node};
  auto parents = sched_dfg.dfg.getNonSinkParents(node);
  auto children = sched_dfg.dfg.getNonSourceChildren(node);
  for (auto node_ : nodes) {
    if(parents.count(node_)) {
      if(debug_print_ || sched_dfg.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
        LOG(INFO) << "[IsAllPartitionable]: Adding parent node: " << sched_dfg.dfg.getNodeNameOrDefault(node_);
      }
    }
    if(children.count(node_)) {
      if(debug_print_ || sched_dfg.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
        LOG(INFO) << "[IsAllPartitionable]: Adding children node: " << sched_dfg.dfg.getNodeNameOrDefault(node_);
      }
    }
    if (parents.count(node_) || children.count(node_)) {
      nodes_.push_back(node_);
    }
  }
  Exprs exprs = {};
  for (auto node_ : GetTopologicalOrder(sched_dfg.dfg, nodes_)) {
    exprs.push_back(sched_dfg.dfg.getExprFromNode(node_));
  }
  CPModel model = CPModelBuilder(sched_dfg, dp_group_size).Build(exprs);
  PrintSolvedAxes(model, cp_solution, exprs, sched_dfg.dfg.getExprFromNode(node), sched_dfg, node);
  AddSolvedAxes(model, cp_solution, exprs, sched_dfg.dfg.getExprFromNode(node));
  // const auto* solve_partition_axes =
  //   tvm::runtime::Registry::Get("raf.distributed.solve_partition_axes");
  // CHECK(solve_partition_axes);
  // std::string solution_ = (*solve_partition_axes)(operations);
  // std::string solution_ = SolveConstraintOpt(operations);
  auto solution_ = model.GetSolution();
  auto expr_inout_axes = ParseSolution(model, solution_);
  auto cp_solution_ = CPSolution(model, expr_inout_axes, exprs);
  if (!IsAllPartitionable(cp_solution_)) {
    if(debug_print_) {
      LOG(INFO) << "Incremental solve failed...";
    }
    nodes_ = nodes;
    nodes_.push_back(node);
    return IsAllPartitionable(sched_dfg, nodes_, dp_group_size);
  }
  auto final_solution = cp_solution;
  if(debug_print_) {
    LOG(INFO) << "Incremental solve success.";
  }
  DebugPrintSolvedAxes(model, cp_solution_, exprs);
  auto expr = sched_dfg.dfg.getExprFromNode(node);
  MergeCPSolution(sched_dfg.dfg, final_solution, cp_solution_, expr);
  return std::make_pair(true, final_solution);
}

std::pair<bool, CPSolution> IsAllPartitionable(const ScheduledDFG& sched_dfg, const CPSolution& solution,
                        const NodeSet& nodes, const Node* node, int dp_group_size) {
  Nodes nodes_ = {nodes.begin(), nodes.end()};
  return IsAllPartitionable(sched_dfg, solution, nodes_, node, dp_group_size);
}

std::vector<int> ExpandDimsOfType(Type type, Indices axes) {
  if (type->IsInstance<TensorTypeNode>()) {
    auto tt = Downcast<TensorType>(type);
    if(axes[0] == SpecialAxis::kNone || axes[0] == SpecialAxis::kExpert || axes[0] == SpecialAxis::kIndiceAndLocations) {
      return {};
    }
    CHECK_GE(axes[0], 0);
    CHECK_LT(axes[0], tt->shape.size());
    return {static_cast<int>(*as_const_int(tt->shape[axes[0]]))};
  } else {
    auto tt = Downcast<TupleType>(type);
    std::vector<int> result;
    CHECK_EQ(tt->fields.size(), axes.size());
    for (int i = 0; i < axes.size(); ++i) {
      auto field_dims = ExpandDimsOfType(tt->fields[i], Indices{axes[i]});
      result.insert(result.end(), field_dims.begin(), field_dims.end());
    }
    return result;
  }
}

std::unordered_set<int> GetPartitionDims(const CPSolution& solution, const Exprs& exprs) {
  std::unordered_set<int> dims_set;
  for (auto expr : exprs) {
    expr = TryGetLetValue(expr);
    CHECK(solution.expr_inout_axes.count(expr)) << "Expr " << expr << " not in solution.expr_inout_axes.";
    if(solution.expr_arg_idx_to_inoutaxes_index.count(expr)) {
      auto expr_args = GetArgs<Exprs>(expr);
      for(auto arg_indices_it: solution.expr_arg_idx_to_inoutaxes_index.at(expr)) {
        auto arg_idx = arg_indices_it.first;
        Index var_index = arg_indices_it.second;
        CHECK(solution.expr_inout_axes.at(expr).first.size() > var_index);
        auto in_axes = solution.expr_inout_axes.at(expr).first.at(var_index);
        CHECK(expr_args.size() > arg_idx);
        auto arg = expr_args[arg_idx];
        auto arg_dims = ExpandDimsOfType(arg->checked_type(), in_axes);
        dims_set.insert(arg_dims.begin(), arg_dims.end());
      }
    }

    Indices out_axes = solution.expr_inout_axes.at(expr).second;
    auto out_dims = ExpandDimsOfType(expr->checked_type(), out_axes);
    dims_set.insert(out_dims.begin(), out_dims.end());
    if (solution.expr_func_exprs.count(expr)) {
      // When expr is an function, check whether all expr in its body is partitioned.
      const Exprs& func_exprs = solution.expr_func_exprs.at(expr);
      auto function_dims = GetPartitionDims(solution, func_exprs);
      dims_set.insert(function_dims.begin(), function_dims.end());
    }
  }
  return dims_set;
}

std::vector<int> GetPartitionParts(const ScheduledDFG& sched_dfg, const Nodes& nodes, const CPSolution& solution, int max_partition, int dp_group_size) {
  CHECK(IsAllPartitionable(solution)) << "Solution is not all partitionable.";
  auto partitioned_dims = GetPartitionDims(solution, solution.exprs);
  int min_dim = std::numeric_limits<int>::max();
  for(auto dim: partitioned_dims) {
    min_dim = std::min(min_dim, dim);
  }
  ExprSet exprs = {};
  for (auto node : nodes) {
    auto expr = sched_dfg.dfg.getExprFromNode(node);
    exprs.insert(expr);
    // all2all would not in funcs.
    // if (solution.expr_func_exprs_.count(expr)) {
    //   auto& func_exprs = solution.expr_func_exprs_.at(expr);
    //   exprs.insert(func_exprs.begin(), func_exprs.end());
    // }
  }
  // LOG(INFO) << "Min dim in GetPartitionParts: " << min_dim;
  // if (ContainsAllToAll(exprs)) {
  //   min_dim = min_dim / dp_group_size;
  // }
  std::vector<int> n_parts;
  for (int i = 2; i < min_dim / 2 + 1 && i <= max_partition; ++i) {
    bool all_partitionable = true;
    for(auto dim: partitioned_dims) {
      if (dim % i != 0) {
        all_partitionable = false;
        break;
      }
    }
    if(all_partitionable) {
      n_parts.push_back(i);
    }
  }
  return n_parts;
}

std::vector<int> GetPartitionParts(const ScheduledDFG& sched_dfg, const NodeSet& nodes, const CPSolution& solution, int max_partition, int dp_group_size) {
  Nodes nodes_ = {nodes.begin(), nodes.end()};
  return GetPartitionParts(sched_dfg, nodes_, solution, max_partition, dp_group_size);
}


} // namespace solve_partition_axes
} // namespace raf
} // namespace pass