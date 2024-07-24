/*!
 * Copyright (c) 2022 by Contributors
 * \file partition_common.h
 * \brief Define data structures and provide utilities for partitioning.
 */
#pragma once
#include <unordered_map>
#include <unordered_set>
#include "raf/pass.h"
#include "../common.h"
#include "../let_list.h"
#include "./profile_utils.h"
#include "./scheduler_utils.h"

namespace raf {
namespace pass {
namespace partition_common {

using namespace tvm::relay;
using namespace raf::analysis::dependency_graph;
using namespace raf::pass;
using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::pass::scheduler_utils;
using namespace raf::pass::profile_utils;
using raf::pass::ToGraphNormalForm;
using LetList = raf::pass::LetList;

using tvm::tir::as_const_int;
using Bools = std::vector<bool>;
using Index = int;
using Indices = std::vector<Index>;
using IndexSet = std::unordered_set<Index>;
// A reference to a cell in the model
// (physical meaning: index of a cell in the cells array)
using CellRef = Index;
using CellRefs = std::vector<CellRef>;
template<class T>
using CellRefMap = std::unordered_map<CellRef, T>;
using InOutCellRefs = std::pair<std::vector<CellRefs>, CellRefs>;

using InOutAxes = std::pair<std::vector<Indices>, Indices>;
template<class T>
using IndexMap = std::unordered_map<Index, T>;
template <class T>
using ConstantMap = std::unordered_map<int, T>;
using HashIndex = size_t;
template<class T>
using HashIndexMap = std::unordered_map<HashIndex, T>;

// For creating matching constrint for some var arg ops, e.g., tvm.add.
// Returns argument indices according to the input expr.
using ArgIndicesGetter = std::function<Indices(const Expr&)>;

namespace SpecialAxis {
  const int kNone = -4;
  const int kExpert = -3;
  const int kIndiceAndLocations = -2;
}  // namespace SpecialAxis

inline std::string PrintIndices(const Indices& indices) {
  std::string result = "[";
  for (const auto& index : indices) {
    result += std::to_string(index);
    result += ", ";
  }
  if (result.size() > 1) {
    result.pop_back();
    result.pop_back();
  }
  result += "]";
  return result;
}

inline std::string PrintIndicesVec(const std::vector<Indices>& indices_vec) {
  std::string result = "[";
  for (const auto& indices : indices_vec) {
    result += PrintIndices(indices);
    result += ", ";
  }
  if (result.size() > 1) {
    result.pop_back();
    result.pop_back();
  }
  result += "]";
  return result;
}

inline std::string PrintInOutIndices(const InOutAxes& in_out_indices) {
  std::string result = "In: ";
  result += PrintIndicesVec(in_out_indices.first);
  result += ", Out: " + PrintIndices(in_out_indices.second);
  return result;
}


inline std::string PrintSubgraph(const ExtendedDFG& dfg, const NodeSet& subgraph_nodes, const NodeSet& node_overlap_idle) {
  auto topo_order = GetTopologicalOrder(dfg);
  std::string result = "[";
  for (auto node : topo_order) {
    if(subgraph_nodes.count(node)) {
      std::string name = dfg.getNodeNameOrDefault(node);
      result += name + "-" + std::to_string((int)dfg.getNodeExecTime(node)) + "-" + 
        std::to_string(node_overlap_idle.count(node)) + ", ";
    }
  }
  if (result.size() > 1) {
    result.pop_back();
    result.pop_back();
  }
  result += "]";
  return result;
}

inline std::string GetExprName(Expr expr) {
  if (expr->IsInstance<LetNode>()) {
    Let let = Downcast<Let>(expr);
    return GetExprName(let->value);
  }
  if (expr->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr);
    if (call->op->IsInstance<OpNode>()) {
      std::stringstream ss;
      ss << call->op;
      return ss.str();
    } else {
      return "func";
    }
  } else if (expr->IsInstance<TupleNode>()) {
    return "tuple";
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    return "tgi";
  } else {
    LOG(WARNING) << "Not supported expr: " << expr->GetTypeKey();
    return "unknown";
  }
}

inline Expr TryGetLetValue(Expr expr) {
  if (expr->IsInstance<LetNode>()) {
    return Downcast<Let>(expr)->value;
  }
  return expr;
}

inline Expr TryGetLetVar(Expr expr) {
  if (expr->IsInstance<LetNode>()) {
    return Downcast<Let>(expr)->var;
  }
  return expr;
}

inline Expr TryGetLetBody(Expr expr) {
  if (expr->IsInstance<LetNode>()) {
    return Downcast<Let>(expr)->body;
  }
  return expr;
}

// Tries to get the value as call expr for a let expr.
inline Call GetCall(Expr expr) {
  Expr call_expr = TryGetLetValue(expr);
  CHECK(call_expr->IsInstance<CallNode>());
  return Downcast<Call>(call_expr);
}

// Tries to get the value as call expr for a let expr, and check if the op name matches
// the provided dialect name.
inline Call GetOpCall(Expr expr, std::string dialect_name) {
  Call call = GetCall(expr);
  CHECK(call->op->IsInstance<OpNode>());
  Op op = Downcast<Op>(call->op);
  if (op->name.compare(dialect_name) == 0) {
    return call;   
  }
  LOG(WARNING) << "Expr " << ir::AsText(expr) << ", dialect name " << dialect_name << " does not match.";
  return call;
}

inline Tuple GetTuple(Expr expr) {
  Expr tuple_expr = TryGetLetValue(expr);
  CHECK(tuple_expr->IsInstance<TupleNode>());
  return Downcast<Tuple>(tuple_expr);
}

inline TupleGetItem GetTGI(Expr expr) {
  Expr tgi_expr = TryGetLetValue(expr);
  CHECK(tgi_expr->IsInstance<TupleGetItemNode>());
  return Downcast<TupleGetItem>(tgi_expr);
}

inline HashIndex IndicesHash(Indices& indices) {
  // Use hashing for mapping an indices vector to partitioned exprs,
  // then a tuple's partition indices can be mapped to partitioned tuples.
  static const auto hasher = std::hash<std::string>{};
  std::string indices_str = PrintIndices(indices);
  return hasher(indices_str);
}

template <class T>
inline bool AllEqual(const std::vector<T>& vec, T ele) {
  for (const auto& ele_ : vec) {
    if (ele_ != ele) {
      return false;
    }
  }
  return true;
}

template <class T>
inline bool AllNotEqual(const std::vector<T>& vec, T ele) {
  for (const auto& ele_ : vec) {
    if (ele_ == ele) {
      return false;
    }
  }
  return true;
}

template <class T>
inline bool AnyNotEqual(const std::vector<T>& vec, T ele) {
  for (const auto& ele_ : vec) {
    if (ele_ != ele) {
      return true;
    }
  }
  return false;
}

template <class T>
inline bool VecAllEqual(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs.at(i) != rhs.at(i)) {
      return false;
    }
  }
  return true;
}

template <class T>
inline bool VecsAllEqual(const std::vector<std::vector<T>>& vecs) {
  if (vecs.size() <= 1) {
    return true;
  }
  const auto& vec = vecs.at(0);
  for (int i = 1; i < vecs.size(); ++i) {
    if (!VecAllEqual(vec, vecs.at(i))) {
      return false;
    }
  }
  return true;
}

inline bool IsTest() {
  if (const auto* run_test = getenv("RUN_TEST")) {
    if (strcmp(run_test, "1") == 0) {
      return true;
    }
  }
  return false;
}

class DisjointSet {
 public:
  DisjointSet(int n) {
    if (n == 0) {
      return;
    }
    ranks.resize(n);
    parents.resize(n);
    for(int i = 0; i < n; i++) {
      parents[i] = i;
      ranks[i] = 1;
    }
  }

  void Combine(int i, int j) {
    CHECK(i < ranks.size() && j < ranks.size());
    int parent_i = Find_(i);
    int parent_j = Find_(j);
    if(parent_i == parent_j) {
      return;
    }
    if (ranks[parent_i] > ranks[parent_j]) {
      parents[parent_j] = parent_i;
      ranks[parent_i] += ranks[parent_j];
    } else {
      parents[parent_i] = parent_j;
      ranks[parent_j] += ranks[parent_i];
    }
  }

  std::vector<IndexSet> GetGroups() {
    std::vector<IndexSet> groups = {};
    IndexMap<Index> root_idx = {};
    for (int i = 0; i < parents.size(); ++i) {
      int parent = Find_(i);
      if (!root_idx.count(parent)) {
        groups.push_back(IndexSet{});
        root_idx[parent] = groups.size() - 1;
      }
      groups[root_idx[parent]].insert(i);
    }
    return groups;
  }

 private:
  int Find_(int idx) {
    CHECK(idx < parents.size());
    if (!(idx == parents[idx])) {
      parents[idx] = Find_(parents[idx]);
    }
    return parents[idx];
  }

  std::vector<int> parents;
  std::vector<int> ranks;
};

class LetListUnfolder : public ExprVisitor {
 public:
  Exprs Unfold(Expr expr) {
    unfolded_exprs_.clear();
    VisitExpr(expr);
    return unfolded_exprs_;
  }

  void VisitExpr_(const LetNode* let) {
    unfolded_exprs_.push_back(GetRef<Expr>(let));
    VisitExpr(let->body);
  }

 private:
  Exprs unfolded_exprs_;
};

class ExprUpdater : public ExprMutator {
 public:
  explicit ExprUpdater(ExprMap<Expr> expr_map, Exprs update_range = {}) :
    expr_map_(expr_map), update_range_(update_range.begin(), update_range.end()) {}

  Exprs Update(Expr expr) {
    VisitExpr(expr);
    std::reverse(updated_exprs_.begin(), updated_exprs_.end());
    return updated_exprs_;
  }

  Exprs Update(Exprs exprs) {
    Exprs updated_exprs = {};
    for (auto expr : exprs) {
      updated_exprs.push_back(VisitExpr(expr));
    }
    updated_exprs_.clear();
    return updated_exprs;
  }

  Expr VisitExpr_(const LetNode* let) override {
    if (update_range_.size() > 0 && !update_range_.count(GetRef<Expr>(let))) {
      return GetRef<Expr>(let);
    }
    Var var = Downcast<Var>(VisitExpr(let->var));
    Expr value = VisitExpr(let->value);
    Expr body = VisitExpr(let->body);
    Expr let_expr = Let(var, value, body);
    updated_exprs_.push_back(let_expr);
    return let_expr;
  }

  Expr VisitExpr_(const VarNode* var_node) override {
    auto var = GetRef<Var>(var_node);
    if(expr_map_.count(var)) {
      return expr_map_.at(var);
    }
    return var;
  }

  Expr VisitExpr_(const CallNode* call) override {
    Array<Expr> args = {};
    for (auto arg : call->args) {
      if (expr_map_.count(arg)) {
        args.push_back(TryGetLetVar(expr_map_.at(arg)));
      } else {
        args.push_back(arg);
      }
    }
    Expr call_expr = Call(call->op, args, call->attrs, call->type_args);
    call_expr->checked_type_ = GetRef<Expr>(call)->checked_type_;
    return call_expr;
  }

  Expr VisitExpr_(const TupleNode* tuple) override {
    Array<Expr> fields = {};
    for (auto field : tuple->fields) {
      if (expr_map_.count(field)) {
        fields.push_back(expr_map_.at(field));
      } else {
        fields.push_back(field);
      }
    }
    Expr tuple_expr = Tuple(fields);
    tuple_expr->checked_type_ = GetRef<Expr>(tuple)->checked_type_;
    return tuple_expr;
  }

  Expr VisitExpr_(const TupleGetItemNode* tgi) override {
    Expr tuple;
    if (expr_map_.count(tgi->tuple)) {
      tuple = expr_map_.at(tgi->tuple);
    } else {
      tuple = tgi->tuple;
    }
    Expr tgi_expr = TupleGetItem(tuple, tgi->index);
    tgi_expr->checked_type_ = GetRef<Expr>(tgi)->checked_type_;
    return tgi_expr;
  }

 private:
  Exprs updated_exprs_;
  ExprSet update_range_;
  ExprMap<Expr> expr_map_;
};

// Maps each expr to the set of exprs that consume it
// (along with its argument index in the consumer)
class ConsumptionInfoCollector : public ExprVisitor {
 public:
  ConsumptionInfoCollector(const Exprs& exprs) :
    exprs_(exprs.cbegin(), exprs.cend()) {}
  
  ConsumptionInfoCollector(const ExprSet& exprs) :
    exprs_(exprs) {}
  
  ConsumptionInfoCollector(const Array<Expr>& exprs) :
    exprs_(exprs.begin(), exprs.end()) {}

  ExprMap<ExprMap<Index>> Collect(const Exprs& exprs) {
    for (const auto& expr : exprs) {
      current_ = expr;
      VisitExpr(expr);
    }
    return expr_arg_index_in_consumer_;
  }

  void VisitExpr_(const LetNode* let) override {
    VisitExpr(let->value);
  }

  void VisitExpr_(const CallNode* call) override {
    for (int i = 0; i < call->args.size(); ++i) {
      const auto& arg = call->args[i];
      if (exprs_.count(arg)) {
        expr_arg_index_in_consumer_[arg][current_] = i;
      }
    }
  }

  void VisitExpr_(const TupleNode* tuple) override {
    for (int i = 0; i < tuple->fields.size(); ++i) {
      const auto& field = tuple->fields[i];
      if (exprs_.count(field)) {
        expr_arg_index_in_consumer_[field][current_] = i;
      }
    }
  }

  void VisitExpr_(const TupleGetItemNode* tgi) override {
    const auto& tuple = tgi->tuple;
    if (exprs_.count(tuple)) {
      expr_arg_index_in_consumer_[tuple][current_] = 0;
    }
  }

 private:
  ExprSet exprs_;
  Expr current_;
  // expr -> consuming expr (call) -> the argument index of "expr" in "consumer expr"
  ExprMap<ExprMap<Index>> expr_arg_index_in_consumer_;
};

inline Exprs UnfoldAndUpdateFunction(Expr call, const FunctionNode* func) {
  // Use dependency graph to obtain exprs in a function in topological order,
  // while all parameters are substituted by actual inputs.
  Call func_call = GetCall(call);
  Exprs func_exprs = LetListUnfolder().Unfold(func->body);
  ExprMap<Expr> update_map = {};
  for (int i = 0; i < func->params.size(); ++i) {
    update_map[func->params[i]] = func_call->args[i];
  }
  // also replace all body vars with newly created ones
  for(auto func_expr: func_exprs) {
    auto var = Downcast<Var>(TryGetLetVar(func_expr));
    auto new_var = MakeTypeCheckedVar(var->name_hint() + "_expanded", var->checked_type_);
    update_map[var] = new_var;
  }
  Exprs updated_exprs = ExprUpdater(update_map, func_exprs).Update(func_exprs);
  for(auto expr: updated_exprs) {
    auto var = TryGetLetVar(expr);
    auto value = TryGetLetValue(expr);
  }
  return updated_exprs;
}

inline Indices FindDegenerateAxes(Type tt, int threshold) {
  Indices indices = {};
  TensorType tt_ = Downcast<TensorType>(tt);
  for (int i = 0; i < tt_->shape.size(); ++i) {
    auto node = tt_->shape[i].as<IntImmNode>();
    CHECK(node != nullptr) << "Axis " << tt_->shape[i] << " is not IntImmNode";
    if (node->value < threshold) {
      // forbid partitioning axis with dimension less than threshold
      indices.push_back(i);
    }
  }
  return indices;
}

// An axis can match if their corresponding dim size is the same
// this function always finds the first matching axis
inline IndexMap<Index> FindMatchingAxes(Type type_0, Type type_1) {
  auto shape_0 = ArrayToInt(Downcast<TensorType>(type_0)->shape);
  auto shape_1 = ArrayToInt(Downcast<TensorType>(type_1)->shape);
  // all different dimension sizes
  std::unordered_set<int> dims = {};
  // mapping between dim size and the corresponding dimension index
  std::unordered_map<int, Indices> dim_indices = {};
  for (int i = 0; i < shape_1.size(); ++i) {
    dims.insert(shape_1[i]);
    dim_indices[shape_1[i]].push_back(i);
  }
  IndexMap<Index> matching_axes = {};
  for (int i = 0; i < shape_0.size(); ++i) {
    int dim = shape_0[i];
    if (dim == 1) {
      // Skip the axis that cannot be partitioned.
      continue;
    }
    if (dims.count(dim)) {
      // get the first dimension index where their size is the same
      matching_axes[i] = *dim_indices[dim].begin();
      dim_indices[dim].erase(dim_indices[dim].begin());
      if (dim_indices[dim].size() == 0) {
        dims.erase(dim);
      }
    }
  }
  return matching_axes;
}

// Create the matching axes map with reduced axes.
// This is useful when finding the first matching axis is incorrect.
inline IndexMap<Index> FindMatchingAxes(Type type, Indices reduced_axes, bool exclude = false) {
  int ndim = Downcast<TensorType>(type)->shape.size();
  IndexSet reduced_axes_ = {reduced_axes.begin(), reduced_axes.end()};
  Indices axes = {};
  for (int i = 0; i < ndim; ++i) {
    if (reduced_axes_.count(i) && exclude || !reduced_axes_.count(i) && !exclude) {
      axes.push_back(i);
    }
  }
  IndexMap<Index> matching_axes = {};
  for (int i = 0; i < axes.size(); ++i) {
    matching_axes[axes[i]] = i;
  }
  return matching_axes;
}

inline IndexMap<Index> FindUnmatchedAxes(Type type_0, Type type_1) {
  auto shape_0 = ArrayToInt(Downcast<TensorType>(type_0)->shape);
  auto shape_1 = ArrayToInt(Downcast<TensorType>(type_1)->shape);
  IndexSet unmatched_axes_0 = {};
  IndexSet unmatched_axes_1 = {};
  for (int i = 0; i < shape_0.size(); ++i) {
    if(shape_0[i] != 1) {
      // ignore cases where dim = 1
      unmatched_axes_0.insert(i);
    }
  }
  for (int i = 0; i < shape_1.size(); ++i) {
    if(shape_1[i] != 1) {
      // ignore cases where dim = 1
      unmatched_axes_1.insert(i);
    }
  }
  auto matching_axes = FindMatchingAxes(type_0, type_1);
  for (auto it : matching_axes) {
    unmatched_axes_0.erase(it.first);
    unmatched_axes_1.erase(it.second);
  }
  if(unmatched_axes_0.empty()) {
    // unmatched_axes_1 must also be empty
    CHECK(unmatched_axes_1.empty());
    return {};
  }
  CHECK(!unmatched_axes_0.empty() && !unmatched_axes_1.empty());
  Index min_unmatched_axes_0 = *std::min_element(unmatched_axes_0.begin(), unmatched_axes_0.end());
  Index min_unmatched_axes_1 = *std::min_element(unmatched_axes_1.begin(), unmatched_axes_1.end());
  // TODO(@ye-tian) Find all combinations of unmatched dims and their product. 
  // Get the first axis of these combinations and add to the result.
  IndexMap<Index> unmatched_axes = {};
  unmatched_axes[min_unmatched_axes_0] = min_unmatched_axes_1;
  return unmatched_axes;
}

inline Index FindSplitAxis(Type type_0, Type type_1) {
  auto shape_0 = ArrayToInt(Downcast<TensorType>(type_0)->shape);
  auto shape_1 = ArrayToInt(Downcast<TensorType>(Downcast<TupleType>(type_1)->fields[0])->shape);
  CHECK(shape_0.size() == shape_1.size());
  for (int i = 0; i < shape_0.size(); ++i) {
    if (shape_0[i] != shape_1[i]) {
      return i;
    }
  }
  LOG(FATAL) << "Split axis not found.";
}

// Valid arguments should not be scalar or constants
// scalar var's type can be tensor type with 0 dims
// constant's type can be tensor type with 0 dims
// or tuple type with each field is a tensor type with 0 dims
inline bool IsValidTensorType(Type type) {
  if (!type->IsInstance<TensorTypeNode>()) {
    return false;
  }
  return Downcast<TensorType>(type)->shape.size() != 0;
}

inline bool IsValidTensor(Expr expr) {
  return IsValidTensorType(expr->checked_type_);
}

inline bool IsValidTuple(Expr expr) {
  Type type = expr->checked_type_;
  if (auto tuple_type = type.as<TupleTypeNode>()) {
    bool is_all_valid = true;
    for(auto field_type: tuple_type->fields) {
      if(!IsValidTensorType(field_type)) {
        is_all_valid = false;
        break;
      }
    }
    return is_all_valid;
  }
  return false;
}

// Automatically infer the valid indices from the call arguments.
// Valid indices are the indices that are tensors or tuple of tensors (not constants).
inline Indices GetValidIndices(Call call, Indices arg_indices) {
  Indices valid_indices = {};
  for (auto index : arg_indices) {
    if (auto constant_node = call->args[index].as<ConstantNode>()) {
      // test if is scalar constant
      if (constant_node->IsTensor()) {
        // test if the tensor is 0d
        auto tensor_type = Downcast<TensorType>(constant_node->checked_type_);
        if (tensor_type->shape.size() != 0) {
          valid_indices.push_back(index);
        }
      }
    } else if (IsValidTensor(call->args[index]) || IsValidTuple(call->args[index])) {
      valid_indices.push_back(index);
    }
  }
  return valid_indices;
}

inline std::vector<int64_t> GetShapeFromTensorType(TensorType type) {
  std::vector<int64_t> result = {};
  for (auto axis : Downcast<TensorType>(type)->shape) {
    result.push_back(*as_const_int(axis));
  }
  return result;
}

inline int GetInt(Expr expr) {
  return Downcast<IntValue>(expr.as<ConstantNode>()->value)->value;
}

template <class T>
inline std::vector<T> GetInts(Expr expr) {
  auto fields = Downcast<TupleValue>(expr)->fields;
  std::vector<T> vec = {};
  for (auto field : fields) {
    vec.push_back(Downcast<IntValue>(field)->value);
  }
  return vec;
}

inline bool GetBool(Expr expr) {
  return Downcast<BoolValue>(expr.as<ConstantNode>()->value)->value;
}

inline bool ContainsAllToAll(ExprSet& exprs) {
  // static const std::unordered_set<std::string> all2all_names = {
  //   "raf.op._all_to_all",
  //   "raf.op.nccl._all_to_all",
  // };
  // for (auto expr : exprs) {
  //   if (expr->IsInstance<CallNode>()) {
  //     Call call = Downcast<Call>(expr);
  //     Expr call_op = call->op;
  //     if (call_op->IsInstance<OpNode>()) {
  //       Op op = Downcast<Op>(call_op);
  //       if (all2all_names.count(op->name)) {
  //         return true;
  //       }
  //     }
  //   }
  // }
  return false;
}

inline bool ContainsAllToAll(Exprs& exprs) {
  // static const std::unordered_set<std::string> all2all_names = {
  //   "raf.op._all_to_all",
  //   "raf.op.nccl._all_to_all",
  // };
  // for (auto expr : exprs) {
  //   if (expr->IsInstance<CallNode>()) {
  //     Call call = Downcast<Call>(expr);
  //     Expr call_op = call->op;
  //     if (call_op->IsInstance<OpNode>()) {
  //       Op op = Downcast<Op>(call_op);
  //       if (all2all_names.count(op->name)) {
  //         return true;
  //       }
  //     }
  //   }
  // }
  return false;
}

inline Type GetPaddedOrSplitType(TensorType tt, Index axis, int n_part, bool is_padded) {
  auto shapes = tt->shape;
  if (axis == SpecialAxis::kExpert || axis == SpecialAxis::kIndiceAndLocations || axis == SpecialAxis::kNone) {
    // duplicate
    Array<Type> types = Array<Type>(n_part, tt);
    return TupleType(types);
  }
  int64_t dim = shapes[axis].as<IntImmNode>()->value;
  Array<PrimExpr> new_shapes = {shapes.begin(), shapes.end()};
  int64_t part_dim = dim / n_part;
  if (dim % n_part != 0) {
    ++part_dim;
  }
  if (is_padded) {
    new_shapes.Set(axis, tvm::IntImm(DataType::Int(32), part_dim * n_part));
    return TensorType(new_shapes, tt->dtype);
  } else {
    new_shapes.Set(axis, tvm::IntImm(DataType::Int(32), part_dim));
    TensorType tt_ = TensorType(new_shapes, tt->dtype);
    Array<Type> types = Array<Type>(n_part, tt_);
    return TupleType(types);
  }
}

inline Type GetSplitType(Type type, Index axis, int n_part) {
  CHECK(type->IsInstance<TensorTypeNode>());
  return GetPaddedOrSplitType(Downcast<TensorType>(type), axis, n_part, false);
}

inline Type GetPaddedType(Type type, Index axis, int n_part) {
  CHECK(type->IsInstance<TensorTypeNode>());
  return GetPaddedOrSplitType(Downcast<TensorType>(type), axis, n_part, true);
}

inline Type GetSplitFieldType(Type type, Index axis, int n_part) {
  if(type->IsInstance<TensorTypeNode>()) {
    Type tt = GetPaddedOrSplitType(Downcast<TensorType>(type), axis, n_part, false);
    return Downcast<TupleType>(tt)->fields[0];
  } else {
    Array<Type> tuple_type_fields;
    auto tuple_type = Downcast<TupleType>(type);
    for(int i=0; i<tuple_type->fields.size(); i++) {
      tuple_type_fields.push_back(GetSplitFieldType(tuple_type->fields[i], axis, n_part));
    }
    return TupleType(tuple_type_fields);
  }
}

inline Type GetSplitFieldType(Type type, Indices axes, int n_part) {
  if(type->IsInstance<TensorTypeNode>()) {
    CHECK_EQ(axes.size(), 1);
    return GetSplitFieldType(type, axes[0], n_part);
  } else {
    auto tuple_type = Downcast<TupleType>(type);
    CHECK_EQ(tuple_type->fields.size(), axes.size());
    Array<Type> tuple_type_fields;
    for(int i=0; i<tuple_type->fields.size(); i++) {
      tuple_type_fields.push_back(GetSplitFieldType(tuple_type->fields[i], axes[i], n_part));
    }
    return TupleType(tuple_type_fields);
  }
}

inline Type GetConcatenatedType(Type type, Index axis, int n) {
  CHECK(type->IsInstance<TensorTypeNode>());
  TensorType tt = Downcast<TensorType>(type);
  auto shapes = tt->shape;
  int dim = *as_const_int(shapes[axis]);
  Array<PrimExpr> new_shapes = {shapes.begin(), shapes.end()};
  new_shapes.Set(axis, tvm::IntImm(DataType::Int(32), n * dim));
  return TensorType(new_shapes, tt->dtype);
}

inline Type GetTupleType(Type type, int n) {
  Array<Type> types = {};
  for (int i = 0; i < n; ++i) {
    types.push_back(type);
  }
  return TupleType(types);
}

inline Bools GetFieldGroups(Indices& axes) {
  Bools groups = {};
  for (int i = 0; i < axes.size(); ++i) {
    groups.push_back(!(i == 0 || axes[i] != axes[i - 1] || axes[i] == -1));
  }
  return groups;
}

inline Expr CreateShapeFromTensorType(TensorType tt) {
  auto shape = tt->shape;
  Array<Type> shape_types;
  Array<Value> shape_ = {};
  for (auto dim : shape) {
    shape_.push_back(ScalarValue::make(*as_const_int(dim)));
    shape_types.push_back(TensorType({}, DataType::Int(64)));
  }
  auto shape_const = MakeConstant(TupleValue::make(shape_));
  TupleType shape_tuple_type = TupleType(shape_types);
  shape_const->checked_type_ = shape_tuple_type;
  return shape_const;
}

inline bool IsEqualShape(TensorType tt_0, TensorType tt_1) {
  auto shape_0 = tt_0->shape;
  auto shape_1 = tt_1->shape;
  if (shape_0.size() != shape_1.size()) {
    return false;
  }
  int ndim = shape_0.size();
  int axis = -1;
  for (int i = 0; i < ndim; ++i) {
    if (*as_const_int(shape_0[i]) != *as_const_int(shape_1[i])) {
      axis = i;
      break;
    }
  }
  if (axis != -1) {
    return false;
  }
  return true;
}

inline Expr FindMisMatchedAxis(TensorType tt_0, TensorType tt_1) {
  auto shape_0 = tt_0->shape;
  auto shape_1 = tt_1->shape;
  CHECK(shape_0.size() == shape_1.size());
  int ndim = shape_0.size();
  int axis = -1;
  for (int i = 0; i < ndim; ++i) {
    if (*as_const_int(shape_0[i]) != *as_const_int(shape_1[i])) {
      axis = i;
      break;
    }
  }
  CHECK(axis != -1);
  auto axis_as_const = MakeConstant(ScalarValue::make(axis));
  axis_as_const->checked_type_ = TensorType({}, DataType::Int(32));
  return axis_as_const;
}

inline Array<Expr> ExtractArgs(Expr expr) {
  Array<Expr> args = {};
  if (expr->IsInstance<CallNode>()) {
    args = Downcast<Call>(expr)->args;
  } else if (expr->IsInstance<TupleNode>()) {
    args = Downcast<Tuple>(expr)->fields;
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    args.push_back(Downcast<TupleGetItem>(expr)->tuple);
  }
  return args;
}

inline bool IsCollectiveCall(Expr expr) {
  if (expr->IsInstance<CallNode>()) {
    return IsCollectiveOp(Downcast<Call>(expr)->op);
  } else {
    return false;
  }
}

inline std::vector<int64_t> GetShapes(Type type) {
  std::vector<int64_t> result = {};
  if (type->IsInstance<TensorTypeNode>()) {
    for (auto axis : Downcast<TensorType>(type)->shape) {
      result.push_back(*as_const_int(axis));
    }
    return result;
  } else {
    for (auto field : Downcast<TupleType>(type)->fields) {
      auto result_ = GetShapes(field);
      result.insert(result.end(), result_.begin(), result_.end());
    }
    return result;
  }
}

} // namespace partition_common
} // namespace pass
} // namespace raf