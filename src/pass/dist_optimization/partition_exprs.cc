/*!
 * Copyright (c) 2022 by Contributors
 * \file partition_exprs.cc
 * \brief Partition given exprs.
 */
#include "./partition_common.h"
#include "./solve_partition_axes.h"
#include "partition_exprs.h"

namespace raf {
namespace pass {
namespace partition_exprs {

ScheduleLocation ScheduleAtPrelude() {
  return {ScheduleType::kPrelude, -1};
}

ScheduleLocation ScheduleAtPipelineIndex(int index) {
  return {ScheduleType::kPipeline, index};
}

ScheduleLocation ScheduleAtEpilogue() {
  return {ScheduleType::kEpilogue, -1};
}

ScheduleLocation ArbitrarySchedule() {
  return {ScheduleType::kArbitrary, -1};
}

std::ostream& operator << (std::ostream &os, const ScheduleType &st) {
  if(st == ScheduleType::kPrelude) {
    return os << "Prelude";
  } else if (st == ScheduleType::kPipeline) {
    return os << "Pipeline";
  } else {
    return os << "Epilogue";
  }
}

std::ostream& operator << (std::ostream &os, const ScheduleLocation &location) {
  os << "[" << location.type;
  if(location.type == ScheduleType::kPipeline) {
    os << ", " << location.partition_index;
  }
  return os << "]";
}

std::string PrintVarAndExpr(const VarAndExpr& vae) {
  return std::string("var: ") + ir::AsText(vae.var) + ", expr: " + ir::AsText(vae.expr);
}

std::string PrintVarAndExprs(const VarAndExprs& vaes) {
  std::string result = "[";
  for (const auto& vae : vaes) {
    result += PrintVarAndExpr(vae);
    result += ", ";
  }
  if (result.size() > 1) {
    result.pop_back();
    result.pop_back();
  }
  result += "]";
  return result;
}

std::string PrintInstr(const PartitionInstr& instr) {
  return std::string("n_part: ") + std::to_string(instr.n_part);
}

std::string PrintPartitionedANFBlock(const PartitionedANFBlock& pblock) {
  std::string result_str = std::string("[\n\tPrelude: ") + PrintVarAndExprs(pblock.prelude) + "\n\tPartitions: ";
  for(auto partition_vae: pblock.partitions) {
    result_str += "\n\t\t" + PrintVarAndExprs(partition_vae);
  }
  result_str += "\n\tEpilogue: " + PrintVarAndExprs(pblock.epilogue);
  result_str += "\n]";
  return result_str;
}

std::string GetPartitionedName(std::string name, Index partition_idx) {
  return name + "_part_" + std::to_string(partition_idx);
}

std::string GetOrigNameFromPartitionedName(std::string name) {
  auto position = name.rfind("_part");
  return name.substr(0, position);
}


void PartitionedANFBlock::PushPrelude(Var var, Expr expr) {
  prelude.push_back({var, expr});
}

void PartitionedANFBlock::PushEpilogue(Var var, Expr expr) {
    epilogue.push_back({var, expr});
}

void PartitionedANFBlock::PushExprsToPartition(const Var& var, const Expr& expr, Index partition_idx, bool is_output) {
  if (partitions.size() < partition_idx + 1) {
    partitions.resize(partition_idx + 1);
  }
  if (partition_output_indices.size() < partition_idx + 1) {
    for (int i = partition_output_indices.size(); i < partition_idx + 1; i++) {
      partition_output_indices.push_back(-1);
    }
  }
  partitions[partition_idx].push_back({var, expr});
  if (is_output) {
    partition_output_indices[partition_idx] = partitions[partition_idx].size() - 1;
  }
}

void PartitionedANFBlock::PushExprsToPartition(const VarAndExpr& vae, Index partition_idx, bool is_output) {
  PushExprsToPartition(vae.var, vae.expr, partition_idx, is_output);
}

void PartitionedANFBlock::PushExprsToPartition(const VarAndExprs& vaes, Index partition_idx, bool is_output) {
  if (partitions.size() < partition_idx + 1) {
    partitions.resize(partition_idx + 1);
  }
  if (partition_output_indices.size() < partition_idx + 1) {
    for (int i = partition_output_indices.size(); i < partition_idx + 1; i++) {
      partition_output_indices.push_back(-1);
    }
  }
  partitions[partition_idx].insert(partitions[partition_idx].end(), vaes.begin(), vaes.end());
  if (is_output) {
    partition_output_indices[partition_idx] = partitions[partition_idx].size() - 1;
  }
}

void PartitionedANFBlock::MergePrelude(const PartitionedANFBlock& other) {
  prelude.insert(prelude.end(), other.prelude.begin(), other.prelude.end());
}

void PartitionedANFBlock::Merge(const PartitionedANFBlock& other) {
  MergePrelude(other);
  if (!other.partitions.empty()) {
    if(partitions.empty()) {
      partitions.resize(other.partitions.size());
    }
    CHECK_EQ(partitions.size(), other.partitions.size()) << "Cannot merge two PartitionedANFBlocks with different number of partitions.";
    for(int i=0; i<partitions.size(); i++) {
      partitions[i].insert(partitions[i].end(), other.partitions[i].begin(), other.partitions[i].end());
    }
  }
  MergeEpilogue(other);
}

void PartitionedANFBlock::MergeEpilogue(const PartitionedANFBlock& other) {
  epilogue.insert(epilogue.end(), other.epilogue.begin(), other.epilogue.end());
}

VarAndExprs PartitionedANFBlock::GetPartitionOutputs() const {
  VarAndExprs outputs;
  CHECK_EQ(partition_output_indices.size(), partitions.size());
  for (int i = 0; i < partitions.size(); i++) {
    CHECK_NE(partition_output_indices[i], -1) << "Partition " << i << " does not have an output.";
    outputs.push_back(partitions[i][partition_output_indices[i]]);
  }
  return outputs;
}

VarAndExpr PartitionedANFBlock::GetEpilogueOutputs() const {
  CHECK(!epilogue.empty()) << "Epilogue is empty.";
  return epilogue.back();
}

PartitionedANFBlock PartitionedANFBlock::MakeIdentity(Var var, Expr expr, int n_partitions) {
  PartitionedANFBlock result;
  for(int i = 0; i < n_partitions; i++) {
    result.PushExprsToPartition(var, expr, i, true);
  }
  return result;
}

Expr findNearestOpWithName(const PartitionEnv& env, const Expr& expr, const std::unordered_set<std::string>& names, bool exclude_self=false, bool verbose=false) {
  // first test self
  if (!exclude_self) {
    if (!env.HasInOutAxes(expr)) {
      return Expr();
    }
    if (expr->IsInstance<CallNode>()) {
      auto call = Downcast<Call>(expr);
      if (call->op->IsInstance<OpNode>()) {
        if (names.count(Downcast<Op>(call->op)->name)) {
          return expr;
        }
      }
    }
  }
  // bfs on node children
  NodeMap<bool> checked_node;
  std::queue<const Node*> q;
  auto& dfg = env.sched_dfg.dfg;
  const Node* node = dfg.getNodeFromExpr(expr);
  for (auto child: dfg.getChildren(node)) {
    if (env.HasInOutAxes(dfg.getExprFromNode(child))) {
      q.push(child);
    }
  }
  if (verbose) {
    LOG(INFO) << "Finding nearest op for: " << expr;
  }
  while (!q.empty()) {
    const Node* node = q.front();
    checked_node[node] = true;
    q.pop();
    if (verbose) {
      LOG(INFO) << "Checking node: " << ir::AsText(dfg.getExprFromNode(node));
    }
    auto expr = dfg.getExprFromNode(node);
    if (expr->IsInstance<CallNode>()) {
      auto call = Downcast<Call>(expr);
      if (call->op->IsInstance<OpNode>()) {
        if (names.count(Downcast<Op>(call->op)->name)) {
          return expr;
        }
      }
    }
    for (auto child: dfg.getChildren(node)) {
      if (!checked_node.count(child) && env.HasInOutAxes(dfg.getExprFromNode(child))) {
        q.push(child);
      }
    }
  }
  return Expr();
}

class FindTGI : public ExprVisitor {
  void VisitExpr_(const LetNode* l) final {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      if (!this->result.defined() && op->value->IsInstance<TupleGetItemNode>()) {
        this->result = op->value;
      }
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(l, pre_visit, post_visit);
  }
public:
  Expr result;
};

Expr findNearestTGI(const PartitionEnv& env, const Expr& expr) {
  // first test self
  if (!env.HasInOutAxes(expr)) {
    return Expr();
  }
  if (expr->IsInstance<TupleGetItemNode>()) {
    return expr;
  } else if (expr->IsInstance<CallNode>()) {
    auto call = Downcast<Call>(expr);
    if (call->op->IsInstance<FunctionNode>()) {
      auto func_expr = Downcast<Function>(call->op);
      FindTGI tgi_finder;
      tgi_finder.VisitExpr(func_expr->body);
      if (tgi_finder.result.defined()) {
        return tgi_finder.result;
      }
    }
  }
  // bfs on node children
  NodeMap<bool> checked_node;
  std::queue<const Node*> q;
  auto& dfg = env.sched_dfg.dfg;
  const Node* node = dfg.getNodeFromExpr(expr);
  for (auto child: dfg.getChildren(node)) {
    if (env.HasInOutAxes(dfg.getExprFromNode(child))) {
      q.push(child);
    }
  }
  while (!q.empty()) {
    const Node* node = q.front();
    checked_node[node] = true;
    q.pop();
    auto expr = dfg.getExprFromNode(node);
    if (expr->IsInstance<TupleGetItemNode>()) {
      return expr;
    } else if (expr->IsInstance<CallNode>()) {
      auto call = Downcast<Call>(expr);
      if (call->op->IsInstance<FunctionNode>()) {
        auto func_expr = Downcast<Function>(call->op);
        FindTGI tgi_finder;
        tgi_finder.VisitExpr(func_expr->body);
        if (tgi_finder.result.defined()) {
          return tgi_finder.result;
        }
      }
    }
    for (auto child: dfg.getChildren(node)) {
      if (!checked_node.count(child) && env.HasInOutAxes(dfg.getExprFromNode(child))) {
        q.push(child);
      }
    }
  }
  return Expr();
}

// fields_vec is of shape [fields, partitions]
// output is of shape [partitions]
VarAndExprs MergeFields(ScheduledDFG& sched_dfg, std::vector<VarAndExprs>& fields_vec, Indices& axes, PartitionInstr& instr) {
  // Given partitioned fields, returns tuples with consecutive partitioned fields
  // in the same tuple. For example, given tuple with 4 fields (f_0, f_1, f_2, f_3)
  // partitioned into 2 parts along axes [1, 2, -1, 0] and partitioned fields
  // [[f_0_0, f_0_1], [f_1_0, f_1_1], [f_2], [f_3]], the output would be
  // [(f_0_0, f_1_0, f_2, f_3_0), (f_0_1, f_1_1, f_2, f_3_1)]. Note that square brackets
  // '[]' represents a series of exprs and brackets '()' represents a tuple.

  // we assume the registration of fields in dfg is already finished
  int n_part = instr.n_part;
  VarAndExprs result;
  for(int partition_idx = 0; partition_idx < n_part; partition_idx++) {
    Array<Expr> reconstructed_tuple_fields;
    Array<Type> reconstructed_field_types;

    for (int field_idx = 0; field_idx < fields_vec.size(); ++field_idx) {
      auto& partitions_at_field = fields_vec[field_idx];
      int field_n_part = partitions_at_field.size();
      CHECK(field_n_part == 1 || field_n_part == n_part) << "Encountered non-regular number of partitions in tuple.";
      VarAndExpr& partitioned_vae = field_n_part == 1 ? partitions_at_field[0] : partitions_at_field[partition_idx];
      // get the output vae
      reconstructed_tuple_fields.push_back(partitioned_vae.var);
      reconstructed_field_types.push_back(partitioned_vae.var->checked_type());
    }
    auto new_tuple = Tuple(reconstructed_tuple_fields);
    new_tuple->checked_type_ = TupleType(reconstructed_field_types);
    auto new_var = MakeTypeCheckedVar("part_recon_tuple_"+std::to_string(partition_idx), TupleType(reconstructed_field_types));

    result.push_back({new_var, new_tuple});
    sched_dfg.insertNewCompExprAndAddInputEdges(new_var, new_tuple, reconstructed_tuple_fields, 
                                            new_var->name_hint(), 0);
  }
  return result;
}

std::vector<VarAndExprs> SplitFields(ScheduledDFG& sched_dfg, VarAndExprs& tuples, Indices& axes) {
  // Given tuples with consecutive partitioned fields in the same tuple,
  // returns partitioned fields. Reverse of MergeFields.
  TupleType tuple_type = Downcast<TupleType>(tuples[0].var->checked_type_);
  int n_field = tuple_type->fields.size();

  std::vector<VarAndExprs> reconstrcted_tuple_fields;
  for(int i=0; i< n_field; i++) {
    VarAndExprs partition_fields;
    int n_considered_tuples = axes[i] != SpecialAxis::kNone ? tuples.size() : 1;
    for(int j=0; j<n_considered_tuples; j++) {
      Expr tgi_expr = TupleGetItem(tuples[j].var, i);
      tgi_expr->checked_type_ = tuple_type->fields[i];
      Var tgi_var = MakeTypeCheckedVar("part_splitfields_"+std::to_string(i) + "_" + std::to_string(j), tuple_type->fields[i]);
      partition_fields.push_back({tgi_var, tgi_expr});
      sched_dfg.insertNewCompExprAndAddInputEdges(tgi_var, tgi_expr, {tuples[j].var}, 
                                              tgi_var->name_hint(), 0);
    }
    reconstrcted_tuple_fields.push_back(partition_fields);
  }
  return reconstrcted_tuple_fields;
}

PartitionedANFBlock SplitVar(PartitionEnv& env, Var var, Indices& axes, PartitionInstr& instr);

PartitionedANFBlock SplitTensor(PartitionEnv& env, Var var, Indices& axes, PartitionInstr& instr) {
  static const auto split_op = op::GetDialectOp("raf.op.tvm.split");
  static const auto concat_op = op::GetDialectOp("raf.op.tvm.concatenate");

  int n_split = instr.n_part;
  TensorType tt = Downcast<TensorType>(var->checked_type_);
  auto shapes = tt->shape;
  CHECK(!shapes->IsInstance<IntImmNode>()) << "Do not support dynamic shape yet";
  

  PartitionedANFBlock result;
  Index axis = axes[0];
  if (axis == SpecialAxis::kNone) {
    // duplicate instead of split
    auto expr = env.GetExprFromVar(var);
    for (int i = 0; i < instr.n_part; ++i) {
      result.PushExprsToPartition(var, expr, i);
    }
    return result;
  } else if (axis == SpecialAxis::kExpert || axis == SpecialAxis::kIndiceAndLocations) {
    // invalid partition, return empty result
    throw InvalidPartitionException("Cannot partition along axis kExpert or kIndiceAndLocations");
  }
  // split instead of duplicate, make sure the tensor is not a scalar
  CHECK(shapes.size() > 0) << "Encountered partitioning an scalar: " << ir::AsText(var);
  int partition_axis_length = *as_const_int(shapes[axis]);

  CHECK_EQ(partition_axis_length % n_split, 0) << "Partition axis length is not divisible by n_split";
  Array<Expr> split_args = {var, MakeConstant(ScalarValue::make(n_split)),
                                        MakeConstant(ScalarValue::make(axis))};
  auto split_expr = Call(split_op, split_args);
  auto split_type = GetSplitType(tt, axis, n_split);
  split_expr->checked_type_ = split_type;
  auto split_var = MakeTypeCheckedVar("part_split_" + var->name_hint(), split_type);

  result.PushPrelude(split_var, split_expr);
  env.SetScheduleLocation(split_expr, ScheduleAtPrelude());
  SimulateTimeType exec_time_;
  std::chrono::milliseconds::rep elapsed_time_;
  std::tie(exec_time_, elapsed_time_) = env.op_profiler.GetCompOpExecTime(split_expr);
  env.AccumulateProfileElapsedTime(elapsed_time_);
  env.sched_dfg.insertNewCompExprAndAddInputEdges(split_var, split_expr, split_args, split_var->name_hint(), exec_time_);

  Vars tgi_vars;
  Exprs tgi_exprs;

  Type tgi_type = GetSplitFieldType(tt, axis, n_split);
  for (int i = 0; i < n_split; ++i) {
    Expr tgi_expr = TupleGetItem(split_var, i);
    tgi_expr->checked_type_ = tgi_type;
    auto tgi_var = MakeTypeCheckedVar("part_split_" + var->name_hint() + "_tgi_" + std::to_string(i), tgi_type);
    tgi_vars.push_back(tgi_var);
    tgi_exprs.push_back(tgi_expr);
    env.sched_dfg.insertNewCompExprAndAddInputEdges(tgi_var, tgi_expr, {split_var}, tgi_var->name_hint(), 0);
  }

  for(int i=0; i<tgi_vars.size(); i++) {
    result.PushExprsToPartition(tgi_vars[i], tgi_exprs[i], i);
    env.SetScheduleLocation(tgi_exprs[i], ScheduleAtPipelineIndex(i));
  }
  return result;
}

PartitionedANFBlock SplitTuple(PartitionEnv& env, Var var, Indices& axes, PartitionInstr& instr) {
  TupleType tt = Downcast<TupleType>(var->checked_type_);
  int n_fields = tt->fields.size();
  PartitionedANFBlock result;

  // shape: [fields, partitions]
  std::vector<VarAndExprs> partitioned_fields;

  for (int i = 0; i < n_fields; ++i) {
    Expr tgi_expr = TupleGetItem(var, i);
    tgi_expr->checked_type_ = tt->fields[i];
    Var tgi_var = MakeTypeCheckedVar("part_prelude_" +var->name_hint() + "_tgi_" + std::to_string(i), tt->fields[i]);

    result.PushPrelude(tgi_var, tgi_expr);
    env.SetScheduleLocation(tgi_expr, ScheduleAtPrelude());
    env.sched_dfg.insertNewCompExprAndAddInputEdges(tgi_var, tgi_expr, {var}, tgi_var->name_hint(), 0);

    // partition the field
    Indices field_axes = {axes[i]};
    PartitionedANFBlock partitioned_field = SplitVar(env, tgi_var, field_axes, instr);
    // merge the PartitionedANFBlocks
    result.Merge(partitioned_field);
    partitioned_fields.push_back(partitioned_field.GetPartitionOutputs());
  }
  VarAndExprs split_exprs = MergeFields(env.sched_dfg, partitioned_fields, axes, instr);
  CHECK_EQ(split_exprs.size(), result.partitions.size());
  for(int i=0; i<split_exprs.size(); i++) {
    result.PushExprsToPartition(split_exprs[i], i);
    env.SetScheduleLocation(split_exprs[i].expr, ScheduleAtPipelineIndex(i));
  }
  return result;
}

PartitionedANFBlock SplitVar(PartitionEnv& env, Var var, Indices& axes, PartitionInstr& instr) {
  if (var->checked_type_->IsInstance<TensorTypeNode>()) {
    return SplitTensor(env, var, axes, instr);
  } else {
    return SplitTuple(env, var, axes, instr);
  }
}

PartitionedANFBlock ConcatExpr(PartitionEnv& env, VarAndExprs partitioned_exprs, Type orig_expr_type, Indices& axes, PartitionInstr& instr, int n_experts, Expr orig_expr = Expr());

PartitionedANFBlock ConcatTensor(PartitionEnv& env, VarAndExprs partitioned_exprs, Type orig_expr_type, Indices& axes, PartitionInstr& instr, int n_experts, Expr orig_expr = Expr()) {
  static const auto split_op = op::GetDialectOp("raf.op.tvm.split");
  static const auto concat_op = op::GetDialectOp("raf.op.tvm.concatenate");
  static const auto merge_masks_op = op::GetDialectOp("raf.op.cuda.moe_merge_masks");
  static const auto redispatch_op = op::GetDialectOp("raf.op.cuda.moe_redispatch");
  static const auto redispatch_expert_input_op = op::GetDialectOp("raf.op.cuda.moe_redispatch_expert_input");
  PartitionedANFBlock result;

  CHECK(partitioned_exprs.size() == instr.n_part);
  Index axis = axes[0];
  Array<Expr> concat_fields = {};
  for(auto vae: partitioned_exprs) {
    concat_fields.push_back(vae.var);
  }
  // construct a tuple of the concatenated exprs
  auto partitioned_type = concat_fields[0]->checked_type_;
  auto concat_tuple_type = GetTupleType(partitioned_type, concat_fields.size());
  Expr concat_tuple_expr = Tuple(concat_fields);
  concat_tuple_expr->checked_type_ = concat_tuple_type;
  Var concat_tuple_var = MakeTypeCheckedVar("part_" + GetOrigNameFromPartitionedName(partitioned_exprs[0].var->name_hint()) + "_concat_tuple", concat_tuple_type);
  result.PushEpilogue(concat_tuple_var, concat_tuple_expr);
  env.SetScheduleLocation(concat_tuple_expr, ScheduleAtEpilogue());
  env.sched_dfg.insertNewCompExprAndAddInputEdges(concat_tuple_var, concat_tuple_expr, concat_fields, concat_tuple_var->name_hint(), 0);

  Type concat_type;
  Expr concat_expr;
  Array<Expr> concat_args;
  std::string name_suffix = "";
  bool should_profile = true;
  if (axis == SpecialAxis::kIndiceAndLocations) {
    // special case for the indice and locations axis
    concat_type = partitioned_type;
    concat_args = {concat_tuple_var, MakeConstant(ScalarValue::make(n_experts))};
    concat_expr = Call(merge_masks_op, concat_args);
    // ops's execution time is related to data, can't be correctly profiled
    should_profile = false;
  } else if (axis == SpecialAxis::kExpert) {
    CHECK(orig_expr.defined()) << "orig_expr must be defined for axis kExpert. Concat failed for " << ir::AsText(partitioned_exprs[0].expr);
    concat_type = partitioned_type;
    // we need to use different concat op depending on the tensor layout
    // two cases:
    // 1. if the moe node label is kMoEExpert (i.e. between two a2as), need to use moe_redispatch_expert_input
    // 2. else, use moe_redispatch
    // determine the case
    auto moe_label = env.GetMoENodeLabel(orig_expr);
    CHECK_NE(moe_label.type, MoENodeType::kUnknown) << "Cannot determine the MoE node type for: " << ir::AsText(orig_expr);
    if (moe_label.type == MoENodeType::kMoEExperts) {
      // case 1
      auto nearest_all_to_all = findNearestOpWithName(env, orig_expr, {"raf.op.nccl._all_to_all"});
      // find local expert id by looking at the nearest TupleGetItem node
      auto nearest_tgi = findNearestTGI(env, orig_expr);
      CHECK(nearest_tgi.defined());
      int local_expert_id = Downcast<TupleGetItem>(nearest_tgi)->index;
      int n_local_experts = Downcast<TupleType>(Downcast<TupleGetItem>(nearest_tgi)->tuple->checked_type_)->fields.size();
      auto predecessor_partition_axes = env.GetOutAxes(nearest_all_to_all);
      auto partitioned_exprs = env.GetPartitionedExprs(nearest_all_to_all, predecessor_partition_axes);
      VarAndExprs alltoallv_outputs;
      for (int i=0; i< partitioned_exprs.partitions.size(); i++) {
        auto& vaes = partitioned_exprs.partitions[i];
        int out_index = partitioned_exprs.partition_output_indices[i];
        CHECK_GT(out_index, 0);
        alltoallv_outputs.push_back(vaes[out_index - 1]);
      }
      Array<Expr> per_partition_recv_cnts;
      for (int i=0; i<instr.n_part; i++) {
        Expr part_i_recv_cnt_expr = TupleGetItem(alltoallv_outputs[i].var, 1);
        auto recv_cnt_type = Downcast<TupleType>(alltoallv_outputs[i].var->checked_type_)->fields[1];
        part_i_recv_cnt_expr->checked_type_ = recv_cnt_type;
        Var part_i_recv_cnt_var = MakeTypeCheckedVar(GetPartitionedName("recv", i), recv_cnt_type);
        result.PushEpilogue(part_i_recv_cnt_var, part_i_recv_cnt_expr);
        env.SetScheduleLocation(part_i_recv_cnt_expr, ScheduleAtEpilogue());
        env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_recv_cnt_var, part_i_recv_cnt_expr, 
                                                        {alltoallv_outputs[i].var}, 
                                                        GetPartitionedName("recv", i), 0);
        per_partition_recv_cnts.push_back(part_i_recv_cnt_var);
      }
      // make a tuple of per_partition_recv_cnts
      auto per_partition_recv_cnts_type = GetTupleType(per_partition_recv_cnts[0]->checked_type_, per_partition_recv_cnts.size());
      Expr per_partition_recv_cnts_tuple_expr = Tuple(per_partition_recv_cnts);
      per_partition_recv_cnts_tuple_expr->checked_type_ = per_partition_recv_cnts_type;
      Var per_partition_recv_cnts_tuple_var = MakeTypeCheckedVar("per_partition_recv_cnts_tuple", per_partition_recv_cnts_type);
      result.PushEpilogue(per_partition_recv_cnts_tuple_var, per_partition_recv_cnts_tuple_expr);
      env.SetScheduleLocation(per_partition_recv_cnts_tuple_expr, ScheduleAtEpilogue());
      env.sched_dfg.insertNewCompExprAndAddInputEdges(per_partition_recv_cnts_tuple_var, per_partition_recv_cnts_tuple_expr, 
                                                      per_partition_recv_cnts, "per_partition_recv_cnts_tuple", 0);
      // test if we need to scale the recv_cnts
      auto nearest_tgi_shape = GetShapes(Downcast<TensorType>(nearest_tgi->checked_type_));
      auto unscaled_model_dim = nearest_tgi_shape[nearest_tgi_shape.size() - 1];
      auto current_input_shape = GetShapes(Downcast<TensorType>(Downcast<TupleType>(concat_tuple_type)->fields[0]));
      auto current_model_dim = current_input_shape[current_input_shape.size() - 1];
      int scale_factor = current_model_dim / unscaled_model_dim;
      concat_args = {concat_tuple_var, per_partition_recv_cnts_tuple_var, MakeConstant(ScalarValue::make(local_expert_id)), MakeConstant(ScalarValue::make(n_local_experts)), MakeConstant(ScalarValue::make(scale_factor))};
      concat_expr = Call(redispatch_expert_input_op, concat_args);
      name_suffix = "_redispatch_expert";
    } else {
      // case 2. get dispatch masks for orig_expr
      auto nearest_moe_encode = findNearestOpWithName(env, orig_expr, {"raf.op.cuda.moe_encode", "raf.op.cuda.moe_encode_batch_prioritized"});
      CHECK(nearest_moe_encode.defined());
      auto predecessor_partition_axes = env.GetOutAxes(nearest_moe_encode);
      auto partitioned_exprs = env.GetPartitionedOutputs(nearest_moe_encode, predecessor_partition_axes);
      Array<Expr> per_partition_dispatch_masks;
      for (int i=0; i<instr.n_part; i++) {
        Expr part_i_indlocs = TupleGetItem(partitioned_exprs[i].var, 1);
        auto indlocs_type = Downcast<TupleType>(partitioned_exprs[i].var->checked_type_)->fields[1];
        part_i_indlocs->checked_type_ = indlocs_type;
        Var part_i_indlocs_var = MakeTypeCheckedVar(GetPartitionedName("indlocs", i), indlocs_type);
        result.PushEpilogue(part_i_indlocs_var, part_i_indlocs);
        env.SetScheduleLocation(part_i_indlocs, ScheduleAtEpilogue());
        env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_indlocs_var, part_i_indlocs, 
                                                        {partitioned_exprs[i].var}, 
                                                        GetPartitionedName("indlocs", i), 0);
        per_partition_dispatch_masks.push_back(part_i_indlocs_var);
      }
      // construct a tuple of dispatch masks
      auto dispatch_masks_type = GetTupleType(per_partition_dispatch_masks[0]->checked_type_, per_partition_dispatch_masks.size());
      Expr dispatch_masks_tuple_expr = Tuple(per_partition_dispatch_masks);
      dispatch_masks_tuple_expr->checked_type_ = dispatch_masks_type;
      Var dispatch_masks_tuple_var = MakeTypeCheckedVar("part_" + GetOrigNameFromPartitionedName(partitioned_exprs[0].var->name_hint()) + "_dispatch_masks_tuple", dispatch_masks_type);
      result.PushEpilogue(dispatch_masks_tuple_var, dispatch_masks_tuple_expr);
      env.SetScheduleLocation(dispatch_masks_tuple_expr, ScheduleAtEpilogue());
      env.sched_dfg.insertNewCompExprAndAddInputEdges(dispatch_masks_tuple_var, dispatch_masks_tuple_expr, per_partition_dispatch_masks, dispatch_masks_tuple_var->name_hint(), 0);
      concat_args = {concat_tuple_var, dispatch_masks_tuple_var};
      concat_expr = Call(redispatch_op, concat_args);
      name_suffix = "_redispatch";
    }
    // these ops's execution time is related to data, can't be correctly profiled
    should_profile = false;
  } else {
    concat_type = GetConcatenatedType(partitioned_type, axis, concat_fields.size());
    concat_args = {concat_tuple_var, MakeConstant(ScalarValue::make(axis))};
    concat_expr = Call(concat_op, concat_args);
  }
  concat_expr->checked_type_ = concat_type;
  Var concat_var = MakeTypeCheckedVar("part_" + GetOrigNameFromPartitionedName(partitioned_exprs[0].var->name_hint())+"_concat"+name_suffix, concat_type);
  result.PushEpilogue(concat_var, concat_expr);
  env.SetScheduleLocation(concat_expr, ScheduleAtEpilogue());
  SimulateTimeType exec_time_;
  if (should_profile) {
    std::chrono::milliseconds::rep elapsed_time_;
    std::tie(exec_time_, elapsed_time_) = env.op_profiler.GetCompOpExecTime(concat_expr);
    env.AccumulateProfileElapsedTime(elapsed_time_);
  } else {
    // manually assign a 100 us overhead for redispatch op
    exec_time_ = 100;
  }
  env.sched_dfg.insertNewCompExprAndAddInputEdges(concat_var, concat_expr, concat_args, concat_var->name_hint(), exec_time_);
  return result;
}

PartitionedANFBlock ConcatTuple(PartitionEnv& env, VarAndExprs partitioned_exprs, Type orig_expr_type, Indices& axes, PartitionInstr& instr, int n_experts, Expr orig_expr = Expr()) {
  std::vector<VarAndExprs> tgis_vec = SplitFields(env.sched_dfg, partitioned_exprs, axes); // shape [fields, partitions]
  TupleType tt = Downcast<TupleType>(orig_expr_type);
  PartitionedANFBlock result;

  for(int field_idx = 0; field_idx < tgis_vec.size(); field_idx ++) {
    auto partition_tgis = tgis_vec[field_idx];
    if(partition_tgis.size() == 1) {
      result.PushEpilogue(partition_tgis[0].var, partition_tgis[0].expr);
      env.SetScheduleLocation(partition_tgis[0].expr, ScheduleAtEpilogue());
    } else {
      for(int part_idx = 0; part_idx < partition_tgis.size(); part_idx++) {
        result.PushExprsToPartition(partition_tgis[part_idx].var, partition_tgis[part_idx].expr, part_idx);
        env.SetScheduleLocation(partition_tgis[part_idx].expr, ScheduleAtPipelineIndex(part_idx));
      }
    }
  }

  Array<Expr> fields = {};
  for(int field_idx = 0; field_idx < tgis_vec.size(); field_idx ++) {
    auto partition_tgis = tgis_vec[field_idx];
    if(partition_tgis.size() == 1) {
      fields.push_back(partition_tgis[0].var);
    } else {
      Indices concat_axes = {axes[field_idx]};
      auto concat_pblock = ConcatExpr(env, partition_tgis, tt->fields[field_idx], concat_axes, instr, n_experts, orig_expr);
      result.Merge(concat_pblock);
      auto concat_output = concat_pblock.GetEpilogueOutputs();
      fields.push_back(concat_output.var);
    }
  }
  Expr tuple_expr = Tuple(fields);
  tuple_expr->checked_type_ = tt;
  Var tuple_var = MakeTypeCheckedVar("part_recon_concat_tuple", tt);
  result.PushEpilogue(tuple_var, tuple_expr);
  env.SetScheduleLocation(tuple_expr, ScheduleAtEpilogue());
  env.sched_dfg.insertNewCompExprAndAddInputEdges(tuple_var, tuple_expr, fields, tuple_var->name_hint(), 0);
  return result;
}

PartitionedANFBlock ConcatExpr(PartitionEnv& env, VarAndExprs partitioned_exprs, Type orig_expr_type, Indices& axes, PartitionInstr& instr, int n_experts, Expr orig_expr) {
  if (orig_expr_type->IsInstance<TensorTypeNode>()) {
    return ConcatTensor(env, partitioned_exprs, orig_expr_type, axes, instr, n_experts, orig_expr);
  } else {
    return ConcatTuple(env, partitioned_exprs, orig_expr_type, axes, instr, n_experts, orig_expr);
  }
}

PartitionedANFBlock ConcatExpr(PartitionEnv& env, VarAndExprs partitioned_exprs, Expr orig_expr, Indices& axes, PartitionInstr& instr, int n_experts) {
  if (orig_expr->checked_type_->IsInstance<TensorTypeNode>()) {
    return ConcatTensor(env, partitioned_exprs, orig_expr->checked_type_, axes, instr, n_experts, orig_expr);
  } else {
    return ConcatTuple(env, partitioned_exprs, orig_expr->checked_type_, axes, instr, n_experts, orig_expr);
  }
}

PartitionEnv::PartitionEnv(ScheduledDFG& sched_dfg,
                           ExtendedOpProfiler& op_profiler,
                           const NodeMap<MoENodeLabel>& moe_label_map,
                           const ExprMap<InOutCellRefs>& expr_inout_axes,
                           const ExprMap<Exprs>& expr_func_exprs,
                           const ExprMap<IndexMap<Index>>& expr_arg_var,
                           const ExprMap<ExprMap<ExprMap<Index>>>& expr_arg_consumer_index,
                           int dp_group_size, int n_part) :
  sched_dfg(sched_dfg),
  op_profiler(op_profiler),
  moe_label_map_(moe_label_map),
  expr_inout_axes_(expr_inout_axes),
  expr_func_exprs_(expr_func_exprs),
  expr_arg_var_(expr_arg_var),
  expr_arg_consumer_index_(expr_arg_consumer_index),
  dp_group_size_(dp_group_size) {
  SetFuncExprs_();
  SetAxesExprs_(n_part);
  SetInstr_(sched_dfg.dfg, dp_group_size, n_part);
}

ExprMap<PartitionedANFBlock> PartitionEnv::GetPartitionedExprs() const {
  ExprMap<PartitionedANFBlock> result;
  for (auto it : expr_axes_exprs_) {
    if (HasInOutAxes(it.first)) {
      Indices out_axes = GetOutAxes(it.first);
      auto pblock = GetPartitionedExprs(it.first, out_axes);
      result[it.first] = pblock;
    } else {
      // input variables
      auto axes_exprs_map = it.second;
      CHECK_EQ(axes_exprs_map.size(), 1);
      for(auto map_it: axes_exprs_map) {
        result[it.first] = map_it.second;
      }
    }
  }
  return result;
}

PartitionedANFBlock PartitionEnv::GetPartitionedExprs(Expr expr, Indices& axes) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_axes_exprs_.count(expr));
  return expr_axes_exprs_.at(expr).at(IndicesHash(axes));
}

// partitioned exprs

bool PartitionEnv::HasPartitionedExprs(Expr expr, Indices& axes) const {
  expr = GetAndCheckExpr_(expr);
  return expr_axes_exprs_.count(expr) && expr_axes_exprs_.at(expr).count(IndicesHash(axes));
}

VarAndExprs PartitionEnv::GetPartitionedOutputs(Expr expr, Indices& axes) {
  expr = GetAndCheckExpr_(expr);
  return GetPartitionedExprs(expr, axes).GetPartitionOutputs();
}

void PartitionEnv::SetPartitionedExprs(Expr expr, Indices& axes, PartitionedANFBlock pblock) {
  expr = GetAndCheckExpr_(expr);
  expr_axes_exprs_[expr][IndicesHash(axes)] = pblock;
}

// arg var

bool PartitionEnv::HasArgVar(Expr expr, Index arg) const {
  expr = GetAndCheckExpr_(expr);
  if (expr_arg_var_.count(expr)) {
    return expr_arg_var_.at(expr).count(arg);
  }
  return false;
}

Index PartitionEnv::GetArgVar(Expr expr, Index arg) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_arg_var_.count(expr));
  CHECK(expr_arg_var_.at(expr).count(arg));
  return expr_arg_var_.at(expr).at(arg);
}

void PartitionEnv::SetArgVar(Expr expr, Index arg, Index var) {
  expr = GetAndCheckExpr_(expr);
  expr_arg_var_.at(expr)[arg] = var;
}

// updated arg

bool PartitionEnv::HasUpdatedArg(Expr expr, Index index) const {
  expr = GetAndCheckExpr_(expr);
  return expr_index_updated_arg_.count(expr) && 
    expr_index_updated_arg_.at(expr).count(index);
}

Expr PartitionEnv::GetUpdatedArg(Expr expr, Index index) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_index_updated_arg_.count(expr));
  CHECK(expr_index_updated_arg_.at(expr).count(index));
  return expr_index_updated_arg_.at(expr).at(index);
}

void PartitionEnv::SetUpdatedArg(Expr expr, Index index, Expr updated_arg) {
  expr = GetAndCheckExpr_(expr);
  expr_index_updated_arg_[expr][index] = updated_arg;
}

// consumer index

ExprMap<Index> PartitionEnv::GetConsumerIndex(Expr expr, Expr arg) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_arg_consumer_index_.count(expr));
  CHECK(expr_arg_consumer_index_.at(expr).count(arg));
  return expr_arg_consumer_index_.at(expr).at(arg);
}

// in out axes

bool PartitionEnv::HasInOutAxes(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  return expr_inout_axes_.count(expr);
}

Indices PartitionEnv::GetInAxes(Expr expr, Index index) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_inout_axes_.count(expr));
  CHECK(expr_inout_axes_.at(expr).first.size() > index);
  return expr_inout_axes_.at(expr).first.at(index);
}

Indices PartitionEnv::GetArgInAxes(Expr expr, Index index) const {
  // Should use this instead of GetInAxes when fetching in axes.
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_inout_axes_.count(expr));
  CHECK(HasArgVar(expr, index)) << "Expr " << expr << " do not have arg var at index " << index;
  Index var_index = GetArgVar(expr, index);
  CHECK(expr_inout_axes_.at(expr).first.size() > var_index);
  return expr_inout_axes_.at(expr).first.at(var_index);
}

Indices PartitionEnv::GetOutAxes(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_inout_axes_.count(expr));
  return expr_inout_axes_.at(expr).second;
}

// partition env

Expr PartitionEnv::GetExprFromVar(Var var) const {
  if (sched_dfg.var_expr_map.count(var)) {
    return sched_dfg.var_expr_map.at(var);
  } else if (func_var_expr_.count(var)) {
    return func_var_expr_.at(var);
  }
  LOG(FATAL) << "Not found corresponding expr of var " << ir::AsText(var);
  return Expr{nullptr};
}

Var PartitionEnv::GetVarFromExpr(Expr expr) {
  expr = GetAndCheckExpr_(expr);
  if (sched_dfg.expr_var_map.count(expr)) {
    return sched_dfg.expr_var_map.at(expr);
  } else if (func_expr_var_.count(expr)) {
    return func_expr_var_.at(expr);
  }
  // LOG(WARNING) << "Not found corresponding var of expr " << ir::AsText(expr);
  return Var{nullptr};
}

// partition instr

PartitionInstr PartitionEnv::GetPartitionInstr(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_instr_.count(expr));
  return expr_instr_.at(expr);
}

// func exprs

Exprs PartitionEnv::GetFuncExprsFromCaller(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  CHECK(expr_func_exprs_.count(expr));
  return expr_func_exprs_.at(expr);
}

Exprs PartitionEnv::GetAllFuncExprs() const {
  Exprs result;
  for(auto it: expr_func_exprs_) {
    for(auto expr: it.second) {
      result.push_back(expr);
    }
  }
  return result;
}

// schedule location

void PartitionEnv::SetScheduleLocation(Expr expr, ScheduleLocation loc) {
  expr = GetAndCheckExpr_(expr);
  partitioned_expr_schedule_loc_[expr] = loc;
}

void PartitionEnv::RemoveScheduleLocation(Expr expr) {
  expr = GetAndCheckExpr_(expr);
  if(partitioned_expr_schedule_loc_.count(expr)) {
    partitioned_expr_schedule_loc_.erase(expr);
  }
}

ExprMap<ScheduleLocation> PartitionEnv::GetScheduleLocations() const {
  return partitioned_expr_schedule_loc_;
}

MoENodeLabel PartitionEnv::GetMoENodeLabel(Expr expr) const {
  expr = GetAndCheckExpr_(expr);
  auto node = sched_dfg.dfg.getNodeFromExpr(expr);
  if (moe_label_map_.count(node)) {
    return moe_label_map_.at(node);
  }
  return {MoENodeType::kUnknown, -1};
}

// remove expr

void PartitionEnv::RemoveExpr(Expr expr) {
  expr = GetAndCheckExpr_(expr);
  CHECK_AND_REMOVE(partitioned_expr_schedule_loc_, expr);
  CHECK_AND_REMOVE(expr_axes_exprs_, expr);
  CHECK_AND_REMOVE(expr_inout_axes_, expr);
  CHECK_AND_REMOVE(expr_arg_var_, expr);
  CHECK_AND_REMOVE(expr_arg_consumer_index_, expr);
  CHECK_AND_REMOVE(expr_index_updated_arg_, expr);
  CHECK_AND_REMOVE(expr_instr_, expr);
  CHECK_AND_REMOVE(func_expr_var_, expr);
  CHECK_AND_REMOVE(func_var_expr_, expr);
  CHECK_AND_REMOVE(expr_func_exprs_, expr);
}

// synchronization

void PartitionEnv::AddSynchronization(Expr input_expr, Expr expr, Indices axes) {
  // LOG(INFO) << "Add synchronization.";
  expr = GetAndCheckExpr_(expr);
  if (!HasPartitionedExprs(input_expr, axes)) {
    // Split input expr.
    CHECK(input_expr->IsInstance<VarNode>()) << "Encountered non var input to partition in ANF: input: " << input_expr << ", expr: " << expr;
    auto input_var = Downcast<Var>(input_expr);
    PartitionInstr instr = GetPartitionInstr(expr);
    PartitionedANFBlock split_exprs = SplitVar(*this, input_var, axes, instr);
    SetPartitionedExprs(input_expr, axes, split_exprs);
  }
}

void PartitionEnv::AccumulateProfileElapsedTime(const std::chrono::milliseconds::rep& time) {
  profile_total_ += time;
}

std::chrono::milliseconds::rep PartitionEnv::GetProfileElapsedTime() const {
  return profile_total_;
}

int PartitionEnv::GetDPGroupSize() const {
  return dp_group_size_;
}

// Input expr can be let, var and value.
// This function extracts value expr from let, or find the corresponding value
// if the input expr is a var.
Expr PartitionEnv::GetAndCheckExpr_(Expr expr) const {
  expr = TryGetLetVar(expr);
  if (expr->IsInstance<VarNode>()) {
    auto var = Downcast<Var>(expr);
    if (sched_dfg.var_expr_map.count(var)) {
      expr = sched_dfg.var_expr_map.at(var);
    } else if (func_var_expr_.count(var)) {
      expr = func_var_expr_.at(var);
    }
  }
  return expr;
}

void PartitionEnv::SetAxesExprs_(int n_part) {
  expr_axes_exprs_.clear();
  // for all exprs that do not require partitioning, we directly set their
  // partitioned exprs to themselves.
  for (auto& it : expr_inout_axes_) {
    auto& axes = it.second.second;
    if (AllEqual(axes, SpecialAxis::kNone)) {
      // Set not partitioned exprs.
      auto expr = it.first;
      auto var = GetVarFromExpr(expr);
      PartitionedANFBlock pblock;
      if (var.get() != nullptr) {
        pblock = PartitionedANFBlock::MakeIdentity(var, expr, n_part);
      } else if (expr->IsInstance<VarNode>()){
        // Function parameters don't have a corresponding expr, use an empty expr instead.
        pblock = PartitionedANFBlock::MakeIdentity(Downcast<Var>(expr), Expr{nullptr}, n_part);
      } else {
        // constants?
        pblock = PartitionedANFBlock::MakeIdentity(Var{nullptr}, expr, n_part);
      }
      SetPartitionedExprs(expr, axes, pblock);
    }
  }
  // Already contains func exprs when calling SetAxesExprs_
}

void PartitionEnv::SetFuncExprs_() {
  for (auto& it : expr_func_exprs_) {
    auto& func_exprs = it.second;
    auto& func_call = it.first;
    for (auto& func_expr : func_exprs) {
      auto var = Downcast<Var>(TryGetLetVar(func_expr));
      auto expr = TryGetLetValue(func_expr);
      func_expr_var_[expr] = var;
      func_var_expr_[var] = expr;
      // node of func call has already been deleted, here we just add nodes of func exprs.
      sched_dfg.insertNewCompExprAndAddInputEdges(var, expr, ExtractArgs(expr), GetExprName(expr), 0);
    }
  }
}

void PartitionEnv::SetInstr_(const ExtendedDFG& dfg, int dp_group_size, int n_part) {
  // all exprs to partition
  for (auto& it : expr_inout_axes_) {
    auto& expr = it.first;
    if (!expr_instr_.count(expr)) {
      expr_instr_[expr] = PartitionInstr{n_part};
    }
  }
}

// FPartition creates partitioned exprs using partitioned varaibles of arguments,
// partitioned variables of return value and partition axes.
using FPartition = std::function<void(PartitionEnv&, Expr, bool)>;
using FPartitionMap = std::unordered_map<std::string, FPartition>;

CommComponents partitionCommComponents(CommComponents size, int number_of_partitions) {
    CommComponents partitioned_comm_component;
    for(auto it: size) {
      partitioned_comm_component[it.first] = it.second / number_of_partitions;
      if (it.second % number_of_partitions != 0) {
        // LOG(WARNING) << "Try to partition CommComponents when the tensor size is not a multiple of number of partitions: " << it.second << " v.s. " << number_of_partitions;
        partitioned_comm_component[it.first] += 1;
      }
    }
    return partitioned_comm_component;
}

FPartition CreateDefaultPartition(std::string dialect_name, ArgIndicesGetter getter) {
  // Partition arguments given in arg_indices.
  // Only support all arguments in arg_indices are tensors.
  FPartition default_partition = [=](PartitionEnv& env, Expr expr, bool skip_profile) {
    Indices arg_indices = getter(expr);

    auto var = env.GetVarFromExpr(expr);
    Indices out_axes = env.GetOutAxes(expr);
    PartitionInstr instr = env.GetPartitionInstr(expr);
    int n_part = AllEqual(out_axes, SpecialAxis::kNone) ? 1 : instr.n_part;

    Call call = GetOpCall(expr, dialect_name);
    auto valid_arg_indices = GetValidIndices(call, arg_indices);
    IndexMap<VarAndExprs> arg_partitioned_exprs = {};
    for (int i = 0; i < valid_arg_indices.size(); ++i) {
      Index arg_index = valid_arg_indices[i];
      auto arg = call->args[arg_index];
      if(env.HasArgVar(expr, arg_index)) {
        Indices arg_axes = env.GetArgInAxes(expr, arg_index);
        env.AddSynchronization(arg, expr, arg_axes);
        VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);
        arg_partitioned_exprs[arg_index] = partitioned_args;
        CHECK(n_part == partitioned_args.size()) << "Inconsistent number of partition.";
      }
    }
    Type partitioned_type;
    if (out_axes[0] == SpecialAxis::kNone || out_axes[0] == SpecialAxis::kExpert) {
      partitioned_type = expr->checked_type_;
    } else {
      partitioned_type = GetSplitFieldType(expr->checked_type_, out_axes, n_part);
    }

    PartitionedANFBlock partitioned_exprs;
    SimulateTimeType partitioned_exec_time = 0;

    const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
    CHECK(node) << "Failed to get the corresponding node for expr " << expr;
    auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
    for (int i = 0; i < n_part; ++i) {
      Array<Expr> part_i_args;
      for (int j = 0; j < call->args.size(); ++j) {
        if (arg_partitioned_exprs.count(j)) {
          part_i_args.push_back(arg_partitioned_exprs[j][i].var);
        } else {
          env.SetUpdatedArg(expr, j, call->args[j]);
          part_i_args.push_back(call->args[j]);
        }
      }
      Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
      part_i_expr->checked_type_ = partitioned_type;
      Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
      auto partitioned_full_name = GetPartitionedName(node_full_name, i);

      partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
      env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));

      bool is_comm_expr = IsCollectiveOp(call->op);
      if(is_comm_expr) {
        // create new comm size
        auto partitioned_comm_size = partitionCommComponents(env.sched_dfg.dfg.getCommSize(node), n_part);
        if (partitioned_exec_time == 0 && !skip_profile) {
          partitioned_exec_time = env.op_profiler.GetCommOpExecTime(partitioned_comm_size);
        }
        env.sched_dfg.insertNewCommExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, partitioned_full_name, 
                                                        partitioned_exec_time, partitioned_comm_size);
      } else {
        if (partitioned_exec_time == 0 && !skip_profile) {
          std::chrono::milliseconds::rep elapsed_time_;
          std::tie(partitioned_exec_time, elapsed_time_) = env.op_profiler.GetCompOpExecTime(part_i_expr);
          env.AccumulateProfileElapsedTime(elapsed_time_);
        }
        env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, 
                                                        partitioned_full_name, partitioned_exec_time);
      }
    }
    env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
  };
  return default_partition;
}

FPartition CreateDefaultPartition(std::string dialect_name, Indices arg_indices) {
  // The default function should partitions all args in arg_indices.
  ArgIndicesGetter getter = [=](const Expr& expr) -> Indices {
    return arg_indices;
  };
  return CreateDefaultPartition(dialect_name, getter);
}

FPartition CreateDefaultPartition(std::string dialect_name) {
  // The default function should partitions all args.
  ArgIndicesGetter getter = [=](const Expr& expr) -> Indices {
    Call call = GetOpCall(expr, dialect_name);
    Indices arg_indices = {};
    for (int i = 0; i < call->args.size(); ++i) {
      arg_indices.push_back(i);
    }
    return arg_indices;
  };
  return CreateDefaultPartition(dialect_name, getter);
}

FPartition CreateDefaultPartition(std::string dialect_name, int n_arg) {
  // The default function should partitions first n_arg args.
  Indices indices = {};
  for (int i = 0; i < n_arg; ++i) {
    indices.push_back(i);
  }
  return CreateDefaultPartition(dialect_name, indices);
}

FPartition PartitionMatMulNT = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  static const auto sparse_mm_nt_op = op::GetDialectOp("raf.op.cublas.sparse_expert_matmul_nt");
  static const auto reshape_op = op::GetDialectOp("raf.op.tvm.reshape");
  auto var = env.GetVarFromExpr(expr);
  Indices out_axes = env.GetOutAxes(expr);
  PartitionInstr instr = env.GetPartitionInstr(expr);
  int n_part = AllEqual(out_axes, SpecialAxis::kNone) ? 1 : instr.n_part;

  Call call = GetCall(expr);
  CHECK(call->args.size() == 2);
  MoENodeLabel label = env.GetMoENodeLabel(expr);
  IndexMap<VarAndExprs> arg_partitioned_exprs = {};
  if (out_axes[0] == SpecialAxis::kExpert && label.type == MoENodeType::kMoEExperts) {
    // special case where we replace matmul with the unpadded version
    for (int i = 0; i < 2; ++i) {
      auto arg = call->args[i];
      if(env.HasArgVar(expr, i)) {
        Indices arg_axes = env.GetArgInAxes(expr, i);
        env.AddSynchronization(arg, expr, arg_axes);
        VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);
        arg_partitioned_exprs[i] = partitioned_args;
        CHECK(n_part == partitioned_args.size()) << "Inconsistent number of partition.";
      }
    }
    Type orig_type = expr->checked_type_;
    Type partitioned_type = expr->checked_type_;
    PartitionedANFBlock partitioned_exprs;

    const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
    CHECK(node) << "Failed to get the corresponding node for expr " << expr;
    SimulateTimeType partitioned_exec_time = env.sched_dfg.dfg.getNodeExecTime(node) / n_part;
    auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);

    // now get nearest alltoall
    auto nearest_alltoall = findNearestOpWithName(env, expr, {"raf.op.nccl._all_to_all"});
    CHECK(nearest_alltoall.defined()) << "Failed to find nearest alltoall for expr " << expr;
    Indices predecessor_partition_axes = env.GetOutAxes(nearest_alltoall);
    auto partitioned_predecessor_vars_exprs = env.GetPartitionedOutputs(nearest_alltoall, predecessor_partition_axes);
    CHECK(partitioned_predecessor_vars_exprs.size() == n_part) << "Inconsistent number of partition.";
    CHECK(partitioned_predecessor_vars_exprs[0].expr->IsInstance<TupleGetItemNode>());
    // for alltoallv, GetPartitionedOutputs will get the data output. we need to look one slot before
    auto pred_partitioned_exprs = env.GetPartitionedExprs(nearest_alltoall, predecessor_partition_axes);
    VarAndExprs alltoallv_outputs;
    for (int i=0; i< pred_partitioned_exprs.partitions.size(); i++) {
      auto& vaes = pred_partitioned_exprs.partitions[i];
      int out_index = pred_partitioned_exprs.partition_output_indices[i];
      CHECK_GT(out_index, 0);
      alltoallv_outputs.push_back(vaes[out_index - 1]);
    }
    partitioned_predecessor_vars_exprs = alltoallv_outputs;

    std::vector<int64_t> dense_partitioned_shape = GetShapes(partitioned_type);
    for (int i = 0; i < n_part; ++i) {
      // first reshape to C x G x M (if not already in this shape)
      auto part_i_args = arg_partitioned_exprs[0][i];
      TensorType arg_type = Downcast<TensorType>(part_i_args.expr->checked_type_);
      std::vector<int64_t> arg_shape = GetShapes(arg_type);
      Var arg_var = part_i_args.var;
      Expr arg_expr = part_i_args.expr;
      if (arg_type->shape.size() != 3) {
        CHECK_EQ(arg_type->shape.size(), 2); // [C*G, M]
        CHECK_EQ(arg_shape[0] % env.GetDPGroupSize(), 0);
        int dim_c = arg_shape[0] / env.GetDPGroupSize();
        int dim_g = arg_shape[0] / dim_c;
        TensorType reshaped_type = TensorType({Integer(dim_c), Integer(dim_g), Integer(arg_shape[1])}, arg_type->dtype);
        Array<Integer> new_shape = {Integer(dim_c), Integer(dim_g), Integer(arg_shape[1])};
        Expr reshaped_arg = Call(reshape_op, {part_i_args.var, MakeConstant(ArrayToIntTuple(new_shape)),
                            MakeConstant(BoolValue::make(false))});
        reshaped_arg->checked_type_ = reshaped_type;
        Var reshaped_var = MakeTypeCheckedVar(GetPartitionedName("reshaped_arg", i), reshaped_type);
        partitioned_exprs.PushExprsToPartition(reshaped_var, reshaped_arg, i);
        env.SetScheduleLocation(reshaped_arg, ScheduleAtPipelineIndex(i));
        env.sched_dfg.insertNewCompExprAndAddInputEdges(reshaped_var, reshaped_arg, {part_i_args.var},
                                                        GetPartitionedName(node_full_name + "_reshape", i), 0);
        arg_var = reshaped_var;
        arg_expr = reshaped_arg;
        partitioned_type = TensorType({Integer(dim_c), Integer(dim_g), Integer(dense_partitioned_shape[1])}, arg_type->dtype);
      }
      // first get recv_cnt from tuple
      auto recv_element_type = Downcast<TupleType>(partitioned_predecessor_vars_exprs[i].expr->checked_type_)->fields[1];
      Expr part_i_recv_cnt_expr = TupleGetItem(partitioned_predecessor_vars_exprs[i].var, 1);
      part_i_recv_cnt_expr->checked_type_ = recv_element_type;
      Var part_i_recv_cnt_var = MakeTypeCheckedVar(GetPartitionedName("recv_cnt", i), recv_element_type);
      partitioned_exprs.PushExprsToPartition(part_i_recv_cnt_var, part_i_recv_cnt_expr, i);
      env.SetScheduleLocation(part_i_recv_cnt_expr, ScheduleAtPipelineIndex(i));
      env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_recv_cnt_var, part_i_recv_cnt_expr, 
                                                      {partitioned_predecessor_vars_exprs[i].var}, 
                                                      GetPartitionedName("recv_cnt", i), 0);
      // now get stride and offset for recv_cnt
      // stride = number of local experts
      // offset = local expert id of the current op
      int recv_cnt_dim0 = Downcast<TensorType>(recv_element_type)->shape[0].as<IntImmNode>()->value;
      CHECK_EQ(recv_cnt_dim0 % env.GetDPGroupSize(), 0);
      int stride = recv_cnt_dim0 / env.GetDPGroupSize();
      CHECK_GT(stride, 0);
      auto nearest_tgi = findNearestTGI(env, expr);
      CHECK(nearest_tgi.defined());
      int local_expert_id = Downcast<TupleGetItem>(nearest_tgi)->index;
      // some extra check for debug (not needed)
      int n_local_experts = Downcast<TupleType>(Downcast<TupleGetItem>(nearest_tgi)->tuple->checked_type_)->fields.size();
      CHECK_EQ(n_local_experts, stride);
      // furthermore, if we are the second mm_nt, we need to change the scale of recv_cnt since
      // dim_M has already been scaled up. We get this scale by looking at the shape of the orig op
      auto arg_var_shape = GetShapes(arg_var->checked_type_);
      int scale = arg_var_shape[2] / dense_partitioned_shape[1];
      scale = std::max(scale, 1);
      Expr matmul_expr = Call(sparse_mm_nt_op, {arg_var, arg_partitioned_exprs[1][i].var, part_i_recv_cnt_var,
                                                MakeConstant(ScalarValue::make(local_expert_id)),
                                                MakeConstant(ScalarValue::make(stride)),
                                                MakeConstant(ScalarValue::make(scale))
                                                }, call->attrs, call->type_args);
      matmul_expr->checked_type_ = partitioned_type;
      Var matmul_var = MakeTypeCheckedVar(GetPartitionedName(node_full_name, i), partitioned_type);
      partitioned_exprs.PushExprsToPartition(matmul_var, matmul_expr, i);
      env.SetScheduleLocation(matmul_expr, ScheduleAtPipelineIndex(i));
      // if not using sparse matmul, multiply partitioned_exec_time by n_part
      env.sched_dfg.insertNewCompExprAndAddInputEdges(matmul_var, matmul_expr, {arg_var, arg_partitioned_exprs[1][i].var, part_i_recv_cnt_var},
                                                      GetPartitionedName(node_full_name + "_sparse", i), partitioned_exec_time * n_part);
      if (arg_type->shape.size() != 3) {
        // reshape back
        std::vector<int64_t> orig_shape_vec = GetShapes(orig_type);
        Array<Integer> orig_shape = {Integer(arg_shape[0]), Integer(dense_partitioned_shape[1])};
        Expr reshaped_back_expr = Call(reshape_op, {matmul_var, MakeConstant(ArrayToIntTuple(orig_shape)),
                                       MakeConstant(BoolValue::make(false))});
        reshaped_back_expr->checked_type_ = orig_type;
        Var reshaped_back_var = MakeTypeCheckedVar(GetPartitionedName("reshape_back", i), orig_type);
        partitioned_exprs.PushExprsToPartition(reshaped_back_var, reshaped_back_expr, i);
        env.SetScheduleLocation(reshaped_back_expr, ScheduleAtPipelineIndex(i));
        env.sched_dfg.insertNewCompExprAndAddInputEdges(reshaped_back_var, reshaped_back_expr, {matmul_var},
                                                        GetPartitionedName(node_full_name + "_reshape_back", i), 0);
      }
    }
    env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
  } else {
    CreateDefaultPartition("raf.op.cublas.matmul_nt", 2)(env, expr, skip_profile);
  }
};

FPartition PartitionTuple = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  // Tuple fields may have different partition axes. For consecutive partitioned fields,
  // we partition them to new tuples.
  auto var = env.GetVarFromExpr(expr);
  CHECK(expr->IsInstance<TupleNode>());
  Tuple tuple = Downcast<Tuple>(expr);
  PartitionInstr instr = env.GetPartitionInstr(expr);
  PartitionedANFBlock partitioned_exprs;
  // shape: [fields, partitions]
  std::vector<VarAndExprs> partitioned_fields;
  for (int i = 0; i < tuple->fields.size(); ++i) {
    Expr field = tuple->fields[i];
    Indices field_axes = env.GetArgInAxes(expr, i);
    CHECK(field_axes.size() == 1);
    env.AddSynchronization(field, expr, field_axes);
    VarAndExprs field_i_outputs = env.GetPartitionedOutputs(field, field_axes);
    partitioned_fields.emplace_back(field_i_outputs);
  }
  Indices out_axes = env.GetOutAxes(expr);
  // shape: [partitions]
  VarAndExprs tuples = MergeFields(env.sched_dfg, partitioned_fields, out_axes, instr);
  CHECK_EQ(tuples.size(), instr.n_part);
  for(int i=0; i<tuples.size(); i++) {
    partitioned_exprs.PushExprsToPartition(tuples[i].var, tuples[i].expr, i);
    env.SetScheduleLocation(tuples[i].expr, ScheduleAtPipelineIndex(i));
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionTGI = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  // LOG(INFO) << "Partitioning tgi.";
  auto var = env.GetVarFromExpr(expr);
  TupleGetItem tgi = Downcast<TupleGetItem>(expr);
  PartitionInstr instr = env.GetPartitionInstr(expr);
  int n_part = instr.n_part;
  Expr tuple = tgi->tuple;
  Indices tuple_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(tuple, expr, tuple_axes);
  VarAndExprs tuples = env.GetPartitionedOutputs(tuple, tuple_axes);
  Indices out_axes = env.GetOutAxes(expr);
  PartitionedANFBlock partitioned_exprs;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Expr part_i_expr = TupleGetItem(tuples[i].var, tgi->index);
    part_i_expr->checked_type_ = Downcast<TupleType>(tuples[i].var->checked_type_)->fields[tgi->index];
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), part_i_expr->checked_type_);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, {tuples[i].var}, partitioned_full_name, 0);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionReshape = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  auto var = env.GetVarFromExpr(expr);
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.tvm.reshape");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  TensorType partitioned_type;
  if (out_axes[0] == SpecialAxis::kNone || out_axes[0] == SpecialAxis::kExpert) {
    partitioned_type = Downcast<TensorType>(expr->checked_type_);
  } else {
    partitioned_type = Downcast<TensorType>(GetSplitFieldType(expr->checked_type_, out_axes[0], n_part));
  }

  Expr shape = CreateShapeFromTensorType(partitioned_type);
  env.SetUpdatedArg(expr, 1, shape);
  PartitionedANFBlock partitioned_exprs;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_args[i].var, shape};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args,
                                                    partitioned_full_name, 0);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionSplit = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  auto var = env.GetVarFromExpr(expr);
  // out would be a tuple.
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.tvm.split");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  
  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionSplit on a not partitioned split op.";
  auto orig_split_out_type = Downcast<TupleType>(expr->checked_type_);
  TensorType partitioned_field_type = Downcast<TensorType>(GetSplitFieldType(orig_split_out_type->fields[0], out_axes[0], n_part));
  Array<Type> partitioned_field_types;
  for(int i=0; i< orig_split_out_type->fields.size(); i++) {
    partitioned_field_types.push_back(partitioned_field_type);
  }
  TupleType partitioned_type(partitioned_field_types);
  TensorType arg_type = Downcast<TensorType>(partitioned_args[0].var->checked_type_);
  Expr n_split = call->args[1];
  env.SetUpdatedArg(expr, 1, n_split);
  Expr split_axis;
  if (IsEqualShape(arg_type, partitioned_field_type)) {
    split_axis = call->args[2];
  } else {
    split_axis = FindMisMatchedAxis(arg_type, partitioned_field_type);
  }
  env.SetUpdatedArg(expr, 2, split_axis);
  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_args[i].var, n_split, split_axis};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    // split is comp op
    if (partitioned_exec_time == 0 && !skip_profile) {
      std::chrono::milliseconds::rep elapsed_time_;
      std::tie(partitioned_exec_time, elapsed_time_) = env.op_profiler.GetCompOpExecTime(part_i_expr);
      env.AccumulateProfileElapsedTime(elapsed_time_);
    }
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args,
                                                    partitioned_full_name, partitioned_exec_time);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionConcatenate = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  auto var = env.GetVarFromExpr(expr);
  // out would be a tensor.
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.tvm.concatenate");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  // LOG(INFO) << "Adding synchronization on args completes.";
  
  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionConcatenate on a not partitioned concatenate op.";
  auto orig_concat_out_type = Downcast<TensorType>(expr->checked_type_);
  TensorType partitioned_type = Downcast<TensorType>(GetSplitFieldType(orig_concat_out_type, out_axes[0], n_part));
  Expr concat_axis = call->args[1];
  env.SetUpdatedArg(expr, 1, concat_axis);
  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_args[i].var, concat_axis};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    // concatenate is comp op
    if (partitioned_exec_time == 0 && !skip_profile) {
      std::chrono::milliseconds::rep elapsed_time_;
      try {
        std::tie(partitioned_exec_time, elapsed_time_) = env.op_profiler.GetCompOpExecTime(part_i_expr);
      } catch (...) {
        LOG(FATAL) << "Failed to profile expr " << part_i_expr << ", orig expr is " << expr << ", out axis is: " << out_axes[0];
      }
      env.AccumulateProfileElapsedTime(elapsed_time_);
    }
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args,
                                                    partitioned_full_name, partitioned_exec_time);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionLayerNormDx = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  // partition input 0, 2 and output 0 at the batch axis
  auto var = env.GetVarFromExpr(expr);
  // out would be a tuple.
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.tvm.layer_norm_dx");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto x_arg = call->args[0];
  Indices x_arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(x_arg, expr, x_arg_axes);
  VarAndExprs partitioned_x_args = env.GetPartitionedOutputs(x_arg, x_arg_axes);

  auto dy_arg = call->args[2];
  Indices dy_arg_axes = env.GetArgInAxes(expr, 2);
  env.AddSynchronization(dy_arg, expr, dy_arg_axes);
  VarAndExprs partitioned_dy_args = env.GetPartitionedOutputs(dy_arg, dy_arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_x_args.size() == n_part) << "Inconsistent number of partition.";
  CHECK(partitioned_dy_args.size() == n_part) << "Inconsistent number of partition.";

  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionLayerNormDx on a not partitioned layer_norm_dx op.";
  auto orig_layer_norm_dx_out_type = expr->checked_type_;
  // only the first output field will be split, we manually supplement the rest
  Array<Type> tuple_type_fields;
  auto tuple_type = Downcast<TupleType>(orig_layer_norm_dx_out_type);
  // first field is split
  tuple_type_fields.push_back(GetSplitFieldType(tuple_type->fields[0], out_axes[0], n_part));
  // rest fields are not split
  for (int i = 1; i < tuple_type->fields.size(); ++i) {
    tuple_type_fields.push_back(tuple_type->fields[i]);
  }
  auto partitioned_type = TupleType(tuple_type_fields);
  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  env.SetUpdatedArg(expr, 1, call->args[1]);
  env.SetUpdatedArg(expr, 3, call->args[3]);
  env.SetUpdatedArg(expr, 4, call->args[4]);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_x_args[i].var, call->args[1], partitioned_dy_args[i].var, call->args[3], call->args[4]};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    // layer_norm_dx is comp op
    if (partitioned_exec_time == 0 && !skip_profile) {
      std::chrono::milliseconds::rep elapsed_time_;
      std::tie(partitioned_exec_time, elapsed_time_) = env.op_profiler.GetCompOpExecTime(part_i_expr);
      env.AccumulateProfileElapsedTime(elapsed_time_);
    }
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args,
                                                    partitioned_full_name, partitioned_exec_time);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionAllreduce = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  auto var = env.GetVarFromExpr(expr);
  // out would be a tuple.
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.nccl._allreduce");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionAllreduce on a not partitioned allreduce op.";
  auto orig_allreduce_out_type = expr->checked_type_;
  auto partitioned_type = GetSplitFieldType(orig_allreduce_out_type, out_axes, n_part);
  Expr computation = MakeConstant(StringValue::make("sum"));
  if (call->args.size() == 2) {
    computation = call->args[1];
  }
  env.SetUpdatedArg(expr, 1, computation);

  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_args[i].var, computation};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    // allreduce is comm op
    auto partitioned_comm_size = partitionCommComponents(env.sched_dfg.dfg.getCommSize(node), n_part);
    if (partitioned_exec_time == 0 && !skip_profile) {
      // create new comm size
      partitioned_exec_time = env.op_profiler.GetCommOpExecTime(partitioned_comm_size);
    }
    env.sched_dfg.insertNewCommExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, partitioned_full_name, 
                                                partitioned_exec_time, partitioned_comm_size);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionReduceScatter = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  // LOG(INFO) << "Partition reduce_scatter.";
  auto var = env.GetVarFromExpr(expr);
  // out would be a tuple.
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.nccl._reduce_scatter");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionReduceScatter on a not partitioned reduce_scatter op.";
  auto orig_reduce_scatter_out_type = expr->checked_type_;
  auto partitioned_type = GetSplitFieldType(orig_reduce_scatter_out_type, out_axes, n_part);
  Expr shapes = MakeConstant(ArrayToIntTuple(GetShapes(partitioned_type)));
  Expr shape_indices = call->args[2];
  Expr computation = MakeConstant(StringValue::make("sum"));
  if (call->args.size() == 4) {
    computation = call->args[3];
  }
  env.SetUpdatedArg(expr, 1, shapes);
  env.SetUpdatedArg(expr, 2, shape_indices);
  env.SetUpdatedArg(expr, 3, computation);

  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_args[i].var, shapes, shape_indices, computation};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    // reduce_scatter is comm op
    auto partitioned_comm_size = partitionCommComponents(env.sched_dfg.dfg.getCommSize(node), n_part);
    if (partitioned_exec_time == 0 && !skip_profile) {
      // create new comm size
      partitioned_exec_time = env.op_profiler.GetCommOpExecTime(partitioned_comm_size);
    }
    env.sched_dfg.insertNewCommExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, partitioned_full_name, 
                                                partitioned_exec_time, partitioned_comm_size);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionAllgather = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  // LOG(INFO) << "Partition allgather.";
  auto var = env.GetVarFromExpr(expr);
  // out would be a tuple.
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.nccl._allgather");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionAllgather on a not partitioned allgather op.";
  auto orig_allgather_out_type = expr->checked_type_;
  auto partitioned_type = GetSplitFieldType(orig_allgather_out_type, out_axes, n_part);
  Expr axis = MakeConstant(ScalarValue::make(0));
  if (call->args.size() == 2) {
    axis = call->args[1];
  }
  env.SetUpdatedArg(expr, 1, axis);

  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args = {partitioned_args[i].var, axis};
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    // allgather is comm op
    auto partitioned_comm_size = partitionCommComponents(env.sched_dfg.dfg.getCommSize(node), n_part);
    if (partitioned_exec_time == 0 && !skip_profile) {
      // create new comm size
      partitioned_exec_time = env.op_profiler.GetCommOpExecTime(partitioned_comm_size);
    }
    env.sched_dfg.insertNewCommExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, partitioned_full_name, 
                                                partitioned_exec_time, partitioned_comm_size);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionMoEEncode = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  auto var = env.GetVarFromExpr(expr);
  Indices out_axes = env.GetOutAxes(expr);
  PartitionInstr instr = env.GetPartitionInstr(expr);
  int n_part = AllEqual(out_axes, SpecialAxis::kNone) ? 1 : instr.n_part;

  Call call = GetOpCall(expr, "raf.op.cuda.moe_encode");
  auto valid_arg_indices = GetValidIndices(call, {0, 1, 2});
  IndexMap<VarAndExprs> arg_partitioned_exprs = {};
  for (int i = 0; i < valid_arg_indices.size(); ++i) {
    Index arg_index = valid_arg_indices[i];
    auto arg = call->args[arg_index];
    if(env.HasArgVar(expr, arg_index)) {
      Indices arg_axes = env.GetArgInAxes(expr, arg_index);
      env.AddSynchronization(arg, expr, arg_axes);
      VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);
      arg_partitioned_exprs[arg_index] = partitioned_args;
      CHECK(n_part == partitioned_args.size()) << "Inconsistent number of partition.";
    }
  }
  // GetSplitFieldType already handles kExpert and kIndicesLocations
  Type partitioned_type;
  partitioned_type = GetSplitFieldType(expr->checked_type_, out_axes, n_part);

  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
  Expr last_used_capacity = arg_partitioned_exprs[2][0].var;
  if (!last_used_capacity.defined()) {
    // last_used_capacity is folded as a constant
    last_used_capacity = arg_partitioned_exprs[2][0].expr;
  }
  Type last_used_capacity_type = last_used_capacity->checked_type_;
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args;
    for (int j = 0; j < call->args.size(); ++j) {
      if (j == 2) {
        // used_capacity
        part_i_args.push_back(last_used_capacity);
      } else if (j == 3) {
        // need to multiply capacity_factor by n_part
        auto capacity_factor = call->args[j].as<ConstantNode>();
        CHECK_NOTNULL(capacity_factor);
        float new_capacity_factor_float = Downcast<FloatValue>(capacity_factor->value)->value;
        auto new_capacity_factor = MakeConstant(ScalarValue::make(new_capacity_factor_float * n_part));
        part_i_args.push_back(new_capacity_factor);
        env.SetUpdatedArg(expr, j, new_capacity_factor);
      } else {
        if (arg_partitioned_exprs.count(j)) {
          part_i_args.push_back(arg_partitioned_exprs[j][i].var);
        } else {
          env.SetUpdatedArg(expr, j, call->args[j]);
          part_i_args.push_back(call->args[j]);
        }
      }
    }
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));

    if (partitioned_exec_time == 0 && !skip_profile) {
      std::chrono::milliseconds::rep elapsed_time_;
      std::tie(partitioned_exec_time, elapsed_time_) = env.op_profiler.GetCompOpExecTime(part_i_expr);
      env.AccumulateProfileElapsedTime(elapsed_time_);
    }
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, 
                                                    partitioned_full_name, partitioned_exec_time);
    // get last used capacity
    Expr last_used_capacity_expr = TupleGetItem(part_i_var, 2);
    last_used_capacity_expr->checked_type_ = last_used_capacity_type;
    auto last_used_capacity_name = GetPartitionedName("last_used_capacity", i);
    Var last_used_capacity_var = MakeTypeCheckedVar(last_used_capacity_name, last_used_capacity_type);
    partitioned_exprs.PushExprsToPartition(last_used_capacity_var, last_used_capacity_expr, i, /*is_output=*/false);
    env.SetScheduleLocation(last_used_capacity_expr, ScheduleAtPipelineIndex(i));
    env.sched_dfg.insertNewCompExprAndAddInputEdges(last_used_capacity_var, last_used_capacity_expr, 
                                                    {part_i_var}, last_used_capacity_name, 0);
    last_used_capacity = last_used_capacity_var;
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionMoEEncodeBPR = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  auto var = env.GetVarFromExpr(expr);
  Indices out_axes = env.GetOutAxes(expr);
  PartitionInstr instr = env.GetPartitionInstr(expr);
  int n_part = AllEqual(out_axes, SpecialAxis::kNone) ? 1 : instr.n_part;

  Call call = GetOpCall(expr, "raf.op.cuda.moe_encode_batch_prioritized");
  auto valid_arg_indices = GetValidIndices(call, {0, 1});
  IndexMap<VarAndExprs> arg_partitioned_exprs = {};
  for (int i = 0; i < valid_arg_indices.size(); ++i) {
    Index arg_index = valid_arg_indices[i];
    auto arg = call->args[arg_index];
    if(env.HasArgVar(expr, arg_index)) {
      Indices arg_axes = env.GetArgInAxes(expr, arg_index);
      env.AddSynchronization(arg, expr, arg_axes);
      VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);
      arg_partitioned_exprs[arg_index] = partitioned_args;
      CHECK(n_part == partitioned_args.size()) << "Inconsistent number of partition.";
    }
  }
  // GetSplitFieldType already handles kExpert and kIndicesLocations
  Type partitioned_type;
  partitioned_type = GetSplitFieldType(expr->checked_type_, out_axes, n_part);

  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;

  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  CHECK(node) << "Failed to get the corresponding node for expr " << expr;
  auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);

  for (int i = 0; i < n_part; ++i) {
    Array<Expr> part_i_args;
    for (int j = 0; j < call->args.size(); ++j) {
      switch(j) {
        case 2: // n_partitions
          part_i_args.push_back(MakeConstant(ScalarValue::make(n_part)));
          break;
        case 3: // partition_id
          part_i_args.push_back(MakeConstant(ScalarValue::make(i)));
          break;
        case 4: { // capacity_factor
          // need to multiply capacity_factor by n_part
          auto capacity_factor = call->args[j].as<ConstantNode>();
          CHECK_NOTNULL(capacity_factor);
          float new_capacity_factor_float = Downcast<FloatValue>(capacity_factor->value)->value;
          auto new_capacity_factor = MakeConstant(ScalarValue::make(new_capacity_factor_float * n_part));
          part_i_args.push_back(new_capacity_factor);
          env.SetUpdatedArg(expr, j, new_capacity_factor);
          break;
        }
        default:
          if (arg_partitioned_exprs.count(j)) {
            part_i_args.push_back(arg_partitioned_exprs[j][i].var);
          } else {
            env.SetUpdatedArg(expr, j, call->args[j]);
            part_i_args.push_back(call->args[j]);
          }
      };
    }
    Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
    part_i_expr->checked_type_ = partitioned_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
    auto partitioned_full_name = GetPartitionedName(node_full_name, i);

    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));

    if (partitioned_exec_time == 0 && !skip_profile) {
      std::chrono::milliseconds::rep elapsed_time_;
      std::tie(partitioned_exec_time, elapsed_time_) = env.op_profiler.GetCompOpExecTime(part_i_expr);
      env.AccumulateProfileElapsedTime(elapsed_time_);
    }
    env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, 
                                                    partitioned_full_name, partitioned_exec_time);
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartition PartitionAllToAll = [](PartitionEnv& env, Expr expr, bool skip_profile) {
  static const auto alltoallv_op = op::GetDialectOp("raf.op.nccl._all_to_allv");
  auto var = env.GetVarFromExpr(expr);
  Indices out_axes = env.GetOutAxes(expr);
  Call call = GetOpCall(expr, "raf.op.nccl._all_to_all");
  HashIndexMap<VarAndExprs> arg_partitioned_exprs = {};

  auto arg = call->args[0];
  Indices arg_axes = env.GetArgInAxes(expr, 0);
  env.AddSynchronization(arg, expr, arg_axes);
  VarAndExprs partitioned_args = env.GetPartitionedOutputs(arg, arg_axes);

  int n_part = out_axes[0] == SpecialAxis::kNone ? 1 : env.GetPartitionInstr(expr).n_part;
  CHECK(partitioned_args.size() == n_part) << "Inconsistent number of partition.";

  CHECK(out_axes[0] != SpecialAxis::kNone) << "Calling PartitionAllToAll on a not partitioned all_to_all op.";
  // two cases
  auto orig_alltoall_type = expr->checked_type_;
  const Node* node = env.sched_dfg.dfg.getNodeFromExpr(expr);
  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;
  if (out_axes[0] == SpecialAxis::kExpert) {
    // replace all_to_all with all_to_allv
    Expr predecessor_expr;

    int tuple_index = -1;
    predecessor_expr = findNearestOpWithName(env, expr, {"raf.op.nccl._all_to_all", "raf.op.cuda.moe_encode", "raf.op.cuda.moe_encode_batch_prioritized"}, /*exclude_self=*/true);
    CHECK(predecessor_expr.defined()) << "Failed to find corresponding moe_encode/all_to_all op for expr " << expr;
    if (IsOp(predecessor_expr, Op::Get("raf.op.moe_encode")) || IsOp(predecessor_expr, Op::Get("raf.op.moe_encode_batch_prioritized"))) {
      tuple_index = 3;
    } else {
      CHECK(IsOp(predecessor_expr, Op::Get("raf.op._all_to_all"))) << "Found unexpected op " << predecessor_expr;
      tuple_index = 1;
    }
    Indices predecessor_partition_axes = env.GetOutAxes(predecessor_expr);
    auto partitioned_predecessor_vars_exprs = env.GetPartitionedOutputs(predecessor_expr, predecessor_partition_axes);
    CHECK(partitioned_predecessor_vars_exprs.size() == n_part) << "Inconsistent number of partition.";
    // we need to check if the previous op is moe_encode or all_to_all, or all_to_allv
    if (IsOp(partitioned_predecessor_vars_exprs[0].expr, Op::Get("raf.op.nccl._all_to_all"))) {
      // this shouldn't happen as kExpert tensors can't be generated
      // which already causes an error
      LOG(FATAL) << "Calling PartitionAllToAll but found previous all_to_all not kExpert partitioned.";
    } else if (partitioned_predecessor_vars_exprs[0].expr->IsInstance<TupleGetItemNode>()) {
      // for alltoallv, GetPartitionedOutputs will get the data output. we need to look one slot before
      auto partitioned_exprs = env.GetPartitionedExprs(predecessor_expr, predecessor_partition_axes);
      VarAndExprs alltoallv_outputs;
      for (int i=0; i< partitioned_exprs.partitions.size(); i++) {
        auto& vaes = partitioned_exprs.partitions[i];
        int out_index = partitioned_exprs.partition_output_indices[i];
        CHECK_GT(out_index, 0);
        alltoallv_outputs.push_back(vaes[out_index - 1]);
      }
      partitioned_predecessor_vars_exprs = alltoallv_outputs;
    }
    Type sent_element_type = Downcast<TupleType>(partitioned_predecessor_vars_exprs[0].var->checked_type_)->fields[tuple_index];
    auto partitioned_type = TupleType(Array<Type>{orig_alltoall_type, sent_element_type});
    for (int i=0; i< n_part; i++) {
      // first get send_cnt from tuple
      Expr part_i_sent_cnt_expr = TupleGetItem(partitioned_predecessor_vars_exprs[i].var, tuple_index);
      part_i_sent_cnt_expr->checked_type_ = sent_element_type;
      Var part_i_sent_cnt_var = MakeTypeCheckedVar(GetPartitionedName("sent_cnt", i), sent_element_type);
      partitioned_exprs.PushExprsToPartition(part_i_sent_cnt_var, part_i_sent_cnt_expr, i);
      env.SetScheduleLocation(part_i_sent_cnt_expr, ScheduleAtPipelineIndex(i));
      env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_sent_cnt_var, part_i_sent_cnt_expr, 
                                                      {partitioned_predecessor_vars_exprs[i].var}, 
                                                      GetPartitionedName("sent_cnt", i), 0);
      // then reconstruct another tuple since alltoallv takes tuples as input...
      Expr part_i_send_cnt_tuple_expr = Tuple({part_i_sent_cnt_var});
      part_i_send_cnt_tuple_expr->checked_type_ = TupleType(Array<Type>{sent_element_type});
      Var part_i_send_cnt_tuple_var = MakeTypeCheckedVar(GetPartitionedName("sent_cnt_tuple", i), part_i_send_cnt_tuple_expr->checked_type_);
      partitioned_exprs.PushExprsToPartition(part_i_send_cnt_tuple_var, part_i_send_cnt_tuple_expr, i);
      env.SetScheduleLocation(part_i_send_cnt_tuple_expr, ScheduleAtPipelineIndex(i));
      env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_send_cnt_tuple_var, part_i_send_cnt_tuple_expr, 
                                                      {part_i_sent_cnt_var}, GetPartitionedName("sent_cnt_tuple", i), 0);
      Expr part_i_alltoallv = Call(alltoallv_op, {partitioned_args[i].var, part_i_send_cnt_tuple_var}, call->attrs, call->type_args, {});
      part_i_alltoallv->checked_type_ = partitioned_type;
      Var part_i_alltoallv_var = MakeTypeCheckedVar(GetPartitionedName("all_to_allv", i), partitioned_type);
      partitioned_exprs.PushExprsToPartition(part_i_alltoallv_var, part_i_alltoallv, i);
      env.SetScheduleLocation(part_i_alltoallv, ScheduleAtPipelineIndex(i));
      auto partitioned_comm_size = partitionCommComponents(env.sched_dfg.dfg.getCommSize(node), n_part);
      if (partitioned_exec_time == 0 && !skip_profile) {
        // use orig time / n_part for now
        partitioned_exec_time = env.op_profiler.GetCommOpExecTime(partitioned_comm_size);
      }
      env.sched_dfg.insertNewCommExprAndAddInputEdges(part_i_alltoallv_var, part_i_alltoallv,
                                                      {partitioned_args[i].var, part_i_send_cnt_tuple_var}, 
                                                      GetPartitionedName("all_to_allv", i), partitioned_exec_time, partitioned_comm_size);
      // also need to get result from the output tuple
      Expr part_i_result_expr = TupleGetItem(part_i_alltoallv_var, 0);
      part_i_result_expr->checked_type_ = orig_alltoall_type;
      Var part_i_result_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), orig_alltoall_type);
      partitioned_exprs.PushExprsToPartition(part_i_result_var, part_i_result_expr, i);
      env.SetScheduleLocation(part_i_result_expr, ScheduleAtPipelineIndex(i));
      env.sched_dfg.insertNewCompExprAndAddInputEdges(part_i_result_var, part_i_result_expr, 
                                                      {part_i_alltoallv_var}, GetPartitionedName(var->name_hint(), i), 0);
    }
  } else {
    // normal partition
    auto partitioned_type = GetSplitFieldType(orig_alltoall_type, out_axes, n_part);
    CHECK(node) << "Failed to get the corresponding node for expr " << expr;
    auto node_full_name = env.sched_dfg.dfg.getNodeNameOrDefault(node);
    for (int i = 0; i < n_part; ++i) {
      Array<Expr> part_i_args = {partitioned_args[i].var};
      Expr part_i_expr = Call(call->op, part_i_args, call->attrs, call->type_args, {});
      part_i_expr->checked_type_ = partitioned_type;
      Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), partitioned_type);
      auto partitioned_full_name = GetPartitionedName(node_full_name, i);

      partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
      env.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
      // alltoall is comm op
      auto partitioned_comm_size = partitionCommComponents(env.sched_dfg.dfg.getCommSize(node), n_part);
      if (partitioned_exec_time == 0 && !skip_profile) {
        // create new comm size
        partitioned_exec_time = env.op_profiler.GetCommOpExecTime(partitioned_comm_size);
      }
      env.sched_dfg.insertNewCommExprAndAddInputEdges(part_i_var, part_i_expr, part_i_args, partitioned_full_name, 
                                                  partitioned_exec_time, partitioned_comm_size);
    }
  }
  env.SetPartitionedExprs(expr, out_axes, partitioned_exprs);
};

FPartitionMap op_partitions = {
  {"raf.op.tvm.add", CreateDefaultPartition("raf.op.tvm.add")},
  {"raf.op.tvm.subtract", CreateDefaultPartition("raf.op.tvm.subtract")},
  {"raf.op.tvm.multiply", CreateDefaultPartition("raf.op.tvm.multiply", 2)},
  {"raf.op.tvm.divide", CreateDefaultPartition("raf.op.tvm.divide", 2)},
  {"raf.op.tvm.bias_add", CreateDefaultPartition("raf.op.tvm.bias_add", 2)},
  {"raf.op.tvm.cast", CreateDefaultPartition("raf.op.tvm.cast", 1)},
  {"raf.op.tvm.squeeze", CreateDefaultPartition("raf.op.tvm.squeeze", 1)},
  {"raf.op.tvm.where", CreateDefaultPartition("raf.op.tvm.where", 3)},
  {"raf.op.tvm.scatter", CreateDefaultPartition("raf.op.tvm.scatter", 3)},
  {"raf.op.tvm.layer_norm", CreateDefaultPartition("raf.op.tvm.layer_norm", 1)},
  {"raf.op.tvm.relu_dx", CreateDefaultPartition("raf.op.tvm.relu_dx", 3)}, // UnaryDx
  {"raf.op.tvm.gelu_dx", CreateDefaultPartition("raf.op.tvm.gelu_dx", 3)},
  {"raf.op.cudnn.tanh_dx", CreateDefaultPartition("raf.op.cudnn.tanh_dx", 3)},
  {"raf.op.tvm.tanh_dx", CreateDefaultPartition("raf.op.tvm.tanh_dx", 3)},
  {"raf.op.tvm.softmax_dx", CreateDefaultPartition("raf.op.tvm.softmax_dx", 3)}, // DeclareGeneralDx
  {"raf.op.tvm.power", CreateDefaultPartition("raf.op.tvm.power", 2)},
  {"raf.op.tvm.transpose", CreateDefaultPartition("raf.op.tvm.transpose", 1)},
  {"raf.op.tvm.transpose_dx", CreateDefaultPartition("raf.op.tvm.transpose_dx", 1)},
  {"raf.op.tvm.one_hot", CreateDefaultPartition("raf.op.tvm.one_hot", 3)},
  {"raf.op.tvm.argmax", CreateDefaultPartition("raf.op.tvm.argmax", 1)},
  {"raf.op.tvm.expand_dims", CreateDefaultPartition("raf.op.tvm.expand_dims", 1)},
  {"raf.op.tvm.embedding", CreateDefaultPartition("raf.op.tvm.embedding", 2)},
  {"raf.op.cudnn.tanh", CreateDefaultPartition("raf.op.cudnn.tanh", 1)},
  {"raf.op.cudnn.relu", CreateDefaultPartition("raf.op.cudnn.relu", 1)},
  {"raf.op.tvm.relu", CreateDefaultPartition("raf.op.tvm.relu", 1)},
  {"raf.op.tvm.gelu", CreateDefaultPartition("raf.op.tvm.gelu", 1)},
  {"raf.op.cudnn.softmax", CreateDefaultPartition("raf.op.cudnn.softmax", 1)},
  {"raf.op.tvm.softmax", CreateDefaultPartition("raf.op.tvm.softmax", 1)},
  {"raf.op.cublas.matmul", CreateDefaultPartition("raf.op.cublas.matmul", 2)},
  {"raf.op.cublas.matmul_tn", CreateDefaultPartition("raf.op.cublas.matmul_tn", 2)},
  // {"raf.op.cublas.matmul_nt", CreateDefaultPartition("raf.op.cublas.matmul_nt", 2)},
  {"raf.op.cublas.matmul_nt", PartitionMatMulNT},
  {"raf.op.cublas.matmul_tt", CreateDefaultPartition("raf.op.cublas.matmul_tt", 2)},
  {"raf.op.cublas.batch_matmul", CreateDefaultPartition("raf.op.cublas.batch_matmul", 2)},
  {"raf.op.cublas.batch_matmul_tn", CreateDefaultPartition("raf.op.cublas.batch_matmul_tn", 2)},
  {"raf.op.cublas.batch_matmul_nt", CreateDefaultPartition("raf.op.cublas.batch_matmul_nt", 2)},
  {"raf.op.cublas.batch_matmul_tt", CreateDefaultPartition("raf.op.cublas.batch_matmul_tt", 2)},
  {"raf.op.cuda.moe_encode", PartitionMoEEncode},
  {"raf.op.cuda.moe_encode_batch_prioritized", PartitionMoEEncodeBPR},
  {"raf.op.cuda.moe_decode", CreateDefaultPartition("raf.op.cuda.moe_decode", 3)},
  // {"raf.op.cuda.moe_encode_dx", CreateDefaultPartition("raf.op.cuda.moe_encode_dx", 3)},
  // {"raf.op.cuda.moe_decode_dx", CreateDefaultPartition("raf.op.cuda.moe_decode_dx", 5)},
  // {"raf.op.cuda.moe_encode_dg", CreateDefaultPartition("raf.op.cuda.moe_encode_dg", 3)},
  // {"raf.op.cuda.moe_decode_dg", CreateDefaultPartition("raf.op.cuda.moe_decode_dg", 4)},
  {"raf.op.tvm.reshape", PartitionReshape},
  {"raf.op.tvm.split", PartitionSplit},
  {"raf.op.tvm.concatenate", PartitionConcatenate},
  {"raf.op.tvm.layer_norm_dx", PartitionLayerNormDx},
  {"raf.op.nccl._all_to_all", PartitionAllToAll},
  {"raf.op.nccl._allreduce", PartitionAllreduce},
  {"raf.op.nccl._reduce_scatter", PartitionReduceScatter},
  {"raf.op.nccl._allgather", PartitionAllgather},
};

bool HasFPartition(std::string op) {
  if (op_partitions.count(op)) {
    return true;
  } else if (base_dialect_op.count(op)) {
    return HasFPartition(base_dialect_op.at(op));
  } else {
    return false;
  }
}

FPartition GetFPartition(std::string op) {
  if (op_partitions.count(op)) {
    return op_partitions.at(op);
  } else if (base_dialect_op.count(op)) {
    return GetFPartition(base_dialect_op.at(op));
  } else {
    LOG(FATAL) << "Cannot find partitioning implementation for op " << op;
    return nullptr;
  }
}

PartitionEnvBuilder::PartitionEnvBuilder(ScheduledDFG& sched_dfg,
                                         ExtendedOpProfiler& op_profiler,
                                         const NodeMap<MoENodeLabel>& moe_label_map,
                                         const CPSolution& solution,
                                         int dp_group_size, int n_part, int n_experts) :
  sched_dfg_(sched_dfg),
  env_(sched_dfg, op_profiler, moe_label_map, solution.expr_inout_axes, solution.expr_func_exprs,
       solution.expr_arg_idx_to_inoutaxes_index, solution.expr_arg_consumer_index,
       dp_group_size, n_part),
  n_experts_(n_experts) {}

PartitionEnv PartitionEnvBuilder::Build(const Exprs& exprs) {
  Setup_();
  for (auto expr : exprs) {
    // all expr are value of let.
    stack_.back() = expr;
    VisitExpr(expr);
  }
  Finalize_(exprs);
  return env_;
}

void PartitionEnvBuilder::VisitExpr_(const LetNode* let) {
  LOG(WARNING) << "Should not have received a let node here.";
  VisitExpr(let->value);
}

void PartitionEnvBuilder::VisitExpr_(const TupleNode* tuple) {
  PartitionTuple(env_, stack_.back(), stack_.size() > 1);
}

void PartitionEnvBuilder::VisitExpr_(const TupleGetItemNode* tgi) {
  PartitionTGI(env_, stack_.back(), stack_.size() > 1);
}

void PartitionEnvBuilder::VisitExpr_(const CallNode* call) {
  VisitExpr(call->op);
}

void PartitionEnvBuilder::VisitExpr_(const OpNode* op) {
  FPartition op_partition = GetFPartition(op->name);
  op_partition(env_, stack_.back(), stack_.size() > 1);
}

void PartitionEnvBuilder::VisitExpr_(const FunctionNode* func) {
  Expr caller = stack_.back();
  Exprs func_exprs = env_.GetFuncExprsFromCaller(caller);

  stack_.push_back(Expr{nullptr});
  for (auto func_expr : func_exprs) {
    // Though here func_expr is let, it would be
    // convert to value of let in GetAndCheckExpr_.
    auto expr = TryGetLetValue(func_expr);
    // LOG(INFO) << "Visiting func value " << ir::AsText(expr);
    stack_.back() = expr;
    VisitExpr(expr);
  }
  stack_.pop_back();
  PartitionFunction_();
}

void PartitionEnvBuilder::VisitExpr(const Expr& expr) {
  // Always allow visiting op and function node.
  if (expr->IsInstance<FunctionNode>() || expr->IsInstance<OpNode>()) {
    ExprFunctor<void(const Expr& e)>::VisitExpr(expr);
  } else {
    ExprVisitor::VisitExpr(expr);
  }
}

void PartitionEnvBuilder::Setup_() {
  stack_.clear();
  stack_.push_back(Expr{nullptr});
}

void PartitionEnvBuilder::PartitionFunction_() {
  // Use dependency graph to obtain exprs in a function in topological order,
  // while all parameters are substituted by actual inputs.
  Expr caller = stack_.back();
  Function orig_func = Downcast<Function>(GetCall(caller)->op);
  auto var = env_.GetVarFromExpr(caller);
  Indices out_axes = env_.GetOutAxes(caller);
  Exprs func_exprs = env_.GetFuncExprsFromCaller(caller);
  auto call_args = GetCall(caller)->args;

  // LOG(INFO) << "Partitioning function " << ir::AsText(orig_func);

  // func exprs are values, not lets.
  std::string param_prefix = "param_";
  Array<Var> params = {};
  Array<Type> param_types = {};
  ExprMap<VarAndExprs> param_partitioned_arg = {};
  ExprMap<Expr> param_updated_arg = {};
  ExprMap<Expr> arg_param = {};
  for (int i = 0; i < call_args.size(); ++i) {
    Expr arg = call_args[i];
    std::string param_name = param_prefix + std::to_string(i);
    auto it = *env_.GetConsumerIndex(caller, arg).begin();
    if (env_.HasArgVar(it.first, it.second)) {
      // Get partitioned VarAndExprs of the arg.
      Indices in_indices = env_.GetArgInAxes(it.first, it.second);
      VarAndExprs partitioned_arg = env_.GetPartitionedOutputs(arg, in_indices);
      Var param = MakeTypeCheckedVar(param_name, partitioned_arg[0].var->checked_type_);
      params.push_back(param);
      param_partitioned_arg[param] = partitioned_arg;
      arg_param[partitioned_arg[0].var] = param;
    } else {
      // Get updated non-tensor / tuple arg.
      Expr updated_arg = env_.GetUpdatedArg(it.first, it.second);
      Var param = MakeTypeCheckedVar(param_name, updated_arg->checked_type_);
      param_types.push_back(updated_arg->checked_type_);
      params.push_back(param);
      param_updated_arg[param] = env_.GetUpdatedArg(it.first, it.second);
      arg_param[updated_arg] = param;
    }
  }

  int n_part = env_.GetPartitionInstr(caller).n_part;
  Vars vars = {};
  Exprs exprs = {};
  std::vector<Exprs> partitioned_func_exprs = {};
  partitioned_func_exprs.resize(n_part);
  for (auto& func_expr : func_exprs) {
    // Get the first partitioned func expr for the partitioned func.
    Indices out_axes_ = env_.GetOutAxes(func_expr);
    VarAndExprs func_exprs_ = env_.GetPartitionedOutputs(func_expr, out_axes_);
    partitioned_func_exprs.push_back(Exprs{});
    // Remove all func exprs.
    env_.RemoveExpr(func_expr);
    for (int i = 0; i < n_part; ++i) {
      partitioned_func_exprs[i].push_back(func_exprs_[i].expr);
      env_.RemoveScheduleLocation(func_exprs_[i].expr);
    }
    VarAndExpr func_expr_ = func_exprs_[0];
    vars.push_back(func_expr_.var);
    exprs.push_back(func_expr_.expr);
  }
  Exprs updated_exprs = ExprUpdater(arg_param, exprs).Update(exprs);

  LetList let_list;
  Expr ret;
  for (int i = 0; i < vars.size(); ++i) {
    ret = let_list.Push(vars[i], updated_exprs[i]);
  }
  Expr updated_body = let_list.Get(ret);
  Type ret_type = ret->checked_type_;
  Function partitioned_func = Function(params, updated_body, ret_type, orig_func->type_params, orig_func->attrs);
  FuncType orig_func_type = Downcast<FuncType>(orig_func->checked_type_);
  partitioned_func->checked_type_ = FuncType(param_types, ret_type, orig_func_type->type_params, orig_func_type->type_constraints);
  // LOG(INFO) << "Partitioned function " << ir::AsText(partitioned_func);

  PartitionedANFBlock partitioned_exprs;
  SimulateTimeType partitioned_exec_time = 0;
  for (int i = 0; i < n_part; ++i) {
    Array<Expr> args = {};
    for (auto param : params) {
      if (param_partitioned_arg.count(param)) {
        args.push_back(param_partitioned_arg[param][i].var);
      } else {
        args.push_back(param_updated_arg[param]);
      }
    }
    Expr part_i_expr = Call(partitioned_func, args);
    part_i_expr->checked_type_ = ret_type;
    Var part_i_var = MakeTypeCheckedVar(GetPartitionedName(var->name_hint(), i), ret_type);
    partitioned_exprs.PushExprsToPartition(part_i_var, part_i_expr, i);
    env_.SetScheduleLocation(part_i_expr, ScheduleAtPipelineIndex(i));
    if(partitioned_exec_time == 0) {
      std::chrono::milliseconds::rep elapsed_time_;
      std::tie(partitioned_exec_time, elapsed_time_) = env_.op_profiler.GetCompOpExecTime(part_i_expr);
      env_.AccumulateProfileElapsedTime(elapsed_time_);
    }
    env_.sched_dfg.insertNewCompExprAndMergeExprs(part_i_var, part_i_expr, partitioned_func_exprs[i], GetPartitionedName(var->name_hint(), i), partitioned_exec_time);
  }
  env_.SetPartitionedExprs(caller, out_axes, partitioned_exprs);
}

void PartitionEnvBuilder::Finalize_(const Exprs& exprs) {
  // here we add appropriate concatenation ops
  //  1. We get the exprs whose output will be consumed by exprs outside the pipeline.
  //  2. We add corresponding concatenation operators for the exprs to their corresponding pblocks

  auto& dfg = sched_dfg_.dfg;
  ExprSet expr_set(exprs.begin(), exprs.end());

  // get func exprs first
  ExprSet func_exprs;
  for(auto func_expr_as_let: env_.GetAllFuncExprs()) {
    auto expr = TryGetLetValue(func_expr_as_let);
    func_exprs.insert(expr);
  }


  Exprs exprs_to_concat;
  for(auto expr: exprs) {
    const Node* node = dfg.getNodeFromExpr(expr);
    auto parent_nodes = dfg.getNonSinkParents(node);
    if(parent_nodes.empty()) {
      // LOG(INFO) << "Adding " << expr << " to exprs_to_concat because parents are empty.";
      exprs_to_concat.push_back(expr);
      continue;
    }
    for(auto parent: parent_nodes) {
      auto parent_expr = dfg.getExprFromNode(parent);
      if(!expr_set.count(parent_expr) && !func_exprs.count(parent_expr)) {
        // LOG(INFO) << "Adding " << expr << " to exprs_to_concat because parnet node " << parent_expr << " is not in expr set.";
        exprs_to_concat.push_back(expr);
        break;
      }
    }
  }

  for(auto orig_expr: exprs_to_concat) {
    auto out_axes = env_.GetOutAxes(orig_expr);
    auto partitioned_vaes = env_.GetPartitionedOutputs(orig_expr, out_axes);
    auto instr = env_.GetPartitionInstr(orig_expr);
    auto concat_pblock = ConcatExpr(env_, partitioned_vaes, orig_expr, out_axes, instr, n_experts_);
    auto partition_pblock = env_.GetPartitionedExprs(orig_expr, out_axes);
    partition_pblock.Merge(concat_pblock);
    // rebind the var corresonding to the orig_expr to the new concatenated one
    Var orig_var = env_.GetVarFromExpr(orig_expr);
    CHECK(orig_var.defined()) << "Orig expr " << orig_expr << " not found in sched_dfg_'s expr_var_map.";
    VarAndExpr concat_output = partition_pblock.GetEpilogueOutputs();
    // completely switch all map values, otherwise the corresponding var will be deleted
    sched_dfg_.var_expr_map[orig_var] = concat_output.expr;
    sched_dfg_.var_expr_map[concat_output.var] = orig_expr;
    sched_dfg_.expr_var_map[concat_output.expr] = orig_var;
    sched_dfg_.expr_var_map[orig_expr] = concat_output.var;

    auto orig_node = sched_dfg_.dfg.getNodeFromExpr(orig_expr);
    auto new_node = sched_dfg_.dfg.getNodeFromExpr(concat_output.expr);
    CHECK(orig_node) << "Cannot get original node for expr " << orig_expr;
    CHECK(new_node) << "Cannot get new node for expr " << orig_expr << ", new expr: " << concat_output.expr;
    for(auto parent: sched_dfg_.dfg.getParents(orig_node)) {
      sched_dfg_.dfg.createEdge(parent, new_node);
    }
    VarAndExpr new_concat_output_vae = {orig_var, concat_output.expr};
    partition_pblock.epilogue[partition_pblock.epilogue.size() - 1] = new_concat_output_vae;
    env_.SetPartitionedExprs(orig_expr, out_axes, partition_pblock);
  }
  //  
  // delete original exprs and func exprs
  for(auto orig_expr: exprs) {
    const Node* orig_node = sched_dfg_.dfg.getNodeFromExpr(orig_expr);
    CHECK(orig_node) << "Failed to find the original node in DFG.";
    sched_dfg_.deleteNode(orig_node);
  }
  for(auto expr: func_exprs) {
    const Node* func_expr_node = sched_dfg_.dfg.getNodeFromExpr(expr);
    CHECK(func_expr_node) << "Failed to find the function expr node in DFG: " << expr;
    sched_dfg_.deleteNode(func_expr_node);
  }
  sched_dfg_.dfg.recalculateSourceAndSink();
}

PartitionResult::PartitionResult(ScheduledDFG&& sched_dfg, ExprMap<ScheduleLocation>&& expr_schedule_locations):
  sched_dfg(sched_dfg), expr_schedule_locations(expr_schedule_locations) {}

std::ostream& operator << (std::ostream &os, const PartitionResult &result) {
  os << "Partition Result:" << std::endl;
  os << "DFG:" << std::endl;
  os << result.sched_dfg.dfg << std::endl;
  os << "Expr schedule locations:" << std::endl;
  for(auto it: result.expr_schedule_locations) {
    os << "\tExpr: " << it.first << ", loc: " << it.second;
  }
  return os;
}

PartitionResult PartitionNodes(const ScheduledDFG& sched_dfg, const Nodes& nodes,
                               const CPSolution& solution,
                               ExtendedOpProfiler& op_profiler, 
                               int dp_group_size, int n_part, int n_experts, const NodeMap<MoENodeLabel>& moe_label_map) {
  // create a copy of sched_dfg;
  auto start = std::chrono::system_clock::now();
  ScheduledDFG new_sched_dfg = sched_dfg;
  auto builder_start = std::chrono::system_clock::now();
  PartitionEnvBuilder builder(new_sched_dfg, op_profiler, moe_label_map, solution, dp_group_size, n_part, n_experts);
  PartitionEnv env = builder.Build(solution.exprs);
  auto end = std::chrono::system_clock::now();
  auto elapsed_time = 
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  auto builder_elapsed_time = 
    std::chrono::duration_cast<std::chrono::microseconds>(end - builder_start).count();
  // LOG(INFO) << "Partitioning nodes takes " << elapsed_time / 1000.0f << " ms.";
  // LOG(INFO) << "\t->PartitionEnvBuilder takes " << builder_elapsed_time / 1000.0f << " ms.";
  // LOG(INFO) << "\t\t->Profiling nodes takes " << env.GetProfileElapsedTime() / 1000.0f << " ms.";
  return PartitionResult(std::move(new_sched_dfg), std::move(env.GetScheduleLocations()));
}

PartitionResult PartitionNodes(const ScheduledDFG& sched_dfg, const NodeSet& nodes,
                               const CPSolution& solution,
                               ExtendedOpProfiler& op_profiler,
                               int dp_group_size, int n_part, int n_experts, const NodeMap<MoENodeLabel>& moe_label_map) {
  Nodes nodes_ = {nodes.begin(), nodes.end()};
  return PartitionNodes(sched_dfg, nodes_, solution, op_profiler, dp_group_size, n_part, n_experts, moe_label_map);
}

} // namespace partition_exprs
} // namespace pass
} // namespace raf