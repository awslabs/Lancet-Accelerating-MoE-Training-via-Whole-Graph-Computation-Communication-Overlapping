/*!
 * Copyright (c) 2022 by Contributors
 * \file fuse_exprs.cc
 * \brief Fuse given exprs.
 */
#include "./fuse_exprs.h"

namespace raf {
namespace pass {
namespace fuse_exprs {

std::string GetExprName(Expr expr) {
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

Expr SubstituteArgs(Expr expr, const ExprMap<Expr>& expr_map) {
  if (expr->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr);
    Array<Expr> args = {};
    for (auto arg : call->args) {
      if (expr_map.count(arg)) {
        args.push_back(expr_map.at(arg));
      } else {
        args.push_back(arg);
      }
    }
    auto new_expr = Call(call->op, args);
    new_expr->checked_type_ = expr->checked_type();
    return new_expr;
  } else if (expr->IsInstance<TupleNode>()) {
    Tuple tuple = Downcast<Tuple>(expr);
    Array<Expr> fields = {};
    for (auto field : tuple->fields) {
      if (expr_map.count(field)) {
        fields.push_back(expr_map.at(field));
      } else {
        fields.push_back(field);
      }
    }
    auto new_expr = Tuple(fields);
    new_expr->checked_type_ = expr->checked_type();
    return new_expr;
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    TupleGetItem tgi = Downcast<TupleGetItem>(expr);
    Expr tuple = tgi->tuple;
    if (expr_map.count(tuple)) {
      tuple = expr_map.at(tuple);
    }
    auto new_expr = TupleGetItem(tuple, tgi->index);
    new_expr->checked_type_ = expr->checked_type();
    return new_expr;
  } else {
    LOG(FATAL) << "Not supported expr to get arg : " << ir::AsText(expr);
    return expr;
  }
}

Array<Expr> GetArgArray(Expr expr) {
  if (expr->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr);
    return {call->args.begin(), call->args.end()};
  } else if (expr->IsInstance<TupleNode>()) {
    Tuple tuple = Downcast<Tuple>(expr);
    return {tuple->fields.begin(), tuple->fields.end()};
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    return {Downcast<TupleGetItem>(expr)->tuple};
  } else {
    LOG(FATAL) << "Not supported expr to get arg : " << ir::AsText(expr);
    return {};
  }
}

Expr GetNArg(Expr expr, int n) {
  auto args = GetArgs<Exprs>(expr);
  CHECK(n < args.size());
  return args[n];
}

Type MakeTupleType(Expr expr_a, Expr expr_b) {
  Array<Type> fields = {};
  if(auto tuple_type = expr_a->checked_type().as<TupleTypeNode>()) {
    for(auto field_type: tuple_type->fields) {
      fields.push_back(field_type);
    }
  } else {
    fields.push_back(expr_a->checked_type());
  }
  if(auto tuple_type = expr_b->checked_type().as<TupleTypeNode>()) {
    for(auto field_type: tuple_type->fields) {
      fields.push_back(field_type);
    }
  } else {
    fields.push_back(expr_b->checked_type());
  }
  return TupleType(fields);
}

Type MakeTupleType(Exprs exprs_a, Exprs exprs_b) {
  Array<Type> fields = {};
  for (auto expr : exprs_a) {
    fields.push_back(expr->checked_type_);
  }
  for (auto expr : exprs_b) {
    fields.push_back(expr->checked_type_);
  }
  return TupleType(fields);
}

Array<Expr> CombineFields(Exprs exprs_a, Exprs exprs_b) {
  Array<Expr> fields = {};
  for (auto expr : exprs_a) {
    fields.push_back(expr);
  }
  for (auto expr : exprs_b) {
    fields.push_back(expr);
  }
  return fields;
}

int GetFieldsCount(Var var) {
  Type type = var->type_annotation;
  if (type->IsInstance<TensorTypeNode>()) {
    return 1;
  } else if (type->IsInstance<TupleTypeNode>()) {
    return Downcast<TupleType>(type)->fields.size();
  } else {
    LOG(FATAL) << "Unsupported type " << ir::AsText(type) << ", var " << ir::AsText(var);
    return -1;
  }
}

Type GetFieldType(Var var, int index) {
  Type type = var->type_annotation;
  if (type->IsInstance<TensorTypeNode>()) {
    CHECK(index == 0);
    return type;
  } else if (type->IsInstance<TupleTypeNode>()) {
    CHECK(Downcast<TupleType>(type)->fields.size() > index);
    return Downcast<TupleType>(type)->fields[index];
  } else {
    LOG(FATAL) << "Unsupported type " << ir::AsText(type) << ", var " << ir::AsText(var);
    return Type();
  }
}

Expr TryGetLetValue(Expr expr) {
  if (expr->IsInstance<LetNode>()) {
    return Downcast<Let>(expr)->value;
  }
  return expr;
}

Call GetCall(Expr expr) {
  Expr call_expr = TryGetLetValue(expr);
  CHECK(call_expr->IsInstance<CallNode>());
  return Downcast<Call>(call_expr);
}

Call GetOpCall(Expr expr, std::string dialect_name) {
  Call call = GetCall(expr);
  CHECK(call->op->IsInstance<OpNode>());
  Op op = Downcast<Op>(call->op);
  if (op->name.compare(dialect_name) == 0) {
    return call;
  }
  LOG(WARNING) << "Expr " << ir::AsText(expr) << ", dialect name " << dialect_name << " does not match.";
  return call;
}

void CheckOpCall(Expr expr, std::string dialect_name) {
  Call call = GetCall(expr);
  CHECK(call->op->IsInstance<OpNode>());
  Op op = Downcast<Op>(call->op);
  CHECK(op->name.compare(dialect_name) == 0);
}

std::vector<int64_t> GetInts(Expr expr) {
  // int64_t is required in ArrayToIntTuple to create TupleValue.
  auto fields = Downcast<TupleValue>(expr.as<ConstantNode>()->value)->fields;
  std::vector<int64_t> vec = {};
  for (auto field : fields) {
    vec.push_back(Downcast<IntValue>(field)->value);
  }
  return vec;
}

Array<Expr> FuseAllreduceArgs(Expr fused_tuple, Expr expr_a, Expr expr_b) {
  Call call_a = GetOpCall(expr_a, "raf.op.nccl._allreduce");
  Call call_b = GetOpCall(expr_b, "raf.op.nccl._allreduce");

  Expr computation = MakeConstant(StringValue::make("sum"));
  if (call_a->args.size() == 2) {
    computation = call_a->args[1];
  } else if (call_b->args.size() == 2) {
    computation = call_b->args[1];
  }
  
  Array<Expr> args = {fused_tuple, computation};
  return args;
}

Array<Expr> FuseReduceScatterArgs(Expr fused_tuple, Expr expr_a, Expr expr_b) {
  Call call_a = GetOpCall(expr_a, "raf.op.nccl._reduce_scatter");
  Call call_b = GetOpCall(expr_b, "raf.op.nccl._reduce_scatter");

  auto shapes_a = GetInts(call_a->args[1]);
  auto shapes_b = GetInts(call_b->args[1]);
  shapes_a.insert(shapes_a.end(), shapes_b.begin(), shapes_b.end());
  Expr shapes = MakeConstant(ArrayToIntTuple(shapes_a));

  auto shape_indices_a = GetInts(call_a->args[2]);
  auto shape_indices_b = GetInts(call_b->args[2]);
  for (int i = 0; i < shape_indices_b.size(); ++i) {
    shape_indices_b[i] += shape_indices_a.back();
  }
  shape_indices_a.insert(shape_indices_a.end(), shape_indices_b.begin(), shape_indices_b.end());
  Expr shape_indices = MakeConstant(ArrayToIntTuple(shape_indices_a));

  Expr computation = MakeConstant(StringValue::make("sum"));
  if (call_a->args.size() == 4) {
    computation = call_a->args[3];
  } else if (call_b->args.size() == 4) {
    computation = call_b->args[3];
  }
  Array<Expr> args = {fused_tuple, shapes, shape_indices, computation};
  return args;
}

Array<Expr> FuseAllgatherArgs(Expr fused_tuple, Expr expr_a, Expr expr_b) {
  Call call_a = GetOpCall(expr_a, "raf.op.nccl._allgather");
  Call call_b = GetOpCall(expr_b, "raf.op.nccl._allgather");

  Expr axis = MakeConstant(ScalarValue::make(0));
  if (call_a->args.size() == 2) {
    axis = call_a->args[1];
  } else if (call_b->args.size() == 2) {
    axis = call_b->args[1];
  }
  
  Array<Expr> args = {fused_tuple, axis};
  return args;
}

// fuse functions

using FusedArgsGetter = std::function<Array<Expr>(Expr, Expr, Expr)>;

FFuse CreateDefaultCommFuse(std::string dialect_name, FusedArgsGetter fused_args_getter) {
  size_t pos = dialect_name.find_last_of('.');
  CHECK(pos != std::string::npos);
  std::string comm_name = dialect_name.substr(pos + 1);
  FFuse default_fuse = [=](ExprFusor& fusor, Expr expr_a, Expr expr_b) {
    const auto comm_op = op::GetDialectOp(dialect_name);
    CheckOpCall(expr_a, dialect_name);
    CheckOpCall(expr_b, dialect_name);

    auto input_a = fusor.GetInputExpr(expr_a, 0);
    auto input_b = fusor.GetInputExpr(expr_b, 0);

    // input_a and input_b may be tensor or tuple.
    // We should combine all fields of tuples when fusing.
    Exprs args_a = {input_a};
    if (input_a->IsInstance<TupleNode>()) {
      args_a = GetArgs<Exprs>(input_a);
    }
    Exprs args_b = {input_b};
    if (input_b->IsInstance<TupleNode>()) {
      args_b = GetArgs<Exprs>(input_b);
    }

    auto input_var_a = fusor.GetVar(input_a);
    auto input_var_b = fusor.GetVar(input_b);
    std::string fused_tuple_name_hint = std::string("fused_") + input_var_a->name_hint() + "_" + input_var_b->name_hint();
    auto fused_tuple_type = MakeTupleType(args_a, args_b);
    auto fused_tuple_var = MakeTypeCheckedVar(fused_tuple_name_hint, fused_tuple_type);
    auto fused_tuple = Tuple(CombineFields(args_a, args_b));
    fused_tuple->checked_type_ = fused_tuple_type;
    auto expr_to_append = fusor.GetLaterExpr(input_a, input_b);
    fusor.AddCompExpr(fused_tuple_var, fused_tuple, expr_to_append);
    fusor.SetExprOfSource(fused_tuple);
    fusor.SetExprToPropagate(fused_tuple);

    std::string fused_comm_name_hint = comm_name + "_" + fused_tuple_name_hint;
    auto fused_output_type = MakeTupleType(expr_a, expr_b);
    auto fused_comm_var = MakeTypeCheckedVar(fused_comm_name_hint, fused_output_type);
    auto fused_comm = Call(comm_op, fused_args_getter(fused_tuple_var, expr_a, expr_b));
    fused_comm->checked_type_ = fused_output_type;
    fusor.AddCommExpr(fused_comm_var, fused_comm, expr_to_append, expr_a, expr_b);

    auto var_a = fusor.GetVar(expr_a);
    auto var_b = fusor.GetVar(expr_b);
    fusor.SetFusedComm(fused_comm_var, var_a, var_b);

    if (input_a->IsInstance<TupleNode>()) {
      fusor.RemoveExpr(input_a);
    }
    if (input_b->IsInstance<TupleNode>()) {
      fusor.RemoveExpr(input_b);
    }
    fusor.RemoveExpr(expr_a);
    fusor.RemoveExpr(expr_b);
  };
  return default_fuse;
}

FFuseMap op_fuse = {
  {"raf.op.nccl._allreduce", CreateDefaultCommFuse("raf.op.nccl._allreduce", FuseAllreduceArgs)},
  {"raf.op.nccl._reduce_scatter", CreateDefaultCommFuse("raf.op.nccl._reduce_scatter", FuseReduceScatterArgs)},
  {"raf.op.nccl._allgather", CreateDefaultCommFuse("raf.op.nccl._allgather", FuseAllgatherArgs)},
};

FFuse GetFFuse(std::string op_name) {
  if (op_fuse.count(op_name)) {
    return op_fuse.at(op_name);
  } else if (base_dialect_op.count(op_name)) {
    return GetFFuse(base_dialect_op.at(op_name));
  } else {
    LOG(FATAL) << "Cannot find fuse rule for op " << op_name;
    return nullptr;
  }
}

FFuse GetFFuse(Expr expr) {
  Call call = Downcast<Call>(expr);
  std::string op_name = Downcast<Op>(call->op)->name;
  return GetFFuse(op_name);
}

// expr fusor

ExprFusor::ExprFusor(ScheduledDFG& sched_dfg,
                     ExtendedOpProfiler& op_profiler) :
  sched_dfg_(sched_dfg),
  op_profiler_(op_profiler),
  let_list_(ExplicitLetList::make(sched_dfg.scheduled_expr)) {
  UpdateIdxMap_();
}

void ExprFusor::Fuse(Expr expr_a, Expr expr_b) {
  auto op_fuse = GetFFuse(expr_a);
  op_fuse(*this, expr_a, expr_b);
  Finalize_();
}

Expr ExprFusor::GetInputExpr(Expr expr, int n) const {
  Expr input_var = GetNArg(expr, n);
  if (var_idx_.count(input_var)) {
    return let_list_->exprs.at(var_idx_.at(input_var));
  } else {
    LOG(WARNING) << "Input var not have corresponding expr, return input var.";
    return input_var;
  }
}

Var ExprFusor::GetInputVar(Expr expr, int n) const {
  return Downcast<Var>(GetNArg(expr, n));
}

Expr ExprFusor::GetExpr(Expr var) const {
  CHECK(var_idx_.count(var));
  return let_list_->exprs.at(var_idx_.at(var));
}

Var ExprFusor::GetVar(Expr expr) const {
  CHECK(expr_idx_.count(expr));
  return let_list_->vars.at(expr_idx_.at(expr));
}

Expr ExprFusor::GetPrevExpr(Expr expr) const {
  if (!expr_idx_.count(expr)) {
    return Expr{nullptr};
  }
  if (expr_idx_.at(expr) == 0) {
    LOG(FATAL) << "No prev expr to get.";
  }
  return let_list_->exprs.at(expr_idx_.at(expr) - 1);
}

Expr ExprFusor::GetLaterExpr(Expr expr_1, Expr expr_2) const {
  if (!expr_idx_.count(expr_1)) {
    return expr_2;
  } else if (!expr_idx_.count(expr_2)) {
    return expr_1;
  } else {
    return expr_idx_.at(expr_1) < expr_idx_.at(expr_2) ? expr_2 : expr_1;
  }
}

Expr ExprFusor::GetFormerExpr(Expr expr_1, Expr expr_2) const {
  if (!expr_idx_.count(expr_1)) {
    return expr_2;
  } else if (!expr_idx_.count(expr_2)) {
    return expr_1;
  } else {
    return expr_idx_.at(expr_1) < expr_idx_.at(expr_2) ? expr_1 : expr_2;
  }
}

void ExprFusor::AddCompExpr(Var var, Expr expr, Expr expr_to_append) {
  added_expr_var_[expr] = var;
  expr_appended_exprs_[expr_to_append].push_back(expr);
  // all added comp exprs' exec time is 0.
  sched_dfg_.insertNewCompExprAndAddInputEdges(var, expr, GetArgArray(expr), var->name_hint(), 0);
  sched_dfg_.setDefaultNodeDuration(sched_dfg_.dfg.getNodeFromExpr(expr));
}

void ExprFusor::AddCommExpr(Var var, Expr expr, Expr expr_to_append, Expr component_a, Expr component_b) {
  added_expr_var_[expr] = var;
  expr_appended_exprs_[expr_to_append].push_back(expr);

  auto node_a = sched_dfg_.dfg.getNodeFromExpr(component_a);
  auto comm_size_a = sched_dfg_.dfg.getCommSize(node_a);
  auto node_b = sched_dfg_.dfg.getNodeFromExpr(component_b);
  auto comm_size_b = sched_dfg_.dfg.getCommSize(node_b);
  auto comm_size = MergeComponents(comm_size_a, comm_size_b);

  // all added comm exprs' exec time is computed by profiler.
  sched_dfg_.insertNewCommExprAndAddInputEdges(var, expr, GetArgArray(expr), var->name_hint(),
                                               op_profiler_.GetCommOpExecTime(comm_size),
                                               comm_size);
  sched_dfg_.setDefaultNodeDuration(sched_dfg_.dfg.getNodeFromExpr(expr));
}

void ExprFusor::UpdateCompExpr(Expr expr_to_update, Expr new_expr) {
  updated_expr_[expr_to_update] = new_expr;
  auto var = GetVar(expr_to_update);
  auto node = sched_dfg_.dfg.getNodeFromExpr(expr_to_update);
  auto parents = sched_dfg_.dfg.getNonSinkParents(node);
  sched_dfg_.deleteNode(node);
  sched_dfg_.insertNewCompExprAndAddInputEdges(var, new_expr, GetArgArray(new_expr),
                                               GetExprName(new_expr), 0);
  auto new_node = sched_dfg_.dfg.getNodeFromExpr(new_expr);
  sched_dfg_.setDefaultNodeDuration(new_node);
  for (auto parent : parents) {
    sched_dfg_.dfg.createEdge(parent, new_node);
  }
}

void ExprFusor::RemoveExpr(Expr expr) {
  removed_exprs_.insert(expr);
  if (const Node* node = sched_dfg_.dfg.getNodeFromExpr(expr)) {
    sched_dfg_.deleteNode(node);
  }
}

void ExprFusor::RemoveExprs(Exprs exprs) {
  for (auto expr : exprs) {
    removed_exprs_.insert(expr);
    if (const Node* node = sched_dfg_.dfg.getNodeFromExpr(expr)) {
      sched_dfg_.deleteNode(node);
    }
  }
}

void ExprFusor::IdentifyCommConsumers_(Var comm_var, Exprs& tgi_consumers, ExprSet& other_consumers) {
  int idx = var_idx_.at(comm_var);
  for (auto expr : let_list_->exprs) {
    if (idx == expr_idx_.at(expr)) {
      continue;
    }
    auto args = GetArgs<ExprSet>(expr);
    if (args.count(comm_var)) {
      if (expr->IsInstance<TupleGetItemNode>()) {
        tgi_consumers.push_back(expr);
      } else {
        other_consumers.insert(expr);
      }
    }
  }
}

Expr ExprFusor::GetFirstExpr_(const Exprs& exprs) {
  if (exprs.empty()) {
    return Expr{nullptr};
  }
  std::vector<int> indices = {};
  for (const auto& expr : exprs) {
    indices.push_back(expr_idx_.at(expr));
  }
  return let_list_->exprs[*std::min_element(indices.begin(), indices.end())];
}

void ExprFusor::SetFusedComm(Var fused_comm_var, Var comm_var_a, Var comm_var_b) {
  Exprs tgi_consumers_a = {};
  Exprs tgi_consumers_b = {};
  ExprSet other_consumers_a = {};
  ExprSet other_consumers_b = {};
  ExprSet both_consumers = {};
  IdentifyCommConsumers_(comm_var_a, tgi_consumers_a, other_consumers_a);
  IdentifyCommConsumers_(comm_var_b, tgi_consumers_b, other_consumers_b);

  for (auto consumer : tgi_consumers_a) {
    auto tgi = Downcast<TupleGetItem>(consumer);
    int index = tgi->index;
    int new_index = index;
    auto new_tgi = TupleGetItem(fused_comm_var, new_index);
    new_tgi->checked_type_ = tgi->checked_type_;
    UpdateCompExpr(consumer, new_tgi);
  }

  int offset = GetFieldsCount(comm_var_a);
  for (auto consumer : tgi_consumers_b) {
    auto tgi = Downcast<TupleGetItem>(consumer);
    int index = tgi->index;
    int new_index = index + offset;
    auto new_tgi = TupleGetItem(fused_comm_var, new_index);
    new_tgi->checked_type_ = tgi->checked_type_;
    UpdateCompExpr(consumer, new_tgi);
  }

  Var recon_var_a;
  Var recon_var_b;
  if (!other_consumers_a.empty() || !other_consumers_b.empty()) {
    Exprs other_consumers_a_ = {other_consumers_a.begin(), other_consumers_a.end()};
    Exprs other_consumers_b_ = {other_consumers_b.begin(), other_consumers_b.end()};
    Expr first_consumer_a = GetFirstExpr_(other_consumers_a_);
    Expr first_consumer_b = GetFirstExpr_(other_consumers_b_);
    Expr expr_to_append_a = GetPrevExpr(first_consumer_a);
    Expr expr_to_append_b = GetPrevExpr(first_consumer_b);
    for (auto consumer : other_consumers_a) {
      if (other_consumers_b.count(consumer)) {
        other_consumers_a.erase(consumer);
        other_consumers_b.erase(consumer);
        both_consumers.insert(consumer);
      }
    }
    if (!both_consumers.empty()) {
      expr_to_append_a = expr_to_append_b = GetFormerExpr(expr_to_append_a, expr_to_append_b);
    }

    if (!other_consumers_a.empty() || !both_consumers.empty()) {
      // reconstruct a
      int n_fields = GetFieldsCount(comm_var_a);
      auto type = comm_var_a->type_annotation;
      if (n_fields == 1 && type->IsInstance<TensorTypeNode>()) {
        // only reconstruct expr by tgi
        std::string name_hint_a = std::string("recon_") + comm_var_a->name_hint();
        recon_var_a = MakeTypeCheckedVar(name_hint_a, type);
        auto tgi = TupleGetItem(fused_comm_var, 0);
        tgi->checked_type_ = type;
        AddCompExpr(recon_var_a, tgi, expr_to_append_a);
      } else {
        // insert tgis and reconstruct tuple.
        Array<Expr> fields = {};
        for (int index = 0; index < n_fields; ++index) {
          std::string tgi_name_hint_a = std::string("recon_") + std::to_string(index) + "_" + comm_var_a->name_hint();
          auto field_type = GetFieldType(comm_var_a, index);
          Var tgi_var = MakeTypeCheckedVar(tgi_name_hint_a, field_type);
          auto tgi = TupleGetItem(fused_comm_var, index);
          tgi->checked_type_ = field_type;
          fields.push_back(tgi_var);
          AddCompExpr(tgi_var, tgi, expr_to_append_a);
        }
        std::string name_hint_a = std::string("recon_") + comm_var_a->name_hint();
        recon_var_a = MakeTypeCheckedVar(name_hint_a, type);
        auto tuple = Tuple(fields);
        tuple->checked_type_ = type;
        AddCompExpr(recon_var_a, tuple, expr_to_append_a);
      }
    }

    if (!other_consumers_b.empty() || !both_consumers.empty()) {
      // reconstruct b
      int n_fields = GetFieldsCount(comm_var_b);
      auto type = comm_var_b->type_annotation;
      if (n_fields == 1 && type->IsInstance<TensorTypeNode>()) {
        // only reconstruct expr by tgi
        std::string name_hint_b = std::string("recon_") + comm_var_b->name_hint();
        recon_var_b = MakeTypeCheckedVar(name_hint_b, type);
        auto tgi = TupleGetItem(fused_comm_var, offset);
        tgi->checked_type_ = type;
        AddCompExpr(recon_var_b, tgi, expr_to_append_b);
      } else {
        // insert tgis and reconstruct tuple.
        Array<Expr> fields = {};
        for (int index = offset; index < offset + n_fields; ++index) {
          std::string tgi_name_hint_b = std::string("recon_") + std::to_string(index) + "_" + comm_var_b->name_hint();
          auto field_type = GetFieldType(comm_var_b, index - offset);
          Var tgi_var = MakeTypeCheckedVar(tgi_name_hint_b, field_type);
          auto tgi = TupleGetItem(fused_comm_var, index);
          tgi->checked_type_ = field_type;
          fields.push_back(tgi_var);
          AddCompExpr(tgi_var, tgi, expr_to_append_b);
        }
        std::string name_hint_b = std::string("recon_") + comm_var_b->name_hint();
        recon_var_b = MakeTypeCheckedVar(name_hint_b, type);
        auto tuple = Tuple(fields);
        tuple->checked_type_ = type;
        AddCompExpr(recon_var_b, tuple, expr_to_append_b);
      }
    }
  }

  ExprMap<Expr> arg_map_a = {};
  arg_map_a[comm_var_a] = recon_var_a;
  for (auto consumer : other_consumers_a) {
    auto new_consumer = SubstituteArgs(consumer, arg_map_a);
    UpdateCompExpr(consumer, new_consumer);
  }

  ExprMap<Expr> arg_map_b = {};
  arg_map_b[comm_var_b] = recon_var_b;
  for (auto consumer : other_consumers_b) {
    auto new_consumer = SubstituteArgs(consumer, arg_map_b);
    UpdateCompExpr(consumer, new_consumer);
  }

  ExprMap<Expr> arg_map_both = {};
  arg_map_both[comm_var_a] = recon_var_a;
  arg_map_both[comm_var_b] = recon_var_b;
  for (auto consumer : both_consumers) {
    auto new_consumer = SubstituteArgs(consumer, arg_map_both);
    UpdateCompExpr(consumer, new_consumer);
  }
}

void ExprFusor::SetExprToPropagate(Expr expr) {
  auto node = sched_dfg_.dfg.getNodeFromExpr(expr);
  need_propagate_[node] = true;
}

void ExprFusor::SetExprOfSource(Expr expr) {
  source_ = sched_dfg_.dfg.getNodeFromExpr(expr);
}

NodeMap<bool> ExprFusor::GetNeedPropagate() const {
  return need_propagate_;
}

const Node* ExprFusor::GetSource() const {
  return source_;
}

void ExprFusor::Finalize_() {
  // add exprs
  for (auto it : expr_appended_exprs_) {
    auto expr_to_append = it.first;
    auto exprs = it.second;
    int index = expr_idx_.at(expr_to_append);
    Vars vars = {};
    for (auto expr : exprs) {
      CHECK(added_expr_var_.count(expr));
      vars.push_back(added_expr_var_.at(expr));
    }
    auto var_it = let_list_->vars.begin() + index;
    auto expr_it = let_list_->exprs.begin() + index;
    let_list_->vars.insert(var_it, vars.begin(), vars.end());
    let_list_->exprs.insert(expr_it, exprs.begin(), exprs.end());
    UpdateIdxMap_();
  }
  // update exprs
  for (auto& expr : let_list_->exprs) {
    if (updated_expr_.count(expr)) {
      expr = updated_expr_[expr];
    }
  }
  UpdateIdxMap_();
  // remove exprs
  for (auto it = let_list_->exprs.begin(); it != let_list_->exprs.end(); ) {
    if (removed_exprs_.count(*it)) {
      int index = it - let_list_->exprs.begin();
      let_list_->vars.erase(let_list_->vars.begin() + index);
      it = let_list_->exprs.erase(let_list_->exprs.begin() + index);
    } else {
      ++it;
    }
  }
  // set scheduled expr
  Expr scheduled_expr = let_list_->AsExpr();
  sched_dfg_.setScheduledExpr(scheduled_expr);
  // set comm and comp chain for propagate
  Nodes comm_chain = {};
  Nodes comp_chain = {};
  for (auto expr : let_list_->exprs) {
    auto node = sched_dfg_.dfg.getNodeFromExpr(expr);
    if (sched_dfg_.dfg.getNodeType(node) == NodeType::kCompNode) {
      comp_chain.push_back(node);
    } else {
      comm_chain.push_back(node);
    }
  }
  sched_dfg_.comm_chain = comm_chain;
  sched_dfg_.comp_chain = comp_chain;
}

void ExprFusor::UpdateIdxMap_() {
  for (int i = 0; i < let_list_->vars.size(); ++i) {
    expr_idx_[let_list_->exprs[i]] = i;
    var_idx_[let_list_->vars[i]] = i;
  }
}

std::tuple<const Node*, NodeMap<bool>>
FuseNodes(ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler,
          const Node* node_a, const Node* node_b) {
  ExprFusor fusor(sched_dfg, op_profiler);
  Expr expr_a = sched_dfg.dfg.getExprFromNode(node_a);
  Expr expr_b = sched_dfg.dfg.getExprFromNode(node_b);
  fusor.Fuse(expr_a, expr_b);
  return std::make_tuple(fusor.GetSource(), fusor.GetNeedPropagate());
}

} // namespace fuse_exprs
} // namespace pass
} // namespace raf