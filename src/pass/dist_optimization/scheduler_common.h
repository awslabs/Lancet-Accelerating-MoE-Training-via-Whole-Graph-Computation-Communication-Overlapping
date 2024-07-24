/*!
 * Copyright (c) 2022 by Contributors
 * \file scheduler_common.h
 * \brief Define data structures used in scheduling and optimization.
 */
#pragma once
#include <iostream>
#include <unordered_map>
#include <unistd.h>
#include <queue>
#include <relay/transforms/pass_utils.h>
#include "raf/ir.h"
#include "raf/op_utils.h"
#include "raf/analysis.h"
#include "../../analysis/dependency_graph.h"

#define FUSION_PREFIX "fused_"
#define FUSION_DELIM "+"
#define LOG_ITER(ITER) LOG(INFO) << "[Iter " << ITER << "]\t"

namespace raf {
namespace pass {
namespace scheduler_common {

using namespace raf::analysis;
using DependencyGraph = tvm::relay::DependencyGraph;
using Node = DependencyGraph::Node;
using SimulateTimeType = double;

template <typename T>
using NodeMap = std::unordered_map<const Node*, T>;
using NodeSet = std::unordered_set<const Node*>;
using Nodes = std::vector<const Node*>;
template <class CompareF>
using NodePriorityQueue = std::priority_queue<Node*, std::deque<Node*>, CompareF>;
using PostOrderFunctor =
    std::function<double(const double, const double)>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using Exprs = std::vector<Expr>;
template <typename T>
using VarMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using Vars = std::vector<Var>;

struct VarAndExpr {
  Var var;
  Expr expr;
};

using VarAndExprs = std::vector<VarAndExpr>;

enum class NodeType {
    kCompNode, kCommNode
};

enum class CommType {
    kAllReduce,
    kReduceScatter,
    kAllGather,
    kAllToAll
};

// for sum up the tensor size of different comm ops.
// e.g. assume we have an AllReduce op t1, then its CommComponents
// should be {{allreduce, size}}
// when fusing different kind of ops, just add up the corresponding
// size of each comm type. If they are of the same type,
// cost model would get a reduced time, otherwise the fused time
// is not shorter than the original communication time.
// Then the fusion would not be performed.
using CommComponents = std::unordered_map<CommType, uint64_t>;
CommComponents MergeComponents(CommComponents& lhs, CommComponents& rhs);
std::ostream& operator<< (std::ostream& s, CommType comm_type);
using CommCostModel = std::function<SimulateTimeType(const CommComponents&)>;

CommType IdentifyCommType(const CommComponents& components);

CommType OpToCommType(const Op& op);

enum class ScheduleHeuristics {
    kFIFO,
    kDW,
};

enum class TimelineOptAlgo {
  kHeuristic,
  kDP,
  kRangeBased,
};

std::ostream& operator<<(std::ostream& os, const NodeType& nodetype);

class NodeDuration {
   public:
    NodeDuration(Node* blocked_by, SimulateTimeType start,
                 SimulateTimeType end, SimulateTimeType ready_time);

    NodeDuration();

    const Node* blocked_by;
    SimulateTimeType start;
    SimulateTimeType end;
    SimulateTimeType ready_time;
    std::vector<int> critical_sets;
};

std::ostream& operator << (std::ostream& os, const NodeDuration& duration);

std::string removeFusionPrefix(const std::string& name);

std::string fuseNodeNames(std::string name_a, std::string name_b);

static std::unordered_map<std::string, std::string> base_dialect_op = {
  {"raf.op.cast", "raf.op.tvm.cast"},
  {"raf.op.add", "raf.op.tvm.add"},
  {"raf.op.subtract", "raf.op.tvm.subtract"},
  {"raf.op.multiply", "raf.op.tvm.multiply"},
  {"raf.op.divide", "raf.op.tvm.divide"},
  {"raf.op.where", "raf.op.tvm.where"},
  {"raf.op.scatter", "raf.op.tvm.scatter"},
  {"raf.op.cumsum", "raf.op.tvm.cumsum"},
  // {"raf.op.cumprod", "raf.op.tvm.cumprod"}, // Actually we don't support cumprod in RAF now.
  {"raf.op.layer_norm", "raf.op.tvm.layer_norm"},
  {"raf.op.layer_norm_dx", "raf.op.tvm.layer_norm_dx"},
  {"raf.op.relu_dx", "raf.op.tvm.relu_dx"},
  {"raf.op.softmax_dx", "raf.op.tvm.softmax_dx"},
  {"raf.op.reshape", "raf.op.tvm.reshape"},
  {"raf.op.bias_add", "raf.op.tvm.bias_add"},
  {"raf.op.power", "raf.op.tvm.power"},
  {"raf.op.transpose", "raf.op.tvm.transpose"},
  {"raf.op.transpose_dx", "raf.op.tvm.transpose_dx"},
  {"raf.op.topk", "raf.op.tvm.topk"},
  {"raf.op.sum", "raf.op.tvm.sum"},
  {"raf.op.one_hot", "raf.op.tvm.one_hot"},
  {"raf.op.argmax", "raf.op.tvm.argmax"},
  {"raf.op.expand_dims", "raf.op.tvm.expand_dims"},
  {"raf.op.embedding", "raf.op.tvm.embedding"},
  {"raf.op.tanh", "raf.op.cudnn.tanh"},
  {"raf.op.relu", "raf.op.cudnn.relu"},
  {"raf.op.softmax", "raf.op.cudnn.softmax"},
  {"raf.op.batch_matmul", "raf.op.cublas.batch_matmul"},
  {"raf.op.batch_matmul_tn", "raf.op.cublas.batch_matmul_tn"},
  {"raf.op.batch_matmul_nt", "raf.op.cublas.batch_matmul_nt"},
  {"raf.op.batch_matmul_tt", "raf.op.cublas.batch_matmul_tt"},
  {"raf.op.matmul", "raf.op.cublas.batch_matmul"},
  {"raf.op.matmul_tn", "raf.op.cublas.matmul_tn"},
  {"raf.op.matmul_nt", "raf.op.cublas.matmul_nt"},
  {"raf.op.matmul_tt", "raf.op.cublas.matmul_tt"},
  {"raf.op._allreduce", "raf.op.nccl._allreduce"},
  {"raf.op._all_to_all", "raf.op.nccl._all_to_all"},
  {"raf.op.split", "raf.op.tvm.split"},
};

inline Var MakeTypeCheckedVar(std::string name_hint, Type type_annotation) {
  auto var = MakeVar(name_hint, type_annotation);
  var->checked_type_ = type_annotation;
  return var;
}

template <class T>
inline T GetArgs(Expr expr) {
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

}  // namespace scheduler_common
}  // namespace pass
}  // namespace raf