/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file schedule_test_utils.h
 * \brief Test utils for scheduling.
 */
#pragma once
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/analysis.h"
#include "raf/pass.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "../let_list.h"
#include <relay/transforms/pass_utils.h>
#include "partition_common.h"
#include "scheduler_common.h"
#include "partition_exprs.h"
#include "extended_dfg.h"
#include "solve_partition_axes.h"
#include "../stream_schedule.h"

namespace raf {
namespace pass {
namespace schedule_test_utils {

using namespace raf::analysis;
using namespace raf::analysis::dependency_graph;
using namespace scheduler_common;
using namespace partition_common;
using namespace solve_partition_axes;
using namespace partition_exprs;
using namespace extended_dfg;
using stream_schedule::StreamSchedulerBase;
using op::IsCollectiveOp;

class DebugExtendedOpScheduler : public StreamSchedulerBase {
 public:
  Expr VisitExpr_(const TupleNode* op) override {
    std::vector<Expr> fields;
    for (auto field : op->fields) {
      fields.push_back(VisitExpr(field));
    }
    auto new_tuple = Tuple(fields);
    new_tuple->checked_type_ = op->checked_type_;
    expr_map_[GetRef<Expr>(op)] = new_tuple;
    return let_list_.Push(new_tuple, op->checked_type_, true);
  }

  Expr VisitExpr_(const CallNode* c) override {
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    auto new_call = Call(VisitExpr(c->op), args, c->attrs, c->type_args);
    new_call->checked_type_ = c->checked_type_;
    expr_map_[GetRef<Expr>(c)] = new_call;
    return let_list_.Push(new_call, c->checked_type_, true);
  }

  Expr VisitExpr_(const TupleGetItemNode* tgi) override {
    auto new_tgi = TupleGetItem(VisitExpr(tgi->tuple), tgi->index);
    new_tgi->checked_type_ = tgi->checked_type_;
    expr_map_[GetRef<Expr>(tgi)] = new_tgi;
    return let_list_.Push(new_tgi, tgi->checked_type_, true);
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    // Should also transform a function in GNF or BBNF to ANF for partitioning.
    Arena arena;
    Expr body = func->body;
    auto dfg = CreateDependencyGraph(&arena, body, true);
    auto nodes = dfg.post_dfs_order;
    auto topo_order = dependency_graph::GetTopologicalOrder(nodes);
    NodeMap<Expr> node_expr = {};
    ExprMap<const Node*> dummy = {};
    extractDFGNodeExprRelationship(&dfg, node_expr, dummy);
    DebugExtendedOpScheduler scheduler;
    Expr ret;
    for (auto node : topo_order) {
      ret = scheduler.VisitExpr(node_expr.at(node));
    }
    Expr body_ = scheduler.Get(ret);
    Expr updated_func = Function(func->params, body_, func->ret_type, {}, func->attrs);
    updated_func->checked_type_ = func->checked_type_;
    return updated_func;
  }

  Expr Get(const Expr& ret) {
    return let_list_.Get(ret);
  }

  ExprMap<Expr> GetExprMap() {
    return expr_map_;
  }

 protected:
  ExprMap<Expr> expr_map_;
};

class DebugFIFOScheduler : public DebugExtendedOpScheduler {
 public:
  Expr Schedule(Expr e) {
    // create the data flow graph
    Arena arena;
    DependencyGraph dfg = CreateDependencyGraph(&arena, e, /*prune_atomic_nodes=*/true);
    // map each node in the dependency graph to the expression it represents
    NodeExprMap node_expr;
    // ready queue for all ops that directly depends on a communication op
    std::queue<Node*> comm_successor_ready_queue;
    // ready queue for all other ops
    std::queue<Node*> ready_queue;
    // counter that keeps track of the number of each op's current unscheduled predecessors
    // the dependency graph in tvm is a data flow graph with edge direction reversed, so we
    // use out-degree here instead of in-degree.
    std::unordered_map<Node*, int> out_degree;
    // keeps track of whether an op directly depends on a communication op
    std::unordered_set<Node*> comm_successor_nodes;

    for (auto& it : dfg.expr_node) {
      node_expr[it.second] = it.first;
    }
    std::vector<Node*>& nodes = dfg.post_dfs_order;

    // calculate out-degree for each node and populate comm_successor_nodes map
    for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); node_it++) {
      out_degree[(*node_it)] = 0;
      if (auto call_node = node_expr[*node_it].as<CallNode>()) {
        if (IsCollectiveOp(call_node->op)) {
          // record direct successor nodes of communication op
          for (auto parent = (*node_it)->parents.head; parent; parent = parent->next) {
            comm_successor_nodes.insert(parent->value);
          }
        }
      }

      for (auto child = (*node_it)->children.head; child; child = child->next) {
        out_degree[(*node_it)]++;
      }
    }
    // push nodes with zero predecessors into the queue
    for (auto& node : nodes) {
      if (out_degree[node] == 0) {
        ready_queue.push(node);
      }
    }

    Expr ret;
    // in each step, we pop an op out of the queue, add it to the ANF and
    // push all its ready successors into the corresponding ready queue
    auto process_queue_element = [&](std::queue<Node*>& q) {
      while (!q.empty()) {
        Node* node = q.front();
        ret = VisitExpr(node_expr.at(node));
        for (auto parent = node->parents.head; parent; parent = parent->next) {
          out_degree[parent->value]--;
          if (out_degree[parent->value] == 0) {
            if (comm_successor_nodes.count(parent->value)) {
              comm_successor_ready_queue.push(parent->value);
            } else {
              ready_queue.push(parent->value);
            }
          }
        }
        q.pop();
      }
    };

    while (!ready_queue.empty() || !comm_successor_ready_queue.empty()) {
      process_queue_element(ready_queue);
      process_queue_element(comm_successor_ready_queue);
    }

    return let_list_.Get(ret);
  }
};

inline std::tuple<ScheduledDFG, ExtendedOpProfiler> PrepareDFG(Expr expr) {
  LOG(INFO) << "Input expr: ";
  LOG(INFO) << ir::AsText(expr);
  Arena arena;
  DependencyGraph dfg = CreateDependencyGraph(&arena, expr, true);
  ExtendedDFG extended_dfg(&dfg);

  // obtain a ANF from expr
  auto scheduler = DebugFIFOScheduler();
  auto anf_expr = scheduler.Schedule(expr);
  auto expr_map = scheduler.GetExprMap();
  extended_dfg.updateExprFromMap(expr_map);
  for(auto it: expr_map) {
    std::ostringstream os;
    os << it.second;
    auto node = extended_dfg.getNodeFromExpr(it.second);
    if(!node) {
      LOG(WARNING) << "Expr " << it.second << " not found in dfg.";
    }
    extended_dfg.setNodeName(extended_dfg.getNodeFromExpr(it.second), os.str());
  }
  ScheduledDFG sched_dfg(extended_dfg);
  sched_dfg.setScheduledExpr(anf_expr);

  LOG(INFO) << "ANF expr: ";
  LOG(INFO) << ir::AsText(anf_expr);

  std::unique_ptr<ExplicitLetList> ell(ExplicitLetList::make(anf_expr));
  Exprs exprs;
  // LOG(INFO) << "Exprs: ";
  for(int i = 0; i < ell->exprs.size(); i++) {
    auto expr = ell->exprs[i];
    // LOG(INFO) << "\t -> " << expr;
    exprs.push_back(expr);
  }
  double overhead = 10;
  double throughput = 50000;
  if (const char* cm_fixed_params = getenv("DEBUG_CM_FIXED_PARAMS")) {
    std::string params_str(cm_fixed_params);
    auto overhead_str = params_str.substr(0, params_str.find(";"));
    auto throughput_str = params_str.substr(params_str.find(";")+1, std::string::npos);
    overhead = std::stod(overhead_str);
    throughput = std::stod(throughput_str);
  }
  LOG(INFO) << "Using overhead = " << overhead << ", throughput = " << throughput;
  // construct comm cost function
  CommCostModel f_coll_cost = [=](const CommComponents& comm_components) {
    SimulateTimeType result = 0;
    for(auto it: comm_components) {
        result += overhead + it.second / throughput;
    }
    return result;
  };

  ExtendedOpProfiler op_profiler(Device::Current(/*allow_default=*/false), f_coll_cost);

  // initialize comp exec time and comm size
  for(auto node: sched_dfg.dfg.nodes()) {
    auto expr = sched_dfg.dfg.getExprFromNode(node);
    if(auto call_node = expr.as<CallNode>()) {
      if(IsCollectiveOp(call_node->op)) {
        size_t input_size = 0;
        if(auto tuple_type = expr->checked_type().as<TupleTypeNode>()) {
          for(auto& ty: tuple_type->fields) {
            size_t field_size = 1;
            if(auto tensor_ty = ty.as<TensorTypeNode>()) {
              for(auto& x: tensor_ty->shape) {
                field_size *= Downcast<IntImm>(x)->value;
              }
              field_size *= tensor_ty->dtype.bits();
              input_size += field_size;
            }
          }
        } else {
          auto tensor_type = expr->checked_type().as<TensorTypeNode>();
          CHECK(tensor_type);
          input_size = 1;
          for(auto& x: tensor_type->shape) {
            input_size *= Downcast<IntImm>(x)->value;
          }
          input_size *= tensor_type->dtype.bits();
        }
        CHECK(call_node->op->IsInstance<OpNode>());
        CommComponents size = {{OpToCommType(Downcast<Op>(call_node->op)), input_size}};
        auto exec_time = op_profiler.GetCommOpExecTime(size);
        sched_dfg.dfg.setNodeExecTime(node, exec_time);
        sched_dfg.dfg.setCommSize(node, size);
        // LOG(INFO) << "Time for node " << expr << " is " << exec_time;
      } else {
        SimulateTimeType exec_time;
        std::tie(exec_time, std::ignore) = op_profiler.GetCompOpExecTime(expr);
        sched_dfg.dfg.setNodeExecTime(node, exec_time);
        // LOG(INFO) << "Time for node " << expr << " is " << exec_time;
      }
    } else {
      sched_dfg.dfg.setNodeExecTime(node, 0);
    }
  }

  regenerateNodeOrdersBasedOnExpr(sched_dfg);

  Nodes critical_path;
  std::tie(critical_path, std::ignore) = addResourceEdgesAndFindCritPath(sched_dfg);
  sched_dfg.setCriticalPath(critical_path);

  return std::make_tuple(sched_dfg, op_profiler);
}

} // namespace schedule_test_utils
} // namespace pass
} // namespace raf