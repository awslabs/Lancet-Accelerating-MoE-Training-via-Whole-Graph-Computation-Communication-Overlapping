/*!
 * Copyright (c) 2022 by Contributors
 * \file scheduler_utils.h
 * \brief Define data structures and utility functions for scheduling.
 */
#pragma once
#include <random>
#include "raf/analysis.h"
#include "raf/op_utils.h"
#include "../common.h"
#include "./scheduler_common.h"
#include "./extended_dfg.h"
#include "./schedule_generator.h"

namespace raf {
namespace pass {
namespace scheduler_utils {

using namespace tvm::relay;
using namespace raf::analysis;
using namespace raf::pass;
using namespace raf::pass::scheduler_common;
using namespace raf::pass::extended_dfg;
using namespace raf::pass::schedule_generator;
using op::IsCollectiveOp;
using NodeArgsMap = NodeMap<std::map<std::string, std::string>>;

// See LabelMoENodes for explanation
enum class MoENodeType {
    kNonMoECompute,
    kMoEDispatch,
    kMoEA2A,
    kMoEExperts,
    kMoEGather,
    kMoEOtherCompute,
    kUnknown,
};

std::ostream& operator<< (std::ostream& stream, const MoENodeType& type);

struct MoENodeLabel {
    MoENodeType type;
    int layer_id;
};

// Tuple of:
// 1. NodeMap<Nodes>: map each dW node to overlappable all-to-all nodes
// 2. NodeMap<NodeSet>: map each dW node to its predecessors
// 3. NodeSet: set of nodes to ignore when calculating dW schedule
using dWLabelingResult = std::tuple<NodeMap<Nodes>, NodeMap<NodeSet>, NodeSet>;

enum class TraceColor {
    kGreen,
    kBlue,
    kGray
};

using NodeColorMap = NodeMap<TraceColor>;

static const std::unordered_map<TraceColor, std::string> TraceColorMap {
    {TraceColor::kGreen, "thread_state_running"},
    {TraceColor::kBlue, "thread_state_runnable"},
    {TraceColor::kGray, "thread_state_sleeping"}
};

inline std::string getTraceColor(TraceColor color) {
    return TraceColorMap.at(color);
}

template <class T>
inline void postOrderAggregateNodes(const ExtendedDFG& G, const NodeMap<T>& weight, PostOrderFunctor f,
                                    NodeMap<double>& output, NodeMap<const Node*>* child_map=nullptr) {
    auto rev_topo_order = GetTopologicalOrder(G, /*reversed=*/true);
    for (int i=0; i< rev_topo_order.size(); i++) {
        auto node = rev_topo_order[i];
        CHECK(weight.count(node)) << "Unknown node found in rev_topo_order at index " << i;
        output[node] = static_cast<double>(weight.at(node));
        double child_aggregated = 0;
        for (auto parent: G.getParents(node)) {
            auto new_child_aggregated =
                f(output[parent], child_aggregated);
            if (child_map != nullptr &&
                new_child_aggregated != child_aggregated) {
                (*child_map)[node] = parent;
            }
            child_aggregated = new_child_aggregated;
        }
        output[node] += child_aggregated;
    }
}

inline bool IsOp(const Expr& expr, const Op& op) {
    if (auto call_node = expr.as<CallNode>()) {
        if (call_node->op->IsInstance<OpNode>()) {
            auto dialect_op_ = Downcast<Op>(call_node->op);
            auto op_ = op::IsDialectOp(dialect_op_) ? op::GetBaseOp(dialect_op_) : dialect_op_;
            if (op_ == op) {
                return true;
            }
        }
    }
    return false;
}

inline bool IsAllToAll(const Expr& expr) {
    static const Op& alltoall_op = Op::Get("raf.op._all_to_all");
    return IsOp(expr, alltoall_op);
}

inline bool IsAllToAllv(const Expr& expr) {
    static const Op& alltoallv_op = Op::Get("raf.op._all_to_allv");
    return IsOp(expr, alltoallv_op);
}

inline bool IsAllReduce(const Expr& expr) {
    static const Op& allreduce_op = Op::Get("raf.op._allreduce");
    return IsOp(expr, allreduce_op);
}

inline bool IsWeightUpdateFirstOp(const ExtendedDFG& G, const Node* node) {
    for (auto node_ : G.getChildren(node)) {
        auto expr_ = G.getExprFromNode(node_);
        if (IsAllReduce(expr_)) {
            return true;
        }
    }
    return false;
}

// a wrapper function for calculating FIFO priority
template <class T>
inline void calcFIFOPriority(const ExtendedDFG& G, const NodeMap<T>& weight,
                             NodeMap<double>& output, bool preserve_comm_prio=false,
                             bool deprioritize_allreduce=false,
                             bool deprioritize_weight_update=false) {
    for(auto& it: weight) {
        if (preserve_comm_prio && G.getNodeType(it.first) == NodeType::kCommNode) {
            continue;
        }
        if (deprioritize_allreduce) {
            auto expr = G.getExprFromNode(it.first);
            if (IsAllReduce(expr)) {
                output[it.first] = 0;
                continue;
            }
        }
        if (deprioritize_weight_update) {
            if (IsWeightUpdateFirstOp(G, it.first)) {
                output[it.first] = 0;
                continue;
            }
        }
        output[it.first] = 1;
    }
}

// single source shortest path length
NodeMap<double> shortestPathLengthInDAG(
    const ExtendedDFG& G, const NodeMap<double>& weight, const Node* source);

using PairwiseDistMap = NodeMap<NodeMap<double>>;

PairwiseDistMap allPairsShortestPathInDAG(const ExtendedDFG& G, const NodeMap<double>& weight);

std::pair<Nodes, SimulateTimeType> findCriticalPath(const ExtendedDFG& G);
std::pair<SimulateTimeType, NodeMap<SimulateTimeType>> findSubgraphExecTime(const ExtendedDFG& G, const NodeMap<SimulateTimeType>& exec_time_map, const Nodes& topo_order, const NodeMap<const Node*>& additional_parent_map);

std::pair<Nodes, SimulateTimeType> addResourceEdgesAndFindCritPath(const ScheduledDFG& G);

void regenerateTotalOrder(ScheduledDFG& G);

// This function makes sure that the total order stored in ScheduledDFG is consistent with its scheduled_expr
// which could be modified after fusion/partition
void regenerateTotalOrderBasedOnExpr(ScheduledDFG& G);
// This function regenerates all orders based on the scheduled ANF
void regenerateNodeOrdersBasedOnExpr(ScheduledDFG& G);
// this function must only be called when the DFG and expr var maps are already modified
void regenerateExprBasedOnTotalOrder(ScheduledDFG& G);

void DumpTraceToJSON(std::string dump_simulation_prefix, const ScheduledDFG& sched_dfg, const NodeColorMap& color_map={}, const NodeArgsMap& extra_args={});
void DumpExprToJSON(std::string dump_prefix, const Expr& expr);

template <typename T>
inline std::pair<std::vector<T>, std::vector<T>> WeightedSample(const std::vector<T>& elements, const std::vector<SimulateTimeType>& weights, int n_samples) {
    std::vector<SimulateTimeType> weights_ = weights;
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    int sampled_indices[elements.size()];
    if(n_samples >= elements.size()) {
        return {elements, {}};
    }
    for(int i=0; i<elements.size(); i++) {
        sampled_indices[i] = 0;
    }
    int n_sampled_elements = 0;
    while(n_sampled_elements < n_samples) {
        std::discrete_distribution<int> distribution(weights_.begin(), weights_.end());
        int idx = distribution(generator);
        if (!sampled_indices[idx]) {
            sampled_indices[idx] = 1;
            weights_[idx] = 0;
            n_sampled_elements ++;
        }
    }
    std::vector<T> selected;
    std::vector<T> not_selected;
    for(int i=0; i<elements.size(); i++) {
        if(sampled_indices[i] == 1) {
            selected.push_back(elements[i]);
        } else {
            not_selected.push_back(elements[i]);
        }
    }
    return {selected, not_selected};
}

inline Vars getInputs(const Expr& expr) {
    Vars inputs = {};
    if (expr->IsInstance<CallNode>()) {
        auto args = Downcast<Call>(expr)->args;
        for (auto arg : args) {
            if (arg->IsInstance<VarNode>()) {
                inputs.push_back(Downcast<Var>(arg));
            }
        }
    } else if (expr->IsInstance<TupleNode>()) {
        auto fields = Downcast<Tuple>(expr)->fields;
        for (auto field : fields) {
            if (field->IsInstance<VarNode>()) {
                inputs.push_back(Downcast<Var>(field));
            }
        }
    } else if (expr->IsInstance<TupleGetItemNode>()) {
        inputs.push_back(Downcast<Var>(Downcast<TupleGetItem>(expr)->tuple));
    }
    return inputs;
}

inline void updateVarAndExprs(Vars& vars, Exprs& exprs, ExprSet& binded_vars, ExprMap<Vars>& var_inputs,
                              ExprMap<Vars>& var_to_delay, ExprMap<Exprs>& expr_to_delay,
                              Var var_to_bind) {
    binded_vars.insert(var_to_bind);
    if (var_to_delay.count(var_to_bind)) {
        auto vars_ = var_to_delay.at(var_to_bind);
        auto exprs_ = expr_to_delay.at(var_to_bind);
        var_to_delay.erase(var_to_bind);
        expr_to_delay.erase(var_to_bind);
        for (int i = 0; i < vars_.size(); ++i) {
            auto var = vars_[i];
            auto expr = exprs_[i];
            auto inputs = var_inputs.at(var);
            bool all_input_binded = true;
            for (auto input : inputs) {
                if (!binded_vars.count(input)) {
                    all_input_binded = false;
                    break;
                }
            }
            if (all_input_binded) {
                vars.push_back(var);
                exprs.push_back(expr);
                updateVarAndExprs(vars, exprs, binded_vars, var_inputs, var_to_delay, expr_to_delay, var);
            }
        }
    }
}

inline Function fixOrder(Expr expr) {
    if (WellFormed(expr)) {
        // LOG(INFO) << "Original expr is well formed";
        return Downcast<Function>(expr);
    }
    // LOG(INFO) << "Original expr is not well formed";
    Function func = Downcast<Function>(expr);
    auto ell = ExplicitLetList::make(func->body);
    ExprMap<int> var_idx = {};
    Vars vars = {};
    Exprs exprs = {};
    ExprSet binded_vars = {func->params.begin(), func->params.end()};
    // LOG(INFO) << "Input param size " << binded_vars.size();
    // LOG(INFO) << "Initial vars size " << ell->vars.size() << ", exprs size " << ell->exprs.size();
    ExprMap<Vars> var_inputs = {};
    ExprMap<Vars> var_to_delay = {};
    ExprMap<Exprs> expr_to_delay = {};

    for (int i = 0; i < ell->vars.size(); ++i) {
        auto inputs = getInputs(ell->exprs[i]);
        var_inputs[ell->vars[i]] = inputs;
        Exprs free_vars = {};
        for (auto input : inputs) {
            if (!binded_vars.count(input)) {
                // LOG(INFO) << "Has free var " << ir::AsText(input);
                free_vars.push_back(input);
            }
        }
        if (!free_vars.empty()) {
            // LOG(INFO) << "delay var " << ir::AsText(ell->vars[i]);
            // LOG(INFO) << "delay expr " << ir::AsText(ell->exprs[i]);
            for (auto free_var : free_vars) {
                var_to_delay[free_var].push_back(ell->vars[i]);
                expr_to_delay[free_var].push_back(ell->exprs[i]);
            }
        } else {
            // LOG(INFO) << "add var " << ir::AsText(ell->vars[i]);
            // LOG(INFO) << "add expr " << ir::AsText(ell->exprs[i]);
            vars.push_back(ell->vars[i]);
            exprs.push_back(ell->exprs[i]);
            updateVarAndExprs(vars, exprs, binded_vars, var_inputs, var_to_delay, expr_to_delay, ell->vars[i]);
        }
    }
    ell->vars = vars;
    ell->exprs = exprs;
    Expr body = ell->AsExpr();
    auto result = WithFields(func, func->params, body);
    CHECK(WellFormed(result));
    return result;
}

dWLabelingResult LabeldWNodes(const ExtendedDFG& G, const Nodes& all_nodes_launch_order);
NodeMap<MoENodeLabel> LabelMoENodes(const ExtendedDFG& G, const Nodes& all_nodes_launch_order);
Nodes GreedyReorderDWNodes(const ExtendedDFG& G, const Nodes& all_nodes_launch_order, const dWLabelingResult& node_labeling_result);
std::tuple<Nodes, Nodes, Nodes, NodeMap<NodeDuration>> rescheduleBasedOnCompAndCommOrder(const ExtendedDFG& G, const Nodes& comp_launch_order, const Nodes& comm_launch_order);

}  // namespace scheduler_utils
}  // namespace pass
}  // namespace raf