/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lancet_optimization.cc
 * \brief Implements joint optimization.
 */
#include <chrono>
#include "lancet_optimization.h"

namespace raf {
namespace pass {
namespace lancet_optimization {

bool dbl_approx_eq(double a, double b) {
    // return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * SIMULATOR_EPS);
    return a == b;
}

SimulateTimeType PropagateUpdateThroughNodeDuration(ScheduledDFG& sched_dfg, const Node* source, NodeMap<bool> need_update = {},
                                                    NodeMap<const Node*> stream_pred_map = {}, NodeMap<const Node*> stream_succ_map = {}) {
    auto& dfg = sched_dfg.dfg;
    auto& node_durations = sched_dfg.node_durations;
    auto& comp_chain = sched_dfg.comp_chain;
    auto& comm_chain = sched_dfg.comm_chain;

    // construct a index of comp_chain and comm_chain so each node can get predecessor and successors easily
    if (stream_pred_map.empty() || stream_succ_map.empty()) {
        auto fill_pred_succ_maps = [&](Nodes& node_chain) {
            for(int idx = 0; idx < node_chain.size(); idx++) {
                const Node* curr_node = node_chain[idx];
                if(idx == 0) {
                    stream_pred_map[curr_node] = nullptr;
                } else {
                    DLOG(INFO) << "Node " << dfg.getNodeNameOrDefault(node_chain[idx - 1]) << " precedes " << dfg.getNodeNameOrDefault(curr_node);
                    stream_pred_map[curr_node] = node_chain[idx - 1];
                }
                if(idx == node_chain.size() - 1) {
                    stream_succ_map[curr_node] = nullptr;
                } else {
                    stream_succ_map[curr_node] = node_chain[idx + 1];
                }
            }
        };
        fill_pred_succ_maps(comp_chain);
        fill_pred_succ_maps(comm_chain);
    }

    // 1. set flag need_update for source node
    need_update[source] = true;

    // 2. update the nodes' time in according to the new topological order

    // Function to propagate update from one node to its parents.
    // It adjusts the start and end time of n based on its children, and propagate
    // need_update = true to its parents if its start or end time is modified.
    auto update_func = [&](const Node* n) {
        auto& n_duration = node_durations.at(n);
        SimulateTimeType n_exec_time = n_duration.end - n_duration.start;
        DLOG(INFO) << "Calling update on node " << dfg.getNodeNameOrDefault(n) << " (" << n_duration.start << ", " << n_duration.end << ").";
        SimulateTimeType max_child_end_time = std::numeric_limits<double>::lowest();
        const Node* latest_child = nullptr;
        // update n's start and end time
        // first check children with data dependency
        for(auto child: dfg.getChildren(n)) {
            DLOG(INFO) << "Checking node children " << dfg.getNodeNameOrDefault(child);
            if (node_durations.at(child).end > max_child_end_time) {
                DLOG(INFO) << "Setting last children to " << dfg.getNodeNameOrDefault(child);
                max_child_end_time = node_durations.at(child).end;
                latest_child = child;
            } else {
                DLOG(INFO) << "Not set last children since the child has end time " << node_durations.at(child).end << " leq " << max_child_end_time;
            }
        }

        // then check stream predecessor
        CHECK(stream_pred_map.count(n)) << "Node " << dfg.getNodeNameOrDefault(n) << " which is flagged with need_update is not in comp or comm chain.";
        if(stream_pred_map.at(n) != nullptr) {
            auto stream_pred_node = stream_pred_map.at(n);
            if (node_durations.at(stream_pred_node).end > max_child_end_time) {
                max_child_end_time = node_durations.at(stream_pred_node).end;
                latest_child = stream_pred_node;
            }
        }
        CHECK(latest_child || n == dfg.source()) << "Can't find latest child for " << dfg.getNodeNameOrDefault(n);
        if(n == dfg.source()) {
            max_child_end_time = 0;
        }
        if(!dbl_approx_eq(max_child_end_time, n_duration.start)) {
            n_duration.start = max_child_end_time;
            n_duration.end = max_child_end_time + n_exec_time;
            DLOG(INFO) << "Updated node: (" << n_duration.start << ", " << n_duration.end << ").";
            n_duration.blocked_by = latest_child;
            // need update the node's parents
            for(auto parent: dfg.getNonSinkParents(n)) {
                DLOG(INFO) << "Flagging node: " << dfg.getNodeNameOrDefault(parent) << " as need update.";
                need_update[parent] = true;
            }
            // also stream successor
            if(stream_succ_map.at(n) != nullptr) {
                DLOG(INFO) << "Flagging node: " << dfg.getNodeNameOrDefault(stream_succ_map.at(n)) << " as need update.";
                need_update[stream_succ_map.at(n)] = true;
            }
        } else {
            for(auto parent: dfg.getNonSinkParents(n)) {
                if (node_durations.at(parent).start < n_duration.end) {
                    DLOG(INFO) << "Flagging node: " << dfg.getNodeNameOrDefault(parent) << " as need update.";
                    need_update[parent] = true;
                }
            }
        }
        DLOG(INFO) << "Update finished on node " << dfg.getNodeNameOrDefault(n) << " (" << n_duration.start << ", " << n_duration.end << ").";
    };
    auto topo_order = addResourceEdgesAndGetTopologicalOrder(sched_dfg);
    // run update according to topo order
    for(int i = 0; i < topo_order.size(); i++) {
        if(need_update[topo_order[i]]) {
            update_func(topo_order[i]);
        }
    }
    auto last_comp = comp_chain.back();
    auto last_comm = comm_chain.back();
    float last_comm_end = 0;
    float last_comp_end = 0;
    // for(int i=1; i<comm_chain.size(); i++) {
    //     auto n = comm_chain[i];
    //     if(node_durations[n].start == node_durations[n].end) {
    //         continue;
    //     }
    //     // check intervals are not overlapping
    //     CHECK_GE(node_durations[n].start + 1, last_comm_end) << "Overlapping communication: "
    //         << dfg.getNodeNameOrDefault(comm_chain[i-1]) << " [comm chain idx " << i-1
    //         << " (st: " << node_durations[comm_chain[i-1]].start << ", ed: "
    //         << node_durations[comm_chain[i-1]].end << ")] is overlapping with: "
    //         << dfg.getNodeNameOrDefault(n) << " [comm chain idx "
    //         << i << " (st: " << node_durations[n].start << ", ed: " << node_durations[n].end << ")].";
    //     last_comm_end = node_durations[n].end;
    // }
    // for(int i=1; i<comp_chain.size(); i++) {
    //     auto n = comp_chain[i];
    //     CHECK_GE(node_durations[n].start + 1, last_comp_end) << "Overlapping computation: "
    //         << dfg.getNodeNameOrDefault(comp_chain[i-1]) << " [comp chain idx " << i-1
    //         << " (st: " << node_durations[comp_chain[i-1]].start << ", ed: "
    //         << node_durations[comp_chain[i-1]].end << ")] is overlapping with "
    //         << dfg.getNodeNameOrDefault(n) << " [comp chain idx "
    //         << i << " (st: " << node_durations[n].start << ", ed: " << node_durations[n].end << ")].";
    //     last_comp_end = node_durations[n].end;
    // }
    return std::max(node_durations.at(last_comp).end, node_durations.at(last_comm).end);
}

int GetFusedOpIdx(Expr e, Expr expr_a, Expr expr_b) {
    std::unique_ptr<ExplicitLetList> let_list(ExplicitLetList::make(e));

    int range_end = std::numeric_limits<int>::max();
    int range_start = -1;

    ExprIdxMap expr_idx;
    VarIdxMap var_idx;

    std::vector<int> comm_indices;
    std::vector<int> comp_indices;
    ExprIdxMap comm_order;
    ExprIdxMap comp_order;
    for(int i=0; i<let_list->vars.size(); i++) {
        expr_idx[let_list->exprs[i]] = i;
        var_idx[let_list->vars[i]] = i;
        auto expr = let_list->exprs[i];
        if (auto call_node = expr.as<CallNode>()) {
            if(IsCollectiveOp(call_node->op)) {
                comm_indices.push_back(i);
                comm_order[expr] = comm_indices.size() - 1;
            } else {
                comp_indices.push_back(i);
                comp_order[expr] = comp_indices.size() - 1;
            }
        }
    }
    // LOG(INFO) << "Before fusion: Expr comm order:";
    // for(auto comm_idx: comm_indices) {
    //     LOG(INFO) << "\t -> " << let_list->exprs[comm_idx];
    // }
    CHECK(expr_idx.count(expr_a) && expr_idx.count(expr_b)) << "Cannot find expr_a or expr_b in Expr.";

    auto a_idx = expr_idx.at(expr_a);
    auto b_idx = expr_idx.at(expr_b);
    auto var_a = let_list->vars[a_idx];
    auto var_b = let_list->vars[b_idx];

    auto a_call_node = expr_a.as<CallNode>();
    auto b_call_node = expr_b.as<CallNode>();
    CHECK(a_call_node) << "expr_a is not call node.";
    CHECK(b_call_node) << "expr_b is not call node.";

    for(auto arg_b: b_call_node->args) {
        if(arg_b->IsInstance<VarNode>()) {
            auto arg_var = Downcast<Var>(arg_b);
            CHECK(var_idx.count(arg_var));
            auto arg_idx = var_idx.at(arg_var);
            range_start = std::max(range_start, arg_idx);
        }
    }

    for(int i=0; i<let_list->vars.size(); i++) {
        auto expr_i = let_list->exprs[i];
        if(auto call_node = expr_i.as<CallNode>()) {
            for(auto arg_c: call_node->args) {
                if (arg_c == var_a) {
                    range_end = std::min(range_end, i);
                }
            }
        } else if (auto tuple_node = expr_i.as<TupleNode>()) {
            for(auto field: tuple_node->fields) {
                if (field == var_a) {
                    range_end = std::min(range_end, i);
                }
            }
        } else if (auto tgi_node = expr_i.as<TupleGetItemNode>()) {
            if (tgi_node->tuple == var_a) {
                range_end = std::min(range_end, i);
            }
        }
    }

    bool is_a_comm = IsCollectiveOp(a_call_node->op);
    bool is_b_comm = IsCollectiveOp(b_call_node->op);
    CHECK_EQ(is_a_comm, is_b_comm) << "expr_a and expr_b are in different streams.";

    auto& stream_idx_map = is_a_comm ? comm_order : comp_order;
    auto& stream_indices = is_a_comm ? comm_indices : comp_indices;
    int a_stream_idx = stream_idx_map.at(expr_a);
    int b_stream_idx = stream_idx_map.at(expr_b);

    // LOG(INFO) << "In finding fused idx: before considering same stream op: (" <<  range_start << ", " << range_end << ")";
    if(b_stream_idx >= 1) {
        range_start = std::max(range_start, stream_indices[b_stream_idx-1]);
    }
    // LOG(INFO) << "In finding fused idx: after considering same stream op: (" <<  range_start << ", " << range_end << ")";

    CHECK_LT(range_start, range_end) << "Unable to find valid index to place fused op.";
    return range_start + 1;
}

Expr ReorderOp(Expr e, Expr target_op, int target_idx) {
    std::unique_ptr<ExplicitLetList> let_list(ExplicitLetList::make(e));
    ExprIdxMap expr_idx;
    VarIdxMap var_idx;
    // populate expr -> idx and var -> idx map
    for(int i=0; i<let_list->vars.size(); i++) {
        expr_idx[let_list->exprs[i]] = i;
        var_idx[let_list->vars[i]] = i;
    }
    CHECK(expr_idx.count(target_op)) << "Target op is not found in let list.";
    int current_index = expr_idx.at(target_op);

    if (target_idx == current_index) {
        return e;
    }

    auto target_var = let_list->vars[current_index];

    if(target_idx < current_index) {
        for(int i = current_index - 1; i >= target_idx; i--) {
            let_list->vars[i+1] = let_list->vars[i];
            let_list->exprs[i+1] = let_list->exprs[i];
        }
    } else {
        for(int i = current_index; i < target_idx; i++) {
            let_list->vars[i] = let_list->vars[i+1];
            let_list->exprs[i] = let_list->exprs[i+1];
        }
    }
    let_list->vars[target_idx] = target_var;
    let_list->exprs[target_idx] = target_op;

    // DEBUG
    // LOG(INFO) << "In reorder op, moving " << target_op << " to index " << target_idx;

    // LOG(INFO) << "After reordering: Comm Expr order:";
    // for(int i=0; i<let_list->exprs.size(); i++) {
    //     auto expr = let_list->exprs[i];
    //     if(auto call_node = expr.as<CallNode>()) {
    //         if(IsCollectiveOp(call_node->op)) {
    //             LOG(INFO) << "\t -> " << expr;
    //         }
    //     }
    // }

    return let_list->AsExpr();
}

// This function returns a copy of the ScheduledDFG where the corresponding nodes are fused
std::tuple<SimulateTimeType, ScheduledDFG> GetFusedExecTime(
    const ScheduledDFG& sched_dfg,
    const Node* comm_a, const Node* comm_b,
    ExtendedOpProfiler& op_profiler) {

    ScheduledDFG new_sched_dfg = sched_dfg;

    Expr expr_a = new_sched_dfg.dfg.getExprFromNode(comm_a);
    Expr expr_b = new_sched_dfg.dfg.getExprFromNode(comm_b);
    int fused_expr_idx = GetFusedOpIdx(new_sched_dfg.scheduled_expr, expr_a, expr_b);
    auto reordered_expr = ReorderOp(new_sched_dfg.scheduled_expr, expr_b, fused_expr_idx);
    new_sched_dfg.setScheduledExpr(reordered_expr);

    const Node* source = nullptr;
    NodeMap<bool> need_update = {};
    std::tie(source, need_update) = FuseNodes(new_sched_dfg, op_profiler, comm_a, comm_b);

    auto new_time = PropagateUpdateThroughNodeDuration(new_sched_dfg, /*source=*/source, /*need_update=*/need_update);
    return std::make_tuple(new_time, new_sched_dfg);
}

std::tuple<SimulateTimeType, ScheduledDFG> GetPartitionedExecTime(
    const ScheduledDFG& sched_dfg, int dp_group_size, int n_experts, int number_of_partitions, ExtendedOpProfiler& op_profiler, RegionOfInterest roi) {

    auto partition_result = PartitionNodes(sched_dfg, roi.critical_set, roi.cp_solution, op_profiler, dp_group_size, number_of_partitions, n_experts);

    auto& new_sched_dfg = partition_result.sched_dfg;
    ExprMap<ScheduleLocation>& expr_schedule_locations = partition_result.expr_schedule_locations;

    // convert the exprmap to node map first
    NodeMap<ScheduleLocation> node_schedule_locs;
    for(auto it: expr_schedule_locations) {
        auto node = new_sched_dfg.dfg.getNodeFromExpr(it.first);
        CHECK(node) << "Cannot get corresponding node for partitioned expr " << it.first;
        node_schedule_locs[node] = it.second;
    }
    // get the list of irrelevant nodes in roi region in original order
    Nodes irrelevant_prelude_nodes;
    Nodes irrelevant_epilogue_nodes;
    // we need to use the old dfg here since the original nodes are removed in the new dfg
    std::tie(irrelevant_prelude_nodes, irrelevant_epilogue_nodes) = GetScheduleLocationForIrrelevantNodes(sched_dfg, roi);
    // insert the additional prelude and epilogue nodes into expr_schedule_locations
    // LOG(INFO) << "*********************************";
    // LOG(INFO) << "Irrelevant prelude nodes:";
    for(auto node: irrelevant_prelude_nodes) {
        node_schedule_locs[node] = ScheduleAtPrelude();
        // LOG(INFO) << "\t-> " << new_sched_dfg.dfg.getNodeNameOrDefault(node);
    }
    // LOG(INFO) << "*********************************";
    // LOG(INFO) << "Irrelevant epilogue nodes:";
    for(auto node: irrelevant_epilogue_nodes) {
        node_schedule_locs[node] = ScheduleAtEpilogue();
        // LOG(INFO) << "\t-> " << new_sched_dfg.dfg.getNodeNameOrDefault(node);
    }
    // LOG(INFO) << "*********************************";

    NodeSet all_nodes_to_schedule;
    for(auto it: node_schedule_locs) {
        all_nodes_to_schedule.insert(it.first);
    }

    // LOG(INFO) << "*********************************";
    // LOG(INFO) << "Nodes to schedule:";
    // for(auto it: node_schedule_locs) {
    //     LOG(INFO) << "\t-> " << new_sched_dfg.dfg.getNodeNameOrDefault(it.first) << ", loc: " << it.second;
    // }
    // LOG(INFO) << "*********************************";
    NodeMap<double> pipeline_priority;
    // here we only regenerate schedule for nodes that consist of the pipeline
    calcPipelinePriority(new_sched_dfg, node_schedule_locs, pipeline_priority);
    CompareNode partitioned_static_comp_comparator(pipeline_priority);
    CompareNode partitioned_static_comm_comparator(pipeline_priority);
    NodeMap<NodeDuration> roi_durations;
    StaticScheduleGenerator
        roi_local_generator(new_sched_dfg.dfg, partitioned_static_comp_comparator, partitioned_static_comm_comparator, /*node_mask=*/all_nodes_to_schedule);
    roi_local_generator.Run(roi_durations);
    // we actually don't need roi_durations, we just need the new node launch orders
    Nodes roi_comp_launch_order;
    Nodes roi_comm_launch_order;
    std::tie(std::ignore, roi_comp_launch_order, roi_comm_launch_order) = roi_local_generator.GetNodeOrder();
    // LOG(INFO) << " ";
    // LOG(INFO) << "roi_comp_launch_order: ";
    // for(auto node: roi_comp_launch_order) {
    //     LOG(INFO) << "\t -> " << new_sched_dfg.dfg.getNodeNameOrDefault(node); 
    // }
    // LOG(INFO) << " ";
    // LOG(INFO) << "roi_comm_launch_order: ";
    // for(auto node: roi_comm_launch_order) {
    //     LOG(INFO) << "\t -> " << new_sched_dfg.dfg.getNodeNameOrDefault(node); 
    // }
    // LOG(INFO) << " ";

    // create new comp and comm launch order
    auto create_partitioned_launch_orders = [&](NodeType chain_type, const Nodes& node_chain) {
        Nodes result;
        if(node_chain.empty()) {
            if(chain_type == NodeType::kCompNode) {
                DLOG(INFO) << "Pushing " << roi_comp_launch_order.size() << " nodes into comp chain.";
                for(auto node: roi_comp_launch_order) {
                    result.push_back(node);
                }
            } else {
                DLOG(INFO) << "Pushing " << roi_comm_launch_order.size() << " nodes into comm chain.";
                for(auto node: roi_comm_launch_order) {
                    result.push_back(node);
                }
            }
        } else {
            int current_region_start, current_region_end;
            // we need to use old sched_dfg here since the new source node is partitioned
            if(sched_dfg.dfg.getNodeType(roi.source_node) == chain_type) {
                current_region_start = roi.source_chain.start_index;
                current_region_end = roi.source_chain.end_index;
            } else {
                current_region_start = roi.target_chain.start_index;
                current_region_end = roi.target_chain.end_index;
            }
            // LOG(INFO) << "current_region_start: " << current_region_start << ", current_region_end: " << current_region_end << ", node chain size: " << node_chain.size();
            // LOG(INFO) << "Nodes in between: ";
            // for(int i=current_region_start; i <=current_region_end; i++) {
            //     LOG(INFO) << "\t-> " << sched_dfg.dfg.getExprFromNode(node_chain[i]);
            // }

            for(int chain_idx = 0; chain_idx < node_chain.size(); chain_idx++) {
                if(chain_idx < current_region_start || chain_idx > current_region_end) {
                    result.push_back(node_chain[chain_idx]);
                } else if (chain_idx == current_region_start) {
                    if(chain_type == NodeType::kCompNode) {
                        DLOG(INFO) << "Pushing " << roi_comp_launch_order.size() << " nodes into comp chain.";
                        for(auto node: roi_comp_launch_order) {
                            result.push_back(node);
                        }
                    } else {
                        DLOG(INFO) << "Pushing " << roi_comm_launch_order.size() << " nodes into comm chain.";
                        for(auto node: roi_comm_launch_order) {
                            result.push_back(node);
                        }
                    }
                    chain_idx = current_region_end;
                } else {
                    CHECK(false) << "This should never happen!";
                }
            }
        }
        CHECK_GE(result.size(), node_chain.size());
        return result;
    };
    // we also need to use the old dfg's comp chain and comm chain
    auto partitioned_comp_launch_order = create_partitioned_launch_orders(NodeType::kCompNode, sched_dfg.comp_chain);
    // LOG(INFO) << "partitioned_comp_launch_order: ";
    // for(auto node: partitioned_comp_launch_order) {
    //     LOG(INFO) << "\t -> " << new_sched_dfg.dfg.getNodeNameOrDefault(node); 
    // }
    // LOG(INFO) << "";
    auto partitioned_comm_launch_order = create_partitioned_launch_orders(NodeType::kCommNode, sched_dfg.comm_chain);
    // LOG(INFO) << "partitioned_comm_launch_order: ";
    // for(auto node: partitioned_comm_launch_order) {
    //     LOG(INFO) << "\t -> " << new_sched_dfg.dfg.getNodeNameOrDefault(node); 
    // }
    // LOG(INFO) << "";

    new_sched_dfg.comp_chain = partitioned_comp_launch_order;
    new_sched_dfg.comm_chain = partitioned_comm_launch_order;
    // LOG(INFO) << "Partitioned comm chain: ";
    // for(auto node: new_sched_dfg.comm_chain) {
    //     LOG(INFO) << "\t -> " << new_sched_dfg.dfg.getNodeNameOrDefault(node); 
    // }
    new_sched_dfg.total_order_valid = false;
    new_sched_dfg.critical_path_valid = false;

    // create an empty node duration map where all nodes's durations are reset to start at zero
    NodeMap<NodeDuration> new_node_durations;
    for(auto node: new_sched_dfg.dfg.nodes()) {
        auto new_duration = NodeDuration();
        // we should use the exec time stored in the graph as the source of exec time since
        // we have partitioned some nodes
        auto exec_time = new_sched_dfg.dfg.getNodeExecTime(node);
        // we assign a negative start time for every non-root nodes to ensure that their time will be updated
        if(node == new_sched_dfg.dfg.source()) {
            new_duration.start = 0;
        } else {
            new_duration.start = -1.0;
        }
        new_duration.end = new_duration.start + exec_time;
        new_node_durations[node] = new_duration;
    }

    new_sched_dfg.node_durations = new_node_durations;

    // LOG(INFO) << "Partitioned DFG:";
    // LOG(INFO) << "\n" << new_sched_dfg.dfg;

    auto new_time = PropagateUpdateThroughNodeDuration(new_sched_dfg, /*source=*/new_sched_dfg.dfg.source());
    LOG(INFO) << "Partitioned time is: " << new_time / 1000.0;
    // if(const char* dump_simulation_prefix = getenv("SIMULATION_DEBUG_PREFIX")) {
    //   DumpTraceToJSON(std::string(dump_simulation_prefix) + "_"+std::to_string(new_time), new_sched_dfg);
    // }
    return std::make_tuple(new_time, new_sched_dfg);
}

using NodeRange = std::pair<int, int>;
using PartitionRange = std::tuple<NodeRange, NodeRange, NodeRange>;

std::tuple<SimulateTimeType, ScheduledDFG, PartitionRange> DPApplyPartition(
    const ScheduledDFG& sched_dfg, const NodeMap<MoENodeLabel>& moe_label_map,
    int dp_group_size, int n_experts, int number_of_partitions,
    ExtendedOpProfiler& op_profiler, const PartitionRange& partition_range,
    const NodeSet& connected_components, const NodeSet& other_nodes,
    const CPSolution& cp_solution) {

    // 1. partition the dfg
    auto partition_result = PartitionNodes(sched_dfg, connected_components, cp_solution, op_profiler, dp_group_size, number_of_partitions, n_experts, moe_label_map);

    // 2. reconstruct the op orders
    auto& new_sched_dfg = partition_result.sched_dfg;
    for(auto node: new_sched_dfg.dfg.nodes()) {
        if (new_sched_dfg.dfg.source() == node || new_sched_dfg.dfg.sink() == node) {
            continue;
        }
        auto expr = new_sched_dfg.dfg.getExprFromNode(node);
        if (!expr.defined()) {
            LOG(WARNING) << "After partitioning, node " << new_sched_dfg.dfg.getNodeNameOrDefault(node) << " has undefined expr.";
        }
    }
    ExprMap<ScheduleLocation>& expr_schedule_locations = partition_result.expr_schedule_locations;

    // convert the exprmap to node map first
    NodeMap<ScheduleLocation> node_schedule_locs;
    for(auto it: expr_schedule_locations) {
        auto node = new_sched_dfg.dfg.getNodeFromExpr(it.first);
        CHECK(node) << "Cannot get corresponding node for partitioned expr " << it.first;
        node_schedule_locs[node] = it.second;
    }
    // for all non-partitioned node, we assign a low priority
    for(auto node: other_nodes) {
        node_schedule_locs[node] = ArbitrarySchedule();
    }
    NodeSet all_nodes_to_schedule;
    for(auto it: node_schedule_locs) {
        all_nodes_to_schedule.insert(it.first);
    }

    NodeMap<double> pipeline_priority;
    // here we only regenerate schedule for nodes that consist of the pipeline
    // calcPipelinePriorityBasedOnPartitionIndex(new_sched_dfg, node_schedule_locs, number_of_partitions, pipeline_priority);
    calcPipelinePriority(new_sched_dfg, node_schedule_locs, pipeline_priority);
    CompareNode partitioned_static_comp_comparator(pipeline_priority);
    CompareNode partitioned_static_comm_comparator(pipeline_priority);

    // before scheduling, we zero out durations for other communications since
    // we do not count them during DP
    NodeMap<SimulateTimeType> comm_durations;
    for(auto node: new_sched_dfg.dfg.nodes()) {
        if(new_sched_dfg.dfg.getNodeType(node) == NodeType::kCommNode) {
            auto expr = new_sched_dfg.dfg.getExprFromNode(node);
            if(!IsAllToAll(expr) && !IsAllToAllv(expr)) {
                comm_durations[node] = new_sched_dfg.dfg.getNodeExecTime(node);
                new_sched_dfg.dfg.setNodeExecTime(node, 0);
            }
        }
    }
    NodeMap<NodeDuration> pipeline_durations;
    PhasedScheduleGenerator
        pipeline_local_generator(new_sched_dfg.dfg, partitioned_static_comp_comparator, partitioned_static_comm_comparator, node_schedule_locs, /*node_mask=*/all_nodes_to_schedule);
    pipeline_local_generator.Run(pipeline_durations);

    Nodes pipeline_comp_launch_order;
    Nodes pipeline_comm_launch_order;
    Nodes pipeline_total_order;
    std::tie(pipeline_total_order, pipeline_comp_launch_order, pipeline_comm_launch_order) = pipeline_local_generator.GetNodeOrder();
    // get pipeline time by looking at the end time of the last node
    SimulateTimeType pipeline_time = 0;
    if(!pipeline_comp_launch_order.empty()) {
        pipeline_time = std::max(pipeline_durations[pipeline_comp_launch_order.back()].end, pipeline_time);
    }
    if(!pipeline_comm_launch_order.empty()) {
        pipeline_time = std::max(pipeline_durations[pipeline_comm_launch_order.back()].end, pipeline_time);
    }
    // restore the comm durations
    for(auto it: comm_durations) {
        new_sched_dfg.dfg.setNodeExecTime(it.first, it.second);
    }

    auto replace_nodes_in_chain = [&](const Nodes& chain_to_be_inserted, const NodeRange& range_to_replace, const Nodes& chain_to_insert) {
        Nodes result;
        if(chain_to_be_inserted.empty()) {
            result = chain_to_insert;
        } else if (chain_to_insert.empty()) {
            result = chain_to_be_inserted;
        } else {
            int current_region_start = range_to_replace.first;
            int current_region_end = range_to_replace.second;
            for(int chain_idx = 0; chain_idx < chain_to_be_inserted.size(); chain_idx++) {
                if(chain_idx < current_region_start || chain_idx >= current_region_end) {
                    result.push_back(chain_to_be_inserted[chain_idx]);
                } else if (chain_idx == current_region_start) {
                    result.insert(result.end(), chain_to_insert.begin(), chain_to_insert.end());
                    chain_idx = current_region_end - 1;
                } else {
                    CHECK(false) << "Range Error!";
                }
            }
        }
        CHECK_GE(result.size(), chain_to_insert.size());
        return result;
    };
    // replace partitioned node in the old dfg's comp chain and comm chain
    auto partitioned_comp_launch_order = replace_nodes_in_chain(sched_dfg.comp_chain, std::get<1>(partition_range), pipeline_comp_launch_order);
    NodeRange new_comp_range = std::make_pair(std::get<1>(partition_range).first, std::get<1>(partition_range).first + pipeline_comp_launch_order.size());
    auto partitioned_comm_launch_order = replace_nodes_in_chain(sched_dfg.comm_chain, std::get<2>(partition_range), pipeline_comm_launch_order);
    NodeRange new_comm_range = std::make_pair(std::get<2>(partition_range).first, std::get<2>(partition_range).first + pipeline_comm_launch_order.size());
    auto partitioned_total_order = replace_nodes_in_chain(sched_dfg.total_order, std::get<0>(partition_range), pipeline_total_order);
    NodeRange new_total_range = std::make_pair(std::get<0>(partition_range).first, std::get<0>(partition_range).first + pipeline_total_order.size());


    new_sched_dfg.comp_chain = partitioned_comp_launch_order;
    new_sched_dfg.comm_chain = partitioned_comm_launch_order;
    new_sched_dfg.total_order = partitioned_total_order;

    // new_sched_dfg.total_order_valid = false;
    new_sched_dfg.critical_path_valid = false;
    return std::make_tuple(pipeline_time, new_sched_dfg, std::make_tuple(new_total_range, new_comp_range, new_comm_range));
}

SimulateTimeType GetRangeOrigCost(const ScheduledDFG& sched_dfg, const PartitionRange& partition_range) {
    Nodes nodes_to_partition(sched_dfg.total_order.begin() + std::get<0>(partition_range).first, sched_dfg.total_order.begin() + std::get<0>(partition_range).second);
    Nodes comp_nodes_to_partition(sched_dfg.comp_chain.begin() + std::get<1>(partition_range).first, sched_dfg.comp_chain.begin() + std::get<1>(partition_range).second);
    Nodes comm_nodes_to_partition(sched_dfg.comm_chain.begin() + std::get<2>(partition_range).first, sched_dfg.comm_chain.begin() + std::get<2>(partition_range).second);
    // zero out durations for non-all2all comm nodes since we do not count them during DP
    NodeMap<SimulateTimeType> exec_time_map;
    // NodeMap<int> node_to_idx;
    // int idx = 0;
    for (auto node: nodes_to_partition) {
        // node_to_idx[node] = idx;
        // idx++;
        if(sched_dfg.dfg.getNodeType(node) == NodeType::kCommNode) {
            auto expr = sched_dfg.dfg.getExprFromNode(node);
            if(!IsAllToAll(expr)) {
                exec_time_map[node] = 0;
                continue;
            }
        }
        exec_time_map[node] = sched_dfg.dfg.getNodeExecTime(node);
    }
    // // sanity check
    // for (auto node: comp_nodes_to_partition) {
    //     CHECK(exec_time_map.count(node)) << "Node " << sched_dfg.dfg.getNodeNameOrDefault(node) << " not found in exec_time_map. Partition range mismatch!";
    // }
    // for (auto node: comm_nodes_to_partition) {
    //     CHECK(exec_time_map.count(node)) << "Node " << sched_dfg.dfg.getNodeNameOrDefault(node) << " not found in exec_time_map. Partition range mismatch!";
    // }
    // add control dependencies
    NodeMap<const Node*> node_to_control_parent;
    for(int i=0; i<static_cast<int>(comp_nodes_to_partition.size())-1; i++) {
        node_to_control_parent[comp_nodes_to_partition[i]] = comp_nodes_to_partition[i+1];
    }
    for(int i=0; i<static_cast<int>(comm_nodes_to_partition.size())-1; i++) {
        node_to_control_parent[comm_nodes_to_partition[i]] = comm_nodes_to_partition[i+1];
    }
    // LOG(INFO) << "Getting range cost for nodes:";
    // for(auto node: nodes_to_partition) {
    //     LOG(INFO) << " -> " << sched_dfg.dfg.getNodeNameOrDefault(node) << ", (" << exec_time_map[node] / 1000.0 << " ms)";
    // }
    // LOG(INFO) << "Compute order:";
    // for(auto node: comp_nodes_to_partition) {
    //     LOG(INFO) << " -> " << sched_dfg.dfg.getNodeNameOrDefault(node);
    // }
    // LOG(INFO) << "Comm order:";
    // for(auto node: comm_nodes_to_partition) {
    //     LOG(INFO) << " -> " << sched_dfg.dfg.getNodeNameOrDefault(node);
    // }
    SimulateTimeType range_cost;
    NodeMap<SimulateTimeType> time_till_end_map;
    std::tie(range_cost, time_till_end_map) = findSubgraphExecTime(sched_dfg.dfg, exec_time_map, nodes_to_partition, node_to_control_parent);
    // auto range_cost = findSubgraphExecTime(sched_dfg.dfg, exec_time_map, nodes_to_partition, node_to_control_parent);
    // std::vector<std::pair<const Node*, SimulateTimeType>> end_time_vec(time_till_end_map.begin(), time_till_end_map.end());
    // std::sort(end_time_vec.begin(), end_time_vec.end(), [](const std::pair<const Node*, SimulateTimeType>& a, const std::pair<const Node*, SimulateTimeType>& b) {
    //     return a.second > b.second;
    // });
    // LOG(INFO) << "Time till end:";
    // for (auto& p: end_time_vec) {
    //     LOG(INFO) << " " << sched_dfg.dfg.getNodeNameOrDefault(p.first) << ", end time: " << p.second / 1000.0 << " ms (order index : " << node_to_idx.at(p.first) << ")";
    // }
    // LOG(INFO) << "Range cost: " << range_cost / 1000.0 << " ms";
    return range_cost;
    // return findSubgraphExecTime(sched_dfg.dfg, exec_time_map, nodes_to_partition, node_to_control_parent);;
}

std::tuple<SimulateTimeType, ScheduledDFG, PartitionRange, bool, bool> GetDPLocalCost(
    const ScheduledDFG& sched_dfg, const PartitionRange& partition_range, ExtendedOpProfiler& op_profiler, const NodeMap<MoENodeLabel>& moe_label_map,
    int dp_group_size, int n_experts, int64_t& cc_time, int64_t& spa_time, int64_t& apply_time,
    bool skip_non_alltoall = true, int max_n_partition = 0, bool ignore_orig_cost = false, bool skip=false) {
    // This function calculates the t(i,j) function in the DP algorithm
    // 0. find original cost before partition
    auto orig_cost = GetRangeOrigCost(sched_dfg, partition_range);
    if (skip) {
        return std::make_tuple(orig_cost, sched_dfg, partition_range, false, false);
    }
    // 1. test whether there are all2all in the region. If not, don't partition
    NodeSet all_nodes_in_region;
    NodeSet all2all_nodes;
    Nodes nodes_to_partition(sched_dfg.total_order.begin() + std::get<0>(partition_range).first, sched_dfg.total_order.begin() + std::get<0>(partition_range).second);
    for(auto node: nodes_to_partition) {
        all_nodes_in_region.insert(node);
        if(IsAllToAll(sched_dfg.dfg.getExprFromNode(node)) && moe_label_map.count(node) && moe_label_map.at(node).type == MoENodeType::kMoEA2A) {
            all2all_nodes.insert(node);
        }
        if (sched_dfg.dfg.source() == node || sched_dfg.dfg.sink() == node) {
            // contains source or sink, don't partition
            return std::make_tuple(orig_cost, sched_dfg, partition_range, false, false);
        }
    }
    if (all2all_nodes.empty() && skip_non_alltoall) {
        // no all2all in the region, don't partition
        return std::make_tuple(orig_cost, sched_dfg, partition_range, false, false);
    }
    // 2. find connected components in with respect to all2all nodes
    auto cc_start = std::chrono::system_clock::now();
    NodeSet connected_components;
    if (skip_non_alltoall) {
        std::queue<const Node*> node_queue;
        for(auto node: all2all_nodes) {
            node_queue.push(node);
        }
        while(!node_queue.empty()) {
            auto node = node_queue.front();
            node_queue.pop();
            connected_components.insert(node);
            for(auto node: sched_dfg.dfg.getParents(node)) {
                if(!connected_components.count(node) && all_nodes_in_region.count(node)) {
                    node_queue.push(node);
                }
            }
            for(auto node: sched_dfg.dfg.getChildren(node)) {
                if(!connected_components.count(node) && all_nodes_in_region.count(node)) {
                    node_queue.push(node);
                }
            }
        }
    } else {
        for (auto node: all_nodes_in_region) {
            connected_components.insert(node);
        }
    }
    NodeSet other_nodes;
    for(auto node: all_nodes_in_region) {
        if(!connected_components.count(node)) {
            other_nodes.insert(node);
        }
    }
    auto cc_end = std::chrono::system_clock::now();
    auto cc_elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(cc_end - cc_start).count();
    cc_time = cc_elapsed_time;
    // 3. partition the connected components
    auto spa_start = std::chrono::system_clock::now();
    for(auto node: connected_components) {
        if (!sched_dfg.dfg.getExprFromNode(node).defined()) {
            LOG(FATAL) << "Found undefined node in connected components: " << sched_dfg.dfg.getNodeNameOrDefault(node);
        }
    }
    CPSolution partition_solution = SolvePartitionAxes(sched_dfg, connected_components, dp_group_size);
    auto spa_end = std::chrono::system_clock::now();
    auto spa_elapsed_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(spa_end - spa_start).count();
    spa_time = spa_elapsed_time;
    if (!IsAllPartitionable(partition_solution)) {
        // the region is not partitionable, don't partition
        return std::make_tuple(orig_cost, sched_dfg, partition_range, false, false);
    }
    auto apply_start = std::chrono::system_clock::now();
    int max_partition = max_n_partition == 0 ? 8 : max_n_partition;
    if (const char* max_partition_str = getenv("MAX_PARTITION")) {
        max_partition = std::stod(std::string(max_partition_str));
    }
    // get valid partition numbers
    auto n_partitions = GetPartitionParts(sched_dfg, connected_components, partition_solution, max_partition, dp_group_size);
    // DEBUG: test if any components is kExpert partitioned
    bool has_expert_partitioned = false;
    for (auto it: partition_solution.expr_inout_axes) {
        auto inout_axes = it.second;
        for (auto indices: inout_axes.first) {
            for (auto index: indices) {
                if (index == SpecialAxis::kExpert) {
                    has_expert_partitioned = true;
                    break;
                }
            }
            if (has_expert_partitioned) {
                break;
            }
        }
        if (has_expert_partitioned) {
            break;
        }
    }
    SimulateTimeType min_new_cost = std::numeric_limits<SimulateTimeType>::infinity();
    ScheduledDFG best_sched_dfg = sched_dfg;
    PartitionRange best_partition_range = partition_range;
    LOG(INFO) << "Testing partition numbers: Original cost: " << orig_cost;
    if (has_expert_partitioned) {
        LOG(INFO) << "NOTE: Found expert partitioned components.";
    }
    for(auto n_partition : n_partitions) {
        try {
            auto cost_and_new_sched_dfg = DPApplyPartition(sched_dfg, moe_label_map, dp_group_size, n_experts, n_partition, op_profiler, partition_range, connected_components, other_nodes, partition_solution);
            SimulateTimeType new_cost = std::get<0>(cost_and_new_sched_dfg);
            LOG(INFO) << "-> Partition into " << n_partition << " parts, cost: " << new_cost;
            if (new_cost < min_new_cost) {
                min_new_cost = new_cost;
                best_sched_dfg = std::get<1>(cost_and_new_sched_dfg);
                best_partition_range = std::get<2>(cost_and_new_sched_dfg);
            } else {
                // no need to test higher number of partitions
                break;
            }
        } catch (const InvalidPartitionException& e) {
            // cannot partition
            LOG(INFO) << "Found Invalid Partition axis.";
            break;
        }
    }
    auto apply_end = std::chrono::system_clock::now();
    auto apply_elapsed_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(apply_end - apply_start).count();
    apply_time = apply_elapsed_time;
    if(min_new_cost < orig_cost || ignore_orig_cost) {
        // partitioned cost is better than original cost
        return std::make_tuple(min_new_cost, best_sched_dfg, best_partition_range, true, true);
    } else {
        // non-partitioned cost is better than partitioned cost
        return std::make_tuple(orig_cost, sched_dfg, partition_range, true, false);
    }
}

bool canFuse(const ScheduledDFG& sched_dfg, const Node* node_a, const Node* node_b,
                    const NodeMap<int>& op_idx_map) {
    // to see whether node_a and node_b can be combined, we need to make sure:
    //  1. we calculate last producer and the first consumer of the ops to fuse
    //  2. two ops can only be fused if 
    //     1. Cycle avoidence: first consumer of op_a must appear after the last producer of op_b
    //     2. We should not alter the order of operators on the same stream
    //
    //                                 <---->                
    //             *           *      *   *   *    *           *
    //             |           |      |   |   |    |           |
    //             |           |      |   |<->|    |           |
    //             |--------- op_a ---|---+---|    |           |
    //                                +---+---+-- op_b --------|
    //                                   op_c
    //
    // When constructing the fused total order, we always move the fused op_b to the earliest spot possible

    auto& dfg = sched_dfg.dfg;
    int range_end = std::numeric_limits<int>::max();
    int range_start = -1;

    for(auto parent: dfg.getParents(node_a)) {
        range_end = std::min(range_end, op_idx_map.at(parent));
    }
    for(auto child: dfg.getChildren(node_b)) {
        range_start = std::max(range_start, op_idx_map.at(child));
    }
    // see if there is any op in between op_a and op_b in the same stream
    CHECK_EQ(dfg.getNodeType(node_a), dfg.getNodeType(node_b));

    int a_idx = op_idx_map.at(node_a);
    int b_idx = op_idx_map.at(node_b);
    NodeType a_type = dfg.getNodeType(node_a);
    for(int i=b_idx - 1; i>=0; i--) {
        if(dfg.getNodeType(sched_dfg.total_order[i]) == a_type) {
            range_start = std::max(range_start, i);
            break;
        }
    }
 
    if(range_start < range_end) {
        return true;
    } else {
        return false;
    }
}

FusionStrategy findBestFusionStrategy(
    const ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler, SimulateTimeType initial_cost, bool skip) {
    if(skip) {
        LOG(INFO) << "Skipped fusion.";
        return std::make_tuple(initial_cost, sched_dfg);
    }

    CHECK(sched_dfg.total_order_valid) << "Input to findBestFusionStrategy must be a scheduled DFG with valid total order.";
    CHECK(sched_dfg.critical_path_valid) << "Input to findBestFusionStrategy must be a scheduled DFG with valid critical path.";

    auto& dfg = sched_dfg.dfg;
    auto& comp_chain = sched_dfg.comp_chain;
    auto& comm_chain = sched_dfg.comm_chain;
    auto& node_durations = sched_dfg.node_durations;
    auto& critical_path = sched_dfg.critical_path;

    // create node to chain idx map
    NodeMap<int> op_idx_map;
    for(int idx=0; idx < sched_dfg.total_order.size(); idx++) {
        op_idx_map[sched_dfg.total_order[idx]] = idx;
    }

    SimulateTimeType min_new_cost = initial_cost;
    ScheduledDFG best_sched_dfg = sched_dfg;

    std::vector<FusionCandidate> fusion_candidates;
    std::vector<SimulateTimeType> sample_weight;

    std::vector<CommType> supported_comm_types = {CommType::kAllReduce, CommType::kReduceScatter, CommType::kAllGather};
    SimulateTimeType max_gap = 2000; // 2ms
    if (const char* max_gap_str = getenv("MAX_FUSION_GAP")) {
        max_gap = std::stod(std::string(max_gap_str));
    }
    for(auto curr_comm_type: supported_comm_types) {
        const Node* source = nullptr;
        CommComponents source_size;
        for (size_t cpid = 1; cpid < critical_path.size(); cpid++) {
            const Node* cpid_node = critical_path[cpid];
            if(dfg.getNodeType(cpid_node) == NodeType::kCompNode) {
                // currently only consider fusing two communications
                continue;
            }
            auto cpid_node_size = dfg.getCommSize(cpid_node);
            auto cpid_node_comm_type = IdentifyCommType(cpid_node_size);
            if(cpid_node_comm_type == curr_comm_type) {
                // check if the two nodes can be combined
                if(source != nullptr && canFuse(sched_dfg, source, cpid_node, op_idx_map)) {
                    // check if the gap between the current node and the source node is less than a threshold
                    auto& node_durations = sched_dfg.node_durations;
                    CHECK(node_durations.count(source)) << "Cannot find node duration of source node " << sched_dfg.dfg.getNodeNameOrDefault(source);
                    auto source_end_time = node_durations.at(source).end;
                    CHECK(node_durations.count(cpid_node)) << "Cannot find node duration of node " << sched_dfg.dfg.getNodeNameOrDefault(cpid_node);
                    auto cpid_start_time = node_durations.at(cpid_node).start;
                    auto gap = cpid_start_time - source_end_time;
                    if(gap <= max_gap) {
                        // valid candidate
                        CommComponents fused_size = source_size;
                        for(auto it: cpid_node_size) {
                            fused_size[it.first] += it.second;
                        }
                        auto orig_source_time = op_profiler.GetCommOpExecTime(source_size);
                        auto orig_target_time = op_profiler.GetCommOpExecTime(cpid_node_size);
                        auto fused_time = op_profiler.GetCommOpExecTime(fused_size);
                        auto reduced_time = orig_source_time + orig_target_time - fused_time;
                        // use local time reduction as sample weight
                        sample_weight.push_back(reduced_time * reduced_time);
                        fusion_candidates.push_back(FusionCandidate({source, cpid_node}));
                    }
                }
                source = cpid_node;
                source_size = cpid_node_size;
            }
        }
    }

    int sample_threshold = 8;
    std::vector<FusionCandidate> sampled_candidates;
    std::vector<FusionCandidate> discarded_candidates;
    if(fusion_candidates.size() > sample_threshold) {
        // down sample to sample_threshold
        std::tie(sampled_candidates, discarded_candidates) = WeightedSample(fusion_candidates, sample_weight, sample_threshold);
    } else {
        sampled_candidates = fusion_candidates;
    }

    auto eval_func = [&](FusionCandidate candidate) {
        const Node* source = candidate.first;
        const Node* target = candidate.second;
        DLOG(INFO) << "Trying to fuse node " << dfg.getNodeNameOrDefault(source) << " and " << dfg.getNodeNameOrDefault(target);

        auto costs_and_new_sched_dfg = GetFusedExecTime(sched_dfg, source, target, op_profiler);
        SimulateTimeType new_cost = std::get<0>(costs_and_new_sched_dfg);
        if (new_cost < min_new_cost) {
            min_new_cost = new_cost;
            best_sched_dfg = std::get<1>(costs_and_new_sched_dfg);
        }
    };

    for(auto candidate: sampled_candidates) {
        eval_func(candidate);
    }
    if(min_new_cost == initial_cost) {
        // cannot find better candidates in sampled ones, try disgarded candidates
        for(auto candidate: discarded_candidates) {
            eval_func(candidate);
        }
    }
    return std::make_tuple(min_new_cost, best_sched_dfg);
}

PartitionStrategy findBestPartitionStrategy(
    const ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler, SimulateTimeType initial_cost, int dp_group_size, int n_experts, bool skip) {
    if(skip) {
        LOG(INFO) << "Skipped partition.";
        return std::make_tuple(initial_cost, sched_dfg);
    }

    if (const char* test_partition = getenv("TEST_PARTITION")) {
        if (strcmp(test_partition, "0") != 0) {
            LOG(INFO) << "Test partition.";
            initial_cost = std::numeric_limits<SimulateTimeType>::max();
        }
    }

    SimulateTimeType min_new_cost = initial_cost;
    ScheduledDFG best_sched_dfg = sched_dfg;

    SimulateTimeType min_idle_time = 2000;
    if (const char* min_idle_time_str = getenv("MIN_IDLE_TIME")) {
        min_idle_time = std::stod(std::string(min_idle_time_str));
    }

    auto start = std::chrono::system_clock::now();
    auto rois = findLatencyCriticalROIs(sched_dfg, dp_group_size, min_idle_time);
    auto end = std::chrono::system_clock::now();
    auto elapsed_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG(INFO) << "Finding ROIS took " << elapsed_time / 1000.0f << " ms.";
    LOG(INFO) << "Found " << rois.size() << " ROIS.";
    LOG(INFO) << "Initial cost " << initial_cost;
    int max_partition = 6;
    if (const char* max_partition_str = getenv("MAX_PARTITION")) {
        max_partition = std::stod(std::string(max_partition_str));
    }
    SimulateTimeType min_reduction_time = 500;
    if (const char* min_reduction_time_str = getenv("MIN_PARTITION_REDUCED_TIME")) {
        min_reduction_time = std::stod(std::string(min_reduction_time_str));
    }
    for(auto& roi: rois) {
        // LOG(INFO) << "Testing on ROI -------------------------------------";
        // LOG(INFO) << PrintROI(sched_dfg, roi);
        // LOG(INFO) << "----------------------------------------------------";
        auto n_partitions = GetPartitionParts(sched_dfg, roi.critical_set, roi.cp_solution, max_partition, dp_group_size);
        for(auto n_partition : n_partitions) {
            auto cost_and_new_sched_dfg = GetPartitionedExecTime(sched_dfg, dp_group_size, n_experts, n_partition, op_profiler, roi);
            SimulateTimeType new_cost = std::get<0>(cost_and_new_sched_dfg);
            LOG(INFO) << "n partition " << n_partition << ", cost " << new_cost / 1000.0;
            if (min_new_cost - new_cost > min_reduction_time || (new_cost < min_new_cost && n_partition == 2)) {
                LOG(INFO) << "Choose " << n_partition << " as optimal partition number.";
                min_new_cost = new_cost;
                best_sched_dfg = std::get<1>(cost_and_new_sched_dfg);
            } else {
                // no need to test higher number of partitions
                break;
            }
        }
    }
    regenerateTotalOrder(best_sched_dfg);
    regenerateExprBasedOnTotalOrder(best_sched_dfg);
    // LOG(INFO) << "Best scheduled partitioned expr " << ir::AsText(best_sched_dfg.scheduled_expr);
    return std::make_tuple(min_new_cost, best_sched_dfg);
}

std::tuple<bool, Nodes, FusionStrategy> findNonCritFusionStrategy(
    const ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler, SimulateTimeType initial_time, const Nodes& list_of_nodes_skipped, bool skip) {
    if(skip) {
        return std::make_tuple(false, list_of_nodes_skipped, std::make_tuple(initial_time, sched_dfg));
    }

    CHECK(sched_dfg.total_order_valid) << "Input to findNonCritFusionStrategy must be a scheduled DFG with valid total order.";

    auto& dfg = sched_dfg.dfg;
    auto& comp_chain = sched_dfg.comp_chain;
    auto& comm_chain = sched_dfg.comm_chain;
    auto& node_durations = sched_dfg.node_durations;

    // create node to chain idx map
    NodeMap<int> op_idx_map;
    for(int idx=0; idx < sched_dfg.total_order.size(); idx++) {
        op_idx_map[sched_dfg.total_order[idx]] = idx;
    }

    // std::vector<CommType> supported_comm_types = {CommType::kAllReduce, CommType::kReduceScatter, CommType::kAllGather};
    std::vector<CommType> supported_comm_types = {CommType::kAllReduce};
    bool skip_phase = true;
    Nodes skipped_nodes = {comm_chain[0]};
    for(auto curr_comm_type: supported_comm_types) {
        for (int cpid = 1; cpid < comm_chain.size(); cpid++) {
            const Node* source = comm_chain[cpid - 1];
            const Node* cpid_node = comm_chain[cpid];
            if (skip_phase && list_of_nodes_skipped.size() > cpid && list_of_nodes_skipped[cpid] == cpid_node && list_of_nodes_skipped[cpid - 1] == source) {
                skipped_nodes.push_back(cpid_node);
                continue;
            } else {
                skip_phase = false;
            }
            CHECK(dfg.getNodeType(cpid_node) == NodeType::kCommNode);
            auto cpid_node_size = dfg.getCommSize(cpid_node);
            auto source_size = dfg.getCommSize(source);
            auto cpid_node_comm_type = IdentifyCommType(cpid_node_size);
            auto source_comm_type = IdentifyCommType(source_size);
            if(cpid_node_comm_type == curr_comm_type && source_comm_type == curr_comm_type) {
                // check if the two nodes can be combined
                if(source != nullptr && canFuse(sched_dfg, source, cpid_node, op_idx_map)) {
                    // estimate fusion gain
                    CommComponents fused_size = source_size;
                    for(auto it: cpid_node_size) {
                        fused_size[it.first] += it.second;
                    }
                    auto orig_source_time = op_profiler.GetCommOpExecTime(source_size);
                    auto orig_target_time = op_profiler.GetCommOpExecTime(cpid_node_size);
                    auto fused_time = op_profiler.GetCommOpExecTime(fused_size);
                    auto& node_durations = sched_dfg.node_durations;
                    CHECK(node_durations.count(cpid_node)) << "Cannot find node duration of node " << sched_dfg.dfg.getNodeNameOrDefault(cpid_node);
                    // get end time of children ops
                    SimulateTimeType source_children_end_time = 0;
                    for (auto node: dfg.getChildren(source)) {
                        if (dfg.getNodeType(node) == NodeType::kCompNode) {
                            source_children_end_time = std::max(source_children_end_time, node_durations.at(node).end);
                        }
                    }
                    SimulateTimeType cpid_children_end_time = 0;
                    for (auto node: dfg.getChildren(cpid_node)) {
                        if (dfg.getNodeType(node) == NodeType::kCompNode) {
                            cpid_children_end_time = std::max(cpid_children_end_time, node_durations.at(node).end);
                        }
                    }
                    auto estimated_end_time = std::max(source_children_end_time, cpid_children_end_time);
                    if (cpid_children_end_time >= 2) {
                        estimated_end_time = std::max(estimated_end_time, node_durations.at(comm_chain[cpid - 2]).end);
                    }
                    if (estimated_end_time < node_durations.at(cpid_node).end) {
                        auto new_time_and_sched_dfg = GetFusedExecTime(sched_dfg, source, cpid_node, op_profiler);
                        if (std::get<0>(new_time_and_sched_dfg) <= initial_time) {
                            return std::make_tuple(true, skipped_nodes, new_time_and_sched_dfg);
                        }
                    }
                }
            }
            skipped_nodes.push_back(cpid_node);
        }
    }
    return std::make_tuple(false, skipped_nodes, std::make_tuple(initial_time, sched_dfg));
}

std::pair<SimulateTimeType, ScheduledDFG> FuseNonCritCommNodes(
    const ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler, bool skip) {

    ScheduledDFG optimized_sched_dfg = sched_dfg;
    SimulateTimeType current_makespan = INFINITY;

    Nodes critical_path;
    SimulateTimeType cp_time_beg_iter;
    std::tie(critical_path, cp_time_beg_iter) = addResourceEdgesAndFindCritPath(optimized_sched_dfg);

    current_makespan = cp_time_beg_iter;

    int fused_comm_counter = 0;
    Nodes list_of_nodes_skipped;
    // loop until no optimization available or max iteration reached
    while (true) {
        // regenerate total order since the optimizations may not preserve total order
        regenerateTotalOrderBasedOnExpr(optimized_sched_dfg);
        // get best makespan after fusion or partition
        auto validity_and_fusion_strategy = findNonCritFusionStrategy(optimized_sched_dfg, op_profiler, current_makespan, list_of_nodes_skipped, skip);
        if(std::get<0>(validity_and_fusion_strategy)) {
            list_of_nodes_skipped = std::get<1>(validity_and_fusion_strategy);
            current_makespan = std::get<0>(std::get<2>(validity_and_fusion_strategy));
            optimized_sched_dfg = std::get<1>(std::get<2>(validity_and_fusion_strategy));
            fused_comm_counter ++;
            if(fused_comm_counter % 10 == 0) {
                LOG(INFO) << "[NonCrit] Fused " << fused_comm_counter << " nodes, nodes left: " << optimized_sched_dfg.comm_chain.size();
            }
        } else {
            // no optimizations left
            break;
        }
    }
    return std::make_pair(current_makespan, optimized_sched_dfg);
}

// for debug purposes
void DumpGroupingResult(const ScheduledDFG& sched_dfg, const std::vector<int>& region_boundaries) {
    std::vector<TraceColor> alt_colors = {TraceColor::kGreen, TraceColor::kBlue};
    NodeColorMap color_map;
    int group_id = 0;
    for (int i=0; i < sched_dfg.total_order.size(); i++) {
        if (i == region_boundaries[group_id]) {
            group_id ++;
        }
        color_map[sched_dfg.total_order[i]] = alt_colors[group_id % 2];
    }
    DumpTraceToJSON("./group_partitioning", sched_dfg, color_map);
    exit(0);
}

void DumpRangePartitionResult(const ScheduledDFG& sched_dfg, const std::vector<std::pair<int, int>>& partition_region_boundaries,  const std::vector<int>& dp_region_boundaries) {
    std::vector<TraceColor> alt_colors = {TraceColor::kGreen, TraceColor::kBlue};
    NodeColorMap color_map;
    for (int i=0; i < sched_dfg.total_order.size(); i++) {
        color_map[sched_dfg.total_order[i]] = TraceColor::kGray;
    }
    int group_id = 0;
    for (auto range: partition_region_boundaries) {
        int start_idx = dp_region_boundaries[range.first];
        int end_idx = dp_region_boundaries[range.second];
        for (int i=start_idx; i < end_idx; i++) {
            color_map[sched_dfg.total_order[i]] = alt_colors[group_id % 2];
        }
        group_id ++;
    }
    DumpTraceToJSON("./group_partitioning", sched_dfg, color_map);
    exit(0);
}

std::tuple<std::vector<int>, int> GroupOpsForPartition(const ScheduledDFG& sched_dfg, const NodeMap<MoENodeLabel>& moe_label_map, bool range_fixed) {
    SimulateTimeType partition_group_size_threshold = -1;
    if (range_fixed) {
        partition_group_size_threshold = 1;
    } else if (const char* partition_group_size_threshold_str = getenv("PARTITION_GROUP_SIZE")) {
        partition_group_size_threshold = std::stoi(partition_group_size_threshold_str);
    } else {
        // automatically infer partition group size
        // we set partition group size to be 1/5 the execution time between two
        // MoE blocks
        // this is found by looking at the distance between the 2nd and 3rd
        // all-to-alls
        SimulateTimeType a2a_distance = 0;
        int a2a_count = 0;
        for (auto node: sched_dfg.comm_chain) {
            if (moe_label_map.at(node).type == MoENodeType::kMoEA2A) {
                a2a_count ++;
                if (a2a_count == 2) {
                    a2a_distance = sched_dfg.node_durations.at(node).end;
                }
                if (a2a_count == 3) {
                    a2a_distance = sched_dfg.node_durations.at(node).start - a2a_distance;
                    break;
                }
            }
        }
        CHECK_GT(a2a_distance, 0) << "Cannot find distance between two all-to-all blocks.";
        partition_group_size_threshold = a2a_distance / 5;
    }
    CHECK_GT(partition_group_size_threshold, 0) << "Partition group size must be positive.";
    // each group should have total exeeution time ~= partition_group_size_threshold
    // IMPORTANTLY, if an op is in a group, then all ops that overlaps with it
    // must be in the same group, otherwise the cost calculation will be wrong
    std::vector<int> dp_region_boundaries; // start index of each group
    dp_region_boundaries.push_back(0); // source op is always in the first group
    dp_region_boundaries.push_back(1);
    SimulateTimeType last_group_end_time = 0;
    SimulateTimeType current_group_end_time = 0;
    bool in_all_to_all_region = false;
    for (int i=1; i < sched_dfg.total_order.size() - 1; i++) {
        auto node = sched_dfg.total_order[i];
        auto node_duration = sched_dfg.node_durations.at(node);
        auto next_node = sched_dfg.total_order[i+1];
        auto next_node_duration = sched_dfg.node_durations.at(next_node);
        auto group_end_time_if_add_this_node = std::max(current_group_end_time, node_duration.end);
        bool should_start_new_group = false;
        if (moe_label_map.at(node).type == MoENodeType::kMoEA2A) {
            in_all_to_all_region = !in_all_to_all_region;
        }
        if (!in_all_to_all_region && !(moe_label_map.at(node).type == MoENodeType::kMoEOtherCompute) &&
            (moe_label_map.at(node).type == MoENodeType::kMoEDispatch ||
            moe_label_map.at(next_node).type == MoENodeType::kMoEDispatch ||
            moe_label_map.at(node).type == MoENodeType::kMoEGather ||
            moe_label_map.at(next_node).type == MoENodeType::kMoEGather)) {
            should_start_new_group = true;
        }
        if (!in_all_to_all_region && (next_node_duration.start >= group_end_time_if_add_this_node &&
            group_end_time_if_add_this_node >= last_group_end_time + partition_group_size_threshold)) {
            should_start_new_group = true;
        }
        if (should_start_new_group) {
            // start a new group from the next op
            dp_region_boundaries.push_back(i+1);
            last_group_end_time = group_end_time_if_add_this_node;
            current_group_end_time = 0;
        } else {
            current_group_end_time = group_end_time_if_add_this_node;
        }
    }
    dp_region_boundaries.push_back(sched_dfg.total_order.size() - 1); // sink op is always in the last group
    return {dp_region_boundaries, partition_group_size_threshold};
}

std::tuple<int, std::vector<int>, int, int> GetNumGroupsBetweenA2As(const ScheduledDFG& sched_dfg, const NodeMap<MoENodeLabel>& moe_label_map, const std::vector<int>& dp_region_boundaries) {
    // find number of groups between two all-to-all blocks
    int n_groups_between_a2a = -1;
    std::vector<int> a2a_group_ids;
    int init_groups = 0;
    int first_a2a_block = -1;
    for (int i=1; i<dp_region_boundaries.size()-1; i++) {
        bool contains_alltoall = false;
        for (int j = dp_region_boundaries[i]; j < dp_region_boundaries[i+1]; j++) {
            if (moe_label_map.at(sched_dfg.total_order[j]).type == MoENodeType::kMoEA2A) {
                contains_alltoall = true;
                break;
            }
        }
        for (int j = dp_region_boundaries[i]; j < dp_region_boundaries[i+1]; j++) {
            if (sched_dfg.dfg.getNodeNameOrDefault(sched_dfg.total_order[j]).find("raf_op_cuda_zeros") != std::string::npos) {
                init_groups = i;
                break;
            }
            if (sched_dfg.dfg.getNodeNameOrDefault(sched_dfg.total_order[j]).find("fused_strided_slice_strided_slice_cast") != std::string::npos) {
                init_groups = i;
                break;
            }
        }
        if (contains_alltoall) {
            a2a_group_ids.push_back(i);
        }
        if (contains_alltoall && n_groups_between_a2a == -1) {
            if (first_a2a_block == -1) {
                first_a2a_block = i;
            } else {
                n_groups_between_a2a = i - first_a2a_block - 1;
            }
        }
    }
    return {n_groups_between_a2a, a2a_group_ids, init_groups, first_a2a_block};
}

std::pair<SimulateTimeType, ScheduledDFG> DPBasedPartition(
    const ScheduledDFG& sched_dfg, const NodeMap<MoENodeLabel>& moe_label_map,
    ExtendedOpProfiler& op_profiler, int dp_group_size, int n_experts) {
    // we record the original pre-partitioned node order
    Nodes original_total_order = sched_dfg.total_order;
    Nodes original_comm_chain = sched_dfg.comm_chain;
    Nodes original_comp_chain = sched_dfg.comp_chain;
    // Bellman equation: dp_cache[i] = min_{j < i} (dp_cache[j] + cost(j+1, i))
    // Partition range is inclusive on start and exclusive on end
    std::vector<std::tuple<SimulateTimeType, ScheduledDFG, PartitionRange>> dp_cache;
    std::unordered_set<int> unpartitionable_region;
    // fill in the base case
    dp_cache.emplace_back(std::make_tuple(0, sched_dfg, std::make_tuple(std::make_pair(0, 0), std::make_pair(0, 0), std::make_pair(0, 0))));
    // to reduce search space, we group some ops together and only partition them together
    // the groups should be small enough to not cause too much unpartitionable region
    // Heuristic: each group contains ops that have roughly the same amount of execution time, PARTITION_GROUP_SIZE, in us
    std::vector<int> dp_region_boundaries;
    int partition_group_size_threshold;
    std::tie(dp_region_boundaries, partition_group_size_threshold) = GroupOpsForPartition(sched_dfg, moe_label_map, /*range_fixed=*/false);
    SimulateTimeType max_partition_range = -1;
    if (const char* max_partition_range_str = getenv("MAX_PARTITION_RANGE")) {
        max_partition_range = std::stoi(max_partition_range_str);
    }
    LOG(INFO) << "Using partition group size threshold: " << partition_group_size_threshold << " us.";
    int max_partition_groups;
    if (max_partition_range == -1) {
        int n_groups_between_a2a;
        std::tie(n_groups_between_a2a, std::ignore, std::ignore, std::ignore) = GetNumGroupsBetweenA2As(sched_dfg, moe_label_map, dp_region_boundaries);
        max_partition_groups = n_groups_between_a2a;
        LOG(INFO) << "Using auto partition range: partitioning up to the distance between two all-to-all ops (~" << n_groups_between_a2a * partition_group_size_threshold << " us)";
    } else {
        LOG(INFO) << "Using max partition range: " << max_partition_range << " us.";
        max_partition_groups = max_partition_range / partition_group_size_threshold;
    }
    // DumpGroupingResult(sched_dfg, dp_region_boundaries);
    // bool skip = false;
    for (int i = 1; i < dp_region_boundaries.size(); i++) {
        LOG(INFO) << "DP progress: " << i+1 << "/" << dp_region_boundaries.size();
        SimulateTimeType min_cost = INFINITY;
        auto start = std::chrono::system_clock::now();
        long long time_spent_before_localcost = 0;
        long long time_spent_after_localcost = 0;
        long long accum_cc_time = 0;
        long long accum_spa_time = 0;
        long long accum_apply_time = 0;
        for (int j = std::max(0, i-max_partition_groups); j < i; j++) {
            // we try to partition group j to i (i exclusive)
            auto partition_start_idx = dp_region_boundaries[j];
            auto partition_end_idx = dp_region_boundaries[i];
            for (auto unpartitionable_region_iter: unpartitionable_region) {
                int unpartitionable_region_start = unpartitionable_region_iter / dp_region_boundaries.size();
                int unpartitionable_region_end = unpartitionable_region_iter % dp_region_boundaries.size();
                if (j <= unpartitionable_region_start && i >= unpartitionable_region_end && min_cost != INFINITY) {
                    continue;
                }
            }
            auto local_cost_start = std::chrono::system_clock::now();
            // get the correct node range for partition
            // we need to use the cached PartitionRange since we modify the DAG during partition
            PartitionRange cached_node_range = std::get<2>(dp_cache[j]);
            NodeRange cached_total_order_range = std::get<0>(cached_node_range);
            NodeRange cached_comp_chain_range = std::get<1>(cached_node_range);
            NodeRange cached_comm_chain_range = std::get<2>(cached_node_range);
            auto& cached_sched_dfg = std::get<1>(dp_cache[j]);
            auto cached_cost = std::get<0>(dp_cache[j]);
            // get correct comm and comp range
            int comp_chain_ops = 0;
            int comm_chain_ops = 0;
            for (int k = cached_total_order_range.second; k < cached_total_order_range.second + partition_end_idx - partition_start_idx; k++) {
                if (cached_sched_dfg.dfg.getNodeType(cached_sched_dfg.total_order[k]) == NodeType::kCommNode) {
                    comm_chain_ops++;
                } else {
                    comp_chain_ops++;
                }
            }
            PartitionRange current_node_range = {
                {cached_total_order_range.second, cached_total_order_range.second + partition_end_idx - partition_start_idx},
                {cached_comp_chain_range.second, cached_comp_chain_range.second + comp_chain_ops},
                {cached_comm_chain_range.second, cached_comm_chain_range.second + comm_chain_ops}
            };
            auto local_cost_end= std::chrono::system_clock::now();
            time_spent_before_localcost += std::chrono::duration_cast<std::chrono::microseconds>(local_cost_end - local_cost_start).count();
            int64_t cc_time = 0;
            int64_t spa_time = 0;
            int64_t apply_time = 0;
            // LOG(INFO) << "Partitioning range: group " << j << " to " << i << ", node id " << std::get<0>(current_node_range).first << " to " << std::get<0>(current_node_range).second;
            auto cost_sched_dfg_and_new_part_range = GetDPLocalCost(cached_sched_dfg, current_node_range, op_profiler, moe_label_map, dp_group_size, n_experts, cc_time, spa_time, apply_time,
                                                            /*skip_non_alltoall=*/true,
                                                            /*max_n_partition=*/0,
                                                            /*ignore_orig_cost=*/false,
                                                            /*skip=*/false);
            // LOG(INFO) << "DP: i: " << i << ", j: " << j << ", local_cost: " << std::get<0>(cost_sched_dfg_and_new_part_range);
            bool is_partitionable = std::get<3>(cost_sched_dfg_and_new_part_range);
            bool is_partitioned = std::get<4>(cost_sched_dfg_and_new_part_range);
            // if (is_partitioned) {
            //     skip = true;
            // }
            if (!is_partitionable) {
                unpartitionable_region.insert(j * dp_region_boundaries.size() + i);
            }
            accum_cc_time += cc_time;
            accum_spa_time += spa_time;
            accum_apply_time += apply_time;
            local_cost_start = std::chrono::system_clock::now();
            SimulateTimeType new_cost = std::get<0>(cost_sched_dfg_and_new_part_range) + cached_cost;
            LOG(INFO) << "Cost found for dp iter (" << i << ", " << j << "): " << new_cost << " = " << cached_cost << " + " << std::get<0>(cost_sched_dfg_and_new_part_range);
            if (new_cost < min_cost) {
                if (min_cost != INFINITY) {
                    LOG(INFO) << "Better cost found for dp iter (" << i << ", " << j << "): " << new_cost << " = " << cached_cost << " + " << std::get<0>(cost_sched_dfg_and_new_part_range);
                }
                min_cost = new_cost;
                auto result_cache = std::make_tuple(new_cost, std::get<1>(cost_sched_dfg_and_new_part_range), std::get<2>(cost_sched_dfg_and_new_part_range));
                if (dp_cache.size() <= i) {
                    dp_cache.emplace_back(std::move(result_cache));
                } else {
                    dp_cache[i] = result_cache;
                }
            }
            local_cost_end = std::chrono::system_clock::now();
            time_spent_after_localcost += std::chrono::duration_cast<std::chrono::microseconds>(local_cost_end - local_cost_start).count();
        }
        auto end = std::chrono::system_clock::now();
        auto elapsed_time = 
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (i > 0) {
            LOG(INFO) << "Each iter in DP took " << elapsed_time / i / 1000.0f << " ms, with cc " << accum_cc_time / i / 1000.0f << " ms, spa " << accum_spa_time / i / 1000.0f << " ms, apply " << accum_apply_time / i / 1000.0f << " ms, "
            << time_spent_before_localcost / i / 1000.0f << " ms before local cost, and " << time_spent_after_localcost / i / 1000.0f << " ms after local cost";
        }
        // calculate unpartitioned cost, for range 0~i
        auto partition_end_idx = dp_region_boundaries[i];
        int comp_chain_ops = 0;
        int comm_chain_ops = 0;
        for (int k = 0; k < partition_end_idx; k++) {
            if (sched_dfg.dfg.getNodeType(sched_dfg.total_order[k]) == NodeType::kCommNode) {
                comm_chain_ops++;
            } else {
                comp_chain_ops++;
            }
        }
        PartitionRange unpartitioned_range = {
                {0, partition_end_idx},
                {0, comp_chain_ops},
                {0, comm_chain_ops}
            };
        auto unpartitioned_cost = GetRangeOrigCost(sched_dfg, unpartitioned_range);
        LOG(INFO) << "Final cost for iter " << i << ": " << std::get<0>(dp_cache[i]) << ", unpartitioned cost: " << unpartitioned_cost;
    }
    // reschedule the final result's comm chain
    auto& optimized_sched_dfg = std::get<1>(dp_cache[dp_cache.size() - 1]);
    Nodes optimized_all_nodes_launch_order, optimized_comp_launch_order, optimized_comm_launch_order;
    NodeMap<NodeDuration> optimized_node_durations;
    // during rescheduling, we only keep alltoall order in the comm chain
    Nodes a2a_only_comm_chain;
    for (auto node: optimized_sched_dfg.comm_chain) {
        if (optimized_sched_dfg.dfg.getNodeType(node) == NodeType::kCommNode) {
            auto expr = optimized_sched_dfg.dfg.getExprFromNode(node);
            if (IsAllToAll(expr) || IsAllToAllv(expr)) {
                a2a_only_comm_chain.push_back(node);
            }
        }
    }
    std::tie(optimized_all_nodes_launch_order, optimized_comp_launch_order, optimized_comm_launch_order, optimized_node_durations) = rescheduleBasedOnCompAndCommOrder(optimized_sched_dfg.dfg, optimized_sched_dfg.comp_chain, a2a_only_comm_chain);
    optimized_sched_dfg.total_order = optimized_all_nodes_launch_order;
    optimized_sched_dfg.comp_chain = optimized_comp_launch_order;
    optimized_sched_dfg.comm_chain = optimized_comm_launch_order;
    optimized_sched_dfg.node_durations = optimized_node_durations;
    optimized_sched_dfg.total_order_valid = true;
    // get the final makespan
    SimulateTimeType optimized_makespan = std::max(optimized_node_durations[optimized_comp_launch_order.back()].end, optimized_node_durations[optimized_comm_launch_order.back()].end);
    return std::make_pair(optimized_makespan, optimized_sched_dfg);
}

std::pair<SimulateTimeType, ScheduledDFG> RangeBasedPartition(
    const ScheduledDFG& sched_dfg, const NodeMap<MoENodeLabel>& moe_label_map,
    ExtendedOpProfiler& op_profiler, int n_groups, int dp_group_size, int n_experts, bool range_fixed) {
    // we record the original pre-partitioned node order
    Nodes original_total_order = sched_dfg.total_order;
    Nodes original_comm_chain = sched_dfg.comm_chain;
    Nodes original_comp_chain = sched_dfg.comp_chain;
    // to reduce search space, we group some ops together and only partition them together
    // the groups should be small enough to not cause too much unpartitionable region
    // Heuristic: each group contains ops that have roughly the same amount of execution time, PARTITION_GROUP_SIZE, in us
    std::vector<int> dp_region_boundaries;
    int max_partition_range;
    int partition_group_size_threshold;
    std::tie(dp_region_boundaries, partition_group_size_threshold) = GroupOpsForPartition(sched_dfg, moe_label_map, range_fixed);
    int last_partitioned_group = -1;
    bool first_a2a = true;
    std::vector<std::pair<int, int>> partitioned_groups;
    // find number of groups between two all-to-all blocks
    int n_groups_between_a2a;
    std::vector<int> a2a_group_ids;
    int init_groups;
    int first_a2a_block;
    std::tie(n_groups_between_a2a, a2a_group_ids, init_groups, first_a2a_block) = GetNumGroupsBetweenA2As(sched_dfg, moe_label_map, dp_region_boundaries);
    if (!range_fixed) {
        if (n_groups_between_a2a < n_groups * 2 && n_groups_between_a2a != -1) {
            LOG(WARNING) << "Not enough groups between all-to-all blocks, maxed out at n_groups=" << (n_groups_between_a2a + 1) / 2 << ".";
            n_groups = (n_groups_between_a2a + 1) / 2;
        }
        for (auto group_id: a2a_group_ids) {
            int start_group_id;
            if (n_groups_between_a2a >= 0 && n_groups == (n_groups_between_a2a + 1) / 2 && n_groups_between_a2a % 2 == 1) {
                start_group_id = std::max(init_groups + 1, group_id - n_groups - 1);
            } else {
                start_group_id = std::max(init_groups + 1, group_id - n_groups);
            }
            int end_group_id = std::min((int)dp_region_boundaries.size() - 1, group_id + 1 + n_groups);
            partitioned_groups.emplace_back(std::make_pair(start_group_id, end_group_id));
        }
    } else {
        // partition every 4 layer norms
        for (auto group_id: a2a_group_ids) {
            int start_group_id = group_id;
            int n_layer_norms = 0;
            while (start_group_id > 0) {
                bool contains_layer_norm = false;
                for (int j = dp_region_boundaries[start_group_id]; j < dp_region_boundaries[start_group_id+1]; j++) {
                    if (sched_dfg.dfg.getNodeNameOrDefault(sched_dfg.total_order[j]).find("layer_norm") != std::string::npos) {
                        contains_layer_norm = true;
                        break;
                    }
                }
                if (contains_layer_norm) {
                    n_layer_norms ++;
                }
                if (n_layer_norms == 2) {
                    break;
                }
                start_group_id --;
            }
            int end_group_id = group_id + 1;
            n_layer_norms = 0;
            while (end_group_id < dp_region_boundaries.size() - 1) {
                bool contains_layer_norm = false;
                for (int j = dp_region_boundaries[end_group_id]; j < dp_region_boundaries[end_group_id+1]; j++) {
                    if (sched_dfg.dfg.getNodeNameOrDefault(sched_dfg.total_order[j]).find("layer_norm") != std::string::npos) {
                        contains_layer_norm = true;
                        break;
                    }
                }
                if (contains_layer_norm) {
                    n_layer_norms ++;
                }
                if (n_layer_norms == 3) {
                    break;
                }
                end_group_id ++;
            }
            partitioned_groups.emplace_back(std::make_pair(start_group_id, end_group_id));
        }
    }

    // debug
    // DumpRangePartitionResult(sched_dfg, partitioned_groups, dp_region_boundaries);

    ScheduledDFG optimized_sched_dfg = sched_dfg;
    // partition each pair
    int total_new_ops = 0;
    int total_new_comm_ops = 0;
    int total_new_comp_ops = 0;
    std::vector<int> comm_ops_start_idx_per_group;
    std::vector<int> comp_ops_start_idx_per_group;
    for (auto range: partitioned_groups) {
        int current_start = dp_region_boundaries[range.first];
        int comp_count = 0;
        int comm_count = 0;
        for (int i=0; i < current_start; i++) {
            if (sched_dfg.dfg.getNodeType(sched_dfg.total_order[i]) == NodeType::kCommNode) {
                comm_count ++;
            } else {
                comp_count ++;
            }
        }
        comm_ops_start_idx_per_group.push_back(comm_count);
        comp_ops_start_idx_per_group.push_back(comp_count);
    }
    for (int i=0; i < partitioned_groups.size(); i++) {
        auto range = partitioned_groups[i];
        auto range_start_op_idx = dp_region_boundaries[range.first];
        auto range_end_op_idx = dp_region_boundaries[range.second];
        // get correct comm and comp range after partition previous groups
        int comp_chain_ops = 0;
        int comm_chain_ops = 0;
        int current_group_start = range_start_op_idx + total_new_ops;
        int current_group_comm_start = comm_ops_start_idx_per_group[i] + total_new_comm_ops;
        int current_group_comp_start = comp_ops_start_idx_per_group[i] + total_new_comp_ops;
        for (int k = current_group_start; k < current_group_start + range_end_op_idx - range_start_op_idx; k++) {
            if (optimized_sched_dfg.dfg.getNodeType(optimized_sched_dfg.total_order[k]) == NodeType::kCommNode) {
                comm_chain_ops++;
            } else {
                comp_chain_ops++;
            }
        }
        PartitionRange current_node_range = {
            {current_group_start, current_group_start + range_end_op_idx - range_start_op_idx},
            {current_group_comp_start, current_group_comp_start + comp_chain_ops},
            {current_group_comm_start, current_group_comm_start + comm_chain_ops}
        };
        // debug: sanity check
        NodeSet all_node_set;
        for (int k = current_group_start; k < current_group_start + range_end_op_idx - range_start_op_idx; k++) {
            all_node_set.insert(optimized_sched_dfg.total_order[k]);
        }
        for (int k = current_group_comm_start; k < current_group_comm_start + comm_chain_ops; k++) {
            CHECK(all_node_set.count(optimized_sched_dfg.comm_chain[k])) << "Comp chain node " << optimized_sched_dfg.dfg.getNodeNameOrDefault(optimized_sched_dfg.comm_chain[k]) << " not in range " << current_group_start << " to " << current_group_start + range_end_op_idx - range_start_op_idx;
        }
        for (int k = current_group_comp_start; k < current_group_comp_start + comp_chain_ops; k++) {
            CHECK(all_node_set.count(optimized_sched_dfg.comp_chain[k])) << "Comm chain node " << optimized_sched_dfg.dfg.getNodeNameOrDefault(optimized_sched_dfg.comp_chain[k]) << " not in range " << current_group_start << " to " << current_group_start + range_end_op_idx - range_start_op_idx;
        }
        int64_t cc_time = 0;
        int64_t spa_time = 0;
        int64_t apply_time = 0;
        // LOG(INFO) << "Partitioning range: group " << j << " to " << i << ", node id " << std::get<0>(current_node_range).first << " to " << std::get<0>(current_node_range).second;
        auto cost_sched_dfg_and_new_part_range = GetDPLocalCost(optimized_sched_dfg, current_node_range, op_profiler, moe_label_map, dp_group_size, n_experts, cc_time, spa_time, apply_time,
                                                        /*skip_non_alltoall=*/true,
                                                        /*max_n_partition=*/0,
                                                        /*ignore_orig_cost=*/true,
                                                        /*skip=*/false);
        bool is_partitionable = std::get<3>(cost_sched_dfg_and_new_part_range);
        bool is_partitioned = std::get<4>(cost_sched_dfg_and_new_part_range);
        // if (! is_partitionable) {
        //     for (int j = current_group_start; j < current_group_start + range_end_op_idx - range_start_op_idx; j++) {
        //         LOG(INFO) << "-> " << optimized_sched_dfg.dfg.getNodeNameOrDefault(optimized_sched_dfg.total_order[j]);
        //     }
        //     CHECK(is_partitionable) << "Partitioning range " << range_start_op_idx << " to " << range_end_op_idx << " is not partitionable.";
        // }
        // CHECK(is_partitioned) << "Partitioning range " << range_start_op_idx << " to " << range_end_op_idx << " is not partitioned.";
        optimized_sched_dfg = std::get<1>(cost_sched_dfg_and_new_part_range);
        auto new_part_range = std::get<2>(cost_sched_dfg_and_new_part_range);
        total_new_ops += (std::get<0>(new_part_range).second - std::get<0>(new_part_range).first) - (range_end_op_idx - range_start_op_idx);
        LOG(INFO) << "Total new ops: " << total_new_ops;
        total_new_comp_ops += (std::get<1>(new_part_range).second - std::get<1>(new_part_range).first) - (comp_chain_ops);
        LOG(INFO) << "Total new comp ops: " << total_new_comp_ops;
        total_new_comm_ops += (std::get<2>(new_part_range).second - std::get<2>(new_part_range).first) - (comm_chain_ops);
        LOG(INFO) << "Total new comm ops: " << total_new_comm_ops;
    }
    Nodes optimized_all_nodes_launch_order, optimized_comp_launch_order, optimized_comm_launch_order;
    NodeMap<NodeDuration> optimized_node_durations;
    // during rescheduling, we only keep alltoall order in the comm chain
    Nodes a2a_only_comm_chain;
    for (auto node: optimized_sched_dfg.comm_chain) {
        if (optimized_sched_dfg.dfg.getNodeType(node) == NodeType::kCommNode) {
            auto expr = optimized_sched_dfg.dfg.getExprFromNode(node);
            if (IsAllToAll(expr) || IsAllToAllv(expr)) {
                a2a_only_comm_chain.push_back(node);
            }
        }
    }
    std::tie(optimized_all_nodes_launch_order, optimized_comp_launch_order, optimized_comm_launch_order, optimized_node_durations) = rescheduleBasedOnCompAndCommOrder(optimized_sched_dfg.dfg, optimized_sched_dfg.comp_chain, a2a_only_comm_chain);
    optimized_sched_dfg.total_order = optimized_all_nodes_launch_order;
    optimized_sched_dfg.comp_chain = optimized_comp_launch_order;
    optimized_sched_dfg.comm_chain = optimized_comm_launch_order;
    optimized_sched_dfg.node_durations = optimized_node_durations;
    optimized_sched_dfg.total_order_valid = true;
    // get the final makespan
    SimulateTimeType optimized_makespan = std::max(optimized_node_durations[optimized_comp_launch_order.back()].end, optimized_node_durations[optimized_comm_launch_order.back()].end);
    return std::make_pair(optimized_makespan, optimized_sched_dfg);
}

std::pair<SimulateTimeType, ScheduledDFG> OptimizeScheduledDFG(
    const ScheduledDFG& sched_dfg, TimelineOptAlgo timeline_opt_algo, ExtendedOpProfiler& op_profiler, int max_iterations, int dp_group_size, int n_experts, bool disable_fusion, bool disable_partition) {

    ScheduledDFG optimized_sched_dfg = sched_dfg;
    SimulateTimeType current_makespan = INFINITY;

    // a critical path based heuristic partitioning algorithm, not used in the 
    // final Lancet paper.
    if (timeline_opt_algo == TimelineOptAlgo::kHeuristic) {
        bool always_apply_partition = false;
        if (const char* always_apply_partition_str = getenv("ALWAYS_APPLY_PARTITION")) {
            if (strcmp(always_apply_partition_str, "0") != 0) {
                LOG(INFO) << "Always apply partition.";
                always_apply_partition = true;
            }
        }

        int fusion_partition_ratio = 10;
        if (const char* fusion_partition_ratio_str = getenv("FUSION_PARTITION_RATIO")) {
            fusion_partition_ratio = std::stoi(std::string(fusion_partition_ratio_str));
        }
        int consecutive_fusion_counter = 0;
        // loop until no optimization available or max iteration reached
        for (int i = 0; i < max_iterations; i++) {
            Nodes critical_path;
            SimulateTimeType cp_time_beg_iter;
            std::tie(critical_path, cp_time_beg_iter) = addResourceEdgesAndFindCritPath(optimized_sched_dfg);
            // LOG(INFO) << "Critical path ops: ";
            // for (auto node: critical_path) {
            //     LOG(INFO) << "\t -> node: " << optimized_sched_dfg.dfg.getNodeNameOrDefault(node) << ", time: " << optimized_sched_dfg.dfg.getNodeExecTime(node);
            // }
            // LOG(INFO) << "Node Durations: ";
            // for (auto it: optimized_sched_dfg.node_durations) {
            //     LOG(INFO) << "\t -> node: " << optimized_sched_dfg.dfg.getNodeNameOrDefault(it.first) << ", duration: " << it.second;
            // }
            LOG_ITER(i) << "Makespan at beginning of iteration: " << cp_time_beg_iter / 1000.0 << " ms.";
            current_makespan = cp_time_beg_iter;

            // regenerate total order since the optimizations may not preserve total order
            regenerateTotalOrderBasedOnExpr(optimized_sched_dfg);
            optimized_sched_dfg.critical_path = critical_path;
            optimized_sched_dfg.critical_path_valid = true;

            // skip fusion/partition is determined independently from disable fusion/partition
            // disable = globally disable the feature, skip = skip in this iteration if needed
            bool skip_fusion = false;
            bool skip_partition = false;
            if (const char* debug_alternate_opt = getenv("DEBUG_ALTERNATE_OPT")) {
                if (strcmp(debug_alternate_opt, "0") != 0) {
                    if(i % 2 == 0){
                        LOG(INFO) << "DEBUG_ALTERNATE_OPT is set, skipping partition.";
                        skip_partition = true;
                    } else {
                        LOG(INFO) << "DEBUG_ALTERNATE_OPT is set, skipping fusion.";
                        skip_fusion = true;
                    }
                }
            }

            // get best makespan after fusion or partition
            disable_fusion = disable_fusion || skip_fusion;
            auto best_fusion_strategy = findBestFusionStrategy(optimized_sched_dfg, op_profiler, current_makespan, disable_fusion);
            SimulateTimeType fused_cost = std::get<0>(best_fusion_strategy);
            LOG(INFO) << "Fused cost is " << fused_cost / 1000.0;

            if(fused_cost == current_makespan || disable_fusion || consecutive_fusion_counter >= fusion_partition_ratio) {
                skip_partition = false;
                consecutive_fusion_counter = 0;
            } else {
                skip_partition = true;
                consecutive_fusion_counter ++;
            }
            skip_partition = disable_partition || skip_partition;
            auto best_partition_strategy = findBestPartitionStrategy(optimized_sched_dfg, op_profiler, current_makespan, dp_group_size, n_experts, skip_partition);
            SimulateTimeType partitioned_cost = std::get<0>(best_partition_strategy);
            LOG(INFO) << "Partitioned cost is " << partitioned_cost / 1000.0;

            if (always_apply_partition) {
                current_makespan = partitioned_cost;
                optimized_sched_dfg = std::get<1>(best_partition_strategy);
                // we first generate a valid total order for all nodes
                regenerateTotalOrder(optimized_sched_dfg);
                // then we reconstruct the new ANF expr using dfg and the total order
                regenerateExprBasedOnTotalOrder(optimized_sched_dfg);
                continue;
            }

            if(std::min(fused_cost, partitioned_cost) < current_makespan) {
                bool choose_fusion_over_partition = fused_cost <= partitioned_cost;
                SimulateTimeType new_cost = std::min(fused_cost, partitioned_cost);
                if (const char* debug_alternate_opt = getenv("DEBUG_ALTERNATE_OPT")) {
                    if (strcmp(debug_alternate_opt, "0") != 0) {
                        if(i % 2 == 0){
                            LOG(INFO) << "DEBUG_ALTERNATE_OPT is set, applying fusion.";
                            choose_fusion_over_partition = true;
                            new_cost = fused_cost;
                        } else {
                            LOG(INFO) << "DEBUG_ALTERNATE_OPT is set, applying partition.";
                            choose_fusion_over_partition = false;
                            new_cost = partitioned_cost;
                        }
                    }
                }
                LOG_ITER(i) << "New cost after optimization: " << new_cost / 1000.0 << " ms (before: " << current_makespan / 1000.0 << "ms)";
                if(choose_fusion_over_partition) {
                    // choose fusion
                    current_makespan = fused_cost;
                    optimized_sched_dfg = std::get<1>(best_fusion_strategy);
                } else {
                    current_makespan = partitioned_cost;
                    optimized_sched_dfg = std::get<1>(best_partition_strategy);
                }
            } else {
                // no optimizations available
                break;
            }
        }
    } else {
        if (!disable_partition) {
            // compute MoE labels
            auto moe_label_result = LabelMoENodes(optimized_sched_dfg.dfg, optimized_sched_dfg.total_order);
            SimulateTimeType cost_before_partition = 0;
            for (auto it: optimized_sched_dfg.node_durations) {
                cost_before_partition = std::max(it.second.end, cost_before_partition);
            }
            if (timeline_opt_algo == TimelineOptAlgo::kDP) {
                // DP algorithm used in the Lancet paper
                LOG(INFO) << "Using DP algorithm to optimize schedule. Cost before optimization: " << cost_before_partition / 1000.0 << " ms.";
                std::tie(current_makespan, optimized_sched_dfg) = DPBasedPartition(optimized_sched_dfg, moe_label_result, op_profiler, dp_group_size, n_experts);
            } else {
                // range based partitioning
                int n_groups = 1;
                if (const char* range_ngroups_str = getenv("LANCET_PARTITION_RANGE_NGROUPS")) {
                    n_groups = std::stoi(std::string(range_ngroups_str));
                }
                bool range_fixed = false;
                if (const char* use_fixed_range_str = getenv("LANCET_PARTITION_RANGE_FIXED")) {
                    if (strcmp(use_fixed_range_str, "1") == 0) {
                        LOG(INFO) << "Using fixed range based algorithm to optimize schedule. Cost before optimization: " << cost_before_partition / 1000.0 << " ms.";
                        range_fixed = true;
                    }
                } else {
                    LOG(INFO) << "Using Range based algorithm to optimize schedule. NGroups: " << n_groups << ", cost before optimization: " << cost_before_partition / 1000.0 << " ms.";
                }
                std::tie(current_makespan, optimized_sched_dfg) = RangeBasedPartition(optimized_sched_dfg, moe_label_result, op_profiler, n_groups, dp_group_size, n_experts, /*range_fixed=*/range_fixed);
            }
            LOG(INFO) << "Cost after optimization: " << current_makespan / 1000.0 << " ms.";
        }
    }
    return std::make_pair(current_makespan, optimized_sched_dfg);
}

// This class transforms a expr in GNF or BBNF into ANF while maintaining a map
// between the original op and new ops (values in the let expressions).
class ExtendedOpScheduler : public StreamSchedulerBase {
 public:
  Expr VisitExpr_(const TupleNode* tuple) override {
    std::vector<Expr> fields;
    for (auto field : tuple->fields) {
      fields.push_back(VisitExpr(field));
    }
    auto new_tuple = Tuple(fields);
    new_tuple->checked_type_ = tuple->checked_type_;
    expr_map_[GetRef<Expr>(tuple)] = new_tuple;
    return let_list_.Push(new_tuple, tuple->checked_type_, true);
  }

  Expr VisitExpr_(const CallNode* call) override {
    std::vector<Expr> args;
    for (const auto& arg : call->args) {
      args.push_back(VisitExpr(arg));
    }
    auto new_call = Call(VisitExpr(call->op), args, call->attrs, call->type_args);
    new_call->checked_type_ = call->checked_type_;
    expr_map_[GetRef<Expr>(call)] = new_call;
    return let_list_.Push(new_call, call->checked_type_, true);
  }

  Expr VisitExpr_(const TupleGetItemNode* tgi) override {
    auto new_tgi = TupleGetItem(VisitExpr(tgi->tuple), tgi->index);
    new_tgi->checked_type_ = tgi->checked_type_;
    expr_map_[GetRef<Expr>(tgi)] = new_tgi;
    return let_list_.Push(new_tgi, tgi->checked_type_, true);
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    // Should also transform a function in GNF or BBNF to ANF for partitioning when closure level > 0.
    // We don't add the transformed function to the let list.
    Arena arena;
    Expr body = func->body;
    auto dfg = CreateDependencyGraph(&arena, body, true);
    auto nodes = dfg.post_dfs_order;
    auto topo_order = dependency_graph::GetTopologicalOrder(nodes);
    NodeMap<Expr> node_expr = {};
    ExprMap<const Node*> dummy = {};
    extractDFGNodeExprRelationship(&dfg, node_expr, dummy);
    ExtendedOpScheduler scheduler;
    Expr ret;
    for (auto node : topo_order) {
      ret = scheduler.VisitExpr(node_expr.at(node));
    }
    Expr new_body = scheduler.Get(ret);
    Function new_func = Function(func->params, new_body, func->ret_type, {}, func->attrs);
    new_func->checked_type_ = func->checked_type_;
    return new_func;
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

// DFG checker for debugging purpose
class CheckDFGExpr {
public:
    CheckDFGExpr(Expr e): ell_(ExplicitLetList::make(e)) {
        for(int i=0; i<ell_->vars.size(); i++) {
            expr_idx_[ell_->exprs[i]] = i;
            var_idx_[ell_->vars[i]] = i;
        }
        LOG(INFO) << "expr size: " << ell_->vars.size();
    }

    void Check(const ExtendedDFG& dfg) {
        bool valid = true;
        LOG(INFO) << "=======================================";
        LOG(INFO) << "Checking validity of each node in dfg:";
        for(auto node: dfg.nodes()) {
            auto expr = dfg.getExprFromNode(node);
            if (expr.defined() && expr_idx_.count(expr)) {
                LOG(INFO) << "\t -> " << dfg.getNodeNameOrDefault(node) << ": valid";
            } else {
                LOG(INFO) << "\t -> " << dfg.getNodeNameOrDefault(node) << ": not found";
                if (dfg.getNodeNameOrDefault(node) != "SOURCE" && dfg.getNodeNameOrDefault(node) != "SINK" ) {
                    valid = false;
                }
            }
        }
        CHECK(valid) << "DFG contains invalid expr.";
        LOG(INFO) << "DFG is valid.";
        LOG(INFO) << "=======================================";
    }

protected:
    std::unique_ptr<ExplicitLetList> ell_;
    ExprIdxMap expr_idx_;
    VarIdxMap var_idx_;

    LetList let_list_;
};

std::pair<SimulateTimeType, ScheduledDFG> RunOptimization(
    Function func, const ExtendedDFG& dfg,
    ScheduleHeuristics heuristic, TimelineOptAlgo timeline_opt_algo,
    CommCostModel& comm_cost_model, int dp_group_size, int n_experts, int max_iterations,
    bool disable_fusion, bool disable_partition) {

    // make a copy of DFG
    ExtendedDFG dfg_for_opt = dfg;

    ExtendedOpProfiler op_profiler(Device::Current(/*allow_default=*/false), comm_cost_model);

    // replace exec_time_map with cost model values
    for(auto node: dfg_for_opt.nodes()) {
      if(dfg_for_opt.getNodeType(node) == NodeType::kCommNode) {
        dfg_for_opt.setNodeExecTime(node, op_profiler.GetCommOpExecTime(dfg_for_opt.getCommSize(node)));
      }
    }

    // used by static schedule
    NodeMap<double> priority;

    calcFIFOPriority(dfg_for_opt, dfg_for_opt.getNodeExecTimeAsMap(), priority,
        /*preserve_comm_prio=*/false,
        /*deprioritize_allreduce=*/false,
        /*deprioritize_weight_update*/true);

    ScheduledDFG best_scheduled_dfg(dfg_for_opt);
    SimulateTimeType best_makespan = INFINITY;

    // find weight update nodes to delay
    NodeSet nodes_to_delay;
    for (auto node: dfg_for_opt.nodes()) {
        if (IsWeightUpdateFirstOp(dfg_for_opt, node)) {
            nodes_to_delay.insert(node);
        }
    }
    // first run a simple FIFO schedule
    NodeMap<NodeDuration> static_node_durations;
    CompareNode static_comp_comparator(priority);
    CompareNode static_comm_comparator(priority);
    StaticScheduleGenerator
        generator(dfg_for_opt, static_comp_comparator, static_comm_comparator);
    SimulateTimeType makespan =  generator.Run(static_node_durations, nodes_to_delay);
    Nodes static_all_nodes_launch_order;
    Nodes static_comp_launch_order;
    Nodes static_comm_launch_order;
    std::tie(static_all_nodes_launch_order, static_comp_launch_order, static_comm_launch_order) = generator.GetNodeOrder();

    if (heuristic == ScheduleHeuristics::kDW) {
        LOG(INFO) << "Makespan before dW schedule: " << makespan / 1000.0 << " ms" << std::endl;
        // reorders dW ops
        auto dw_label_result = LabeldWNodes(dfg_for_opt, static_all_nodes_launch_order);
        float accume_overlappable_time = 0;
        NodeArgsMap args_map;
        for (auto node: std::get<0>(dw_label_result)) {
            std::stringstream ss;
            for (auto a2a_node: node.second) {
                ss << dfg_for_opt.getNodeNameOrDefault(a2a_node) << ", ";
            }
            args_map[node.first] = {{"overlappable ops", ss.str()}};
            accume_overlappable_time += dfg_for_opt.getNodeExecTime(node.first);
        }

        Nodes new_comp_launch_order = GreedyReorderDWNodes(dfg_for_opt, static_all_nodes_launch_order, dw_label_result);

        Nodes unused_comm_order;
        std::tie(static_all_nodes_launch_order, static_comp_launch_order, static_comm_launch_order, static_node_durations) = rescheduleBasedOnCompAndCommOrder(dfg_for_opt, new_comp_launch_order, unused_comm_order);

        LOG(INFO) << "Accumulated overlappable time: " << accume_overlappable_time / 1000.0 << " ms" << std::endl;
    }

    // reorder exprs
    Expr ret;
    auto scheduler = ExtendedOpScheduler();
    for(auto& node: static_all_nodes_launch_order) {
        auto expr = dfg_for_opt.getExprFromNode(node);
        if(expr.defined()) {
            ret = scheduler.VisitExpr(expr);
        }
    }
    // the scheduled expr
    Expr scheduled_expr = scheduler.Get(ret);
    ExprMap<Expr> expr_map = scheduler.GetExprMap();

    // update dfg_for_opt using the mapping
    dfg_for_opt.updateExprFromMap(expr_map);

    // construct scheduled DFG
    ScheduledDFG sched_dfg(dfg_for_opt, scheduled_expr, static_all_nodes_launch_order, static_comp_launch_order,
                            static_comm_launch_order, static_node_durations);
    Nodes critical_path;
    SimulateTimeType critical_path_length;
    std::tie(critical_path, critical_path_length) = addResourceEdgesAndFindCritPath(sched_dfg);
    LOG(INFO) << "Critical path length before timeline optimization: " << critical_path_length / 1000.0 << " ms" << std::endl;
    sched_dfg.setCriticalPath(critical_path);
    // perform partitioning and pipelining based on the scheduled DFG
    std::tie(best_makespan, best_scheduled_dfg) = OptimizeScheduledDFG(sched_dfg, timeline_opt_algo, op_profiler, max_iterations, dp_group_size, n_experts, disable_fusion, disable_partition);
    // reschedule based on compute order
    Nodes opt_all_nodes_launch_order, opt_comp_launch_order, opt_comm_launch_order;
    regenerateExprBasedOnTotalOrder(best_scheduled_dfg);

    // Fuse non critical comm ops to reduce unnecessary overlap interference
    std::tie(best_makespan, best_scheduled_dfg) = FuseNonCritCommNodes(best_scheduled_dfg, op_profiler, disable_fusion);
    LOG(INFO) << "Makespan after fusing non crit nodes: " << best_makespan / 1000.0 << " ms" << std::endl;
    if(const char* dump_simulation_prefix = getenv("SIMULATION_DEBUG_PREFIX")) {
        DumpTraceToJSON(dump_simulation_prefix, best_scheduled_dfg);
    }
    LOG(INFO) << "Found best joint optimization result: " << best_makespan / 1000.0 << " ms" << std::endl;

    return std::make_pair(best_makespan, best_scheduled_dfg);
}

std::pair<bool, ScheduledDFG> RunPartitionOnEntireGraph(const ExtendedDFG& dfg, int dp_group_size, int n_experts) {
    ExtendedDFG dfg_for_opt = dfg;
    NodeMap<double> priority;
    calcFIFOPriority(dfg_for_opt, dfg_for_opt.getNodeExecTimeAsMap(), priority, false);
    NodeMap<NodeDuration> static_node_durations;
    CompareNode static_comp_comparator(priority);
    CompareNode static_comm_comparator(priority);
    StaticScheduleGenerator
        generator(dfg_for_opt, static_comp_comparator, static_comm_comparator);
    SimulateTimeType makespan =  generator.Run(static_node_durations);
    Nodes static_all_nodes_launch_order;
    Nodes static_comp_launch_order;
    Nodes static_comm_launch_order;
    std::tie(static_all_nodes_launch_order, static_comp_launch_order, static_comm_launch_order) = generator.GetNodeOrder();
    // reorder exprs
    Expr ret;
    auto scheduler = ExtendedOpScheduler();
    for(auto& node: static_all_nodes_launch_order) {
        auto expr = dfg_for_opt.getExprFromNode(node);
        if(expr.defined()) {
            ret = scheduler.VisitExpr(expr);
        }
    }
    // the scheduled expr
    Expr scheduled_expr = scheduler.Get(ret);
    ExprMap<Expr> expr_map = scheduler.GetExprMap();
    // update dfg_for_opt using the mapping
    dfg_for_opt.updateExprFromMap(expr_map);
    ScheduledDFG sched_dfg(dfg_for_opt, scheduled_expr, static_all_nodes_launch_order, static_comp_launch_order,
                        static_comm_launch_order, static_node_durations);
    Nodes critical_path;
    SimulateTimeType critical_path_length;
    std::tie(critical_path, critical_path_length) = addResourceEdgesAndFindCritPath(sched_dfg);
    sched_dfg.setCriticalPath(critical_path);
    int64_t cc_time = 0;
    int64_t spa_time = 0;
    int64_t apply_time = 0;
    NodeRange all_node_range = {1, static_all_nodes_launch_order.size() - 1}; // excludes SOURCE and SINK
    NodeRange comp_range = {1, static_comp_launch_order.size() - 1}; // excludes SOURCE and SINK
    NodeRange comm_range = {0, static_comm_launch_order.size()};
    PartitionRange partition_range = {all_node_range, comp_range, comm_range};
    DummyExtendedOpProfiler op_profiler;
    LOG(INFO) << "Partitioning comp nodes: ";
    for (int i=comp_range.first; i<comp_range.second; i++) {
        LOG(INFO) << "\t -> node: " << dfg_for_opt.getNodeNameOrDefault(static_comp_launch_order[i]);
    }
    LOG(INFO) << "Partitioning comm nodes: ";
    for (int i=comm_range.first; i<comm_range.second; i++) {
        LOG(INFO) << "\t -> node: " << dfg_for_opt.getNodeNameOrDefault(static_comm_launch_order[i]);
    }
    auto moe_label_result = LabelMoENodes(sched_dfg.dfg, sched_dfg.total_order);
    auto cost_sched_dfg_and_new_part_range = GetDPLocalCost(sched_dfg, partition_range, op_profiler, moe_label_result, dp_group_size, 
                                                            n_experts, cc_time, spa_time, apply_time,
                                                            /*skip_non_alltoall=*/false,
                                                            /*max_n_partition=*/2,
                                                            /*ignore_orig_cost=*/true);
    bool partitioned = std::get<3>(cost_sched_dfg_and_new_part_range);
    if (partitioned) {
        auto new_sched_dfg = std::get<1>(cost_sched_dfg_and_new_part_range);
        Nodes optimized_all_nodes_launch_order, optimized_comp_launch_order, optimized_comm_launch_order;
        NodeMap<NodeDuration> optimized_node_durations;
        // during rescheduling, we only keep alltoall order in the comm chain
        Nodes a2a_only_comm_chain;
        for (auto node: new_sched_dfg.comm_chain) {
            if (new_sched_dfg.dfg.getNodeType(node) == NodeType::kCommNode) {
                auto expr = new_sched_dfg.dfg.getExprFromNode(node);
                if (IsAllToAll(expr) || IsAllToAllv(expr)) {
                    a2a_only_comm_chain.push_back(node);
                }
            }
        }
        std::tie(optimized_all_nodes_launch_order, optimized_comp_launch_order, optimized_comm_launch_order, optimized_node_durations) = rescheduleBasedOnCompAndCommOrder(new_sched_dfg.dfg, new_sched_dfg.comp_chain, a2a_only_comm_chain);
        new_sched_dfg.total_order = optimized_all_nodes_launch_order;
        new_sched_dfg.comp_chain = optimized_comp_launch_order;
        new_sched_dfg.comm_chain = optimized_comm_launch_order;
        new_sched_dfg.node_durations = optimized_node_durations;
        new_sched_dfg.total_order_valid = true;
        regenerateExprBasedOnTotalOrder(new_sched_dfg);
        // construct expr
        return std::make_pair(true, new_sched_dfg);
    } else {
        return std::make_pair(false, sched_dfg);
    }
}

}  // namespace lancet_optimization
}  // namespace pass
}  // namespace raf

