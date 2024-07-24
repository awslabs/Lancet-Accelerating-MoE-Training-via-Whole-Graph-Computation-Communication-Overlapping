/*!
 * Copyright (c) 2022 by Contributors
 * \file scheduler_utils.cc
 * \brief Define data structures and utility functions for scheduling.
 */
#include "raf/serialization.h"
#include "raf/dialect.h"
#include "../common.h"
#include "../let_list.h"
#include "scheduler_utils.h"

namespace raf {
namespace pass {
namespace scheduler_utils {

std::ostream& operator<< (std::ostream& stream, const MoENodeType& type) {
    static std::unordered_map<MoENodeType, std::string> MoENodeTypeMap {
        {MoENodeType::kNonMoECompute, "NonMoECompute"},
        {MoENodeType::kMoEDispatch, "MoEDispatch"},
        {MoENodeType::kMoEA2A, "MoEA2A"},
        {MoENodeType::kMoEExperts, "MoEExperts"},
        {MoENodeType::kMoEGather, "MoEGather"},
    };
    if (!MoENodeTypeMap.count(type)) {
        LOG(FATAL) << "Unknown MoENodeType: " << static_cast<int>(type);
    }
    stream << MoENodeTypeMap.at(type);
    return stream;
}

NodeMap<double> shortestPathLengthInDAG(
    const ExtendedDFG& G, const NodeMap<double>& weight, const Node* source) {
    NodeMap<double> dist_map;
    for(auto n: G.nodes()) {
        dist_map[n] = INFINITY;
        if(n == source) {
            dist_map[n] = 0;
        }
    }
    auto rev_topo_order = GetTopologicalOrder(G, /*reversed=*/true);
    for (int idx = rev_topo_order.size() - 1; idx >= 0; idx--) {
        auto source = rev_topo_order[idx];
        for(auto target: G.getParents(source)) {
            CHECK(weight.count(source));
            if(dist_map[source] + weight.at(source) < dist_map[target]) {
                dist_map[target] = dist_map[source] + weight.at(source);
            }
        }
    }
    return dist_map;
}

PairwiseDistMap allPairsShortestPathInDAG(const ExtendedDFG& G, const NodeMap<double>& weight) {
    PairwiseDistMap result;
    for(auto n: G.nodes()) {
        result[n] = shortestPathLengthInDAG(G, weight, n);
    }
    return result;
}

std::pair<Nodes, SimulateTimeType> findCriticalPath(const ExtendedDFG& G) {
    NodeMap<double> output_weight;
    NodeMap<const Node*> critical_path_map;

    auto exec_time_map = G.getNodeExecTimeAsMap();
    postOrderAggregateNodes(
        G, exec_time_map, [](double x, double y) { return std::max(x, y); },
        output_weight, &critical_path_map);
    const Node* next_node = G.source();
    Nodes result;
    while (true) {
        result.push_back(next_node);
        if (critical_path_map.count(next_node)) {
            next_node = critical_path_map.at(next_node);
        } else {
            break;
        }
    }
    return std::make_pair(result, output_weight.at(G.source()));
}

std::pair<SimulateTimeType, NodeMap<SimulateTimeType>> findSubgraphExecTime(const ExtendedDFG& G, const NodeMap<SimulateTimeType>& exec_time_map, const Nodes& topo_order, const NodeMap<const Node*>& additional_parent_map) {
    NodeMap<SimulateTimeType> end_time;
    for (auto it: exec_time_map) {
        end_time[it.first] = it.second;
    }
    try {
        for (int i=topo_order.size() -1; i>=0; i--) {
            auto node = topo_order[i];
            CHECK(exec_time_map.count(node));
            SimulateTimeType child_aggregated = 0;
            for (auto parent: G.getParents(node)) {
                if (!exec_time_map.count(parent)) {
                    continue;
                }
                child_aggregated = std::max(end_time[parent], child_aggregated);
            }
            if (additional_parent_map.count(node)) {
                auto parent = additional_parent_map.at(node);
                CHECK(end_time.count(parent));
                child_aggregated = std::max(end_time[parent], child_aggregated);
            }
            end_time[node] += child_aggregated;
        }
    } catch (const std::exception& ex) {
        LOG(WARNING) << "Failed to find subgraph exec time for nodes: (in topo order provided)";
        for (auto node: topo_order) {
            LOG(WARNING) << " -- " << G.getNodeNameOrDefault(node);
        }
        LOG(WARNING) << "Parent map:";
        for (auto it: additional_parent_map) {
            LOG(WARNING) << " -- " << G.getNodeNameOrDefault(it.first) << " --> " << G.getNodeNameOrDefault(it.second);
        }
        LOG(FATAL) << "Exception: " << ex.what();
    }

    // we cannot simply get topo_order[0] since the graph may be disconnected
    SimulateTimeType max_time = 0;
    for (auto it: end_time) {
        max_time = std::max(max_time, it.second);
    }
    return {max_time, end_time};
}

std::pair<Nodes, SimulateTimeType> addResourceEdgesAndFindCritPath(const ScheduledDFG& G) {
    // make a local copy
    ExtendedDFG tmp_dfg = G.dfg;
    for (size_t i=1; i< G.comp_chain.size(); i++) {
        tmp_dfg.createEdge(G.comp_chain[i], G.comp_chain[i-1]);
    }
    for (size_t i=1; i< G.comm_chain.size(); i++) {
        tmp_dfg.createEdge(G.comm_chain[i], G.comm_chain[i-1]);
    }
    return findCriticalPath(tmp_dfg);
}

void regenerateTotalOrder(ScheduledDFG& G) {
    ExtendedDFG tmp_dfg = G.dfg;

    // check resources dependency and add them to dfg if not exists.
    for (size_t i=1; i< G.comp_chain.size(); i++) {
        tmp_dfg.createEdge(G.comp_chain[i], G.comp_chain[i-1]);
    }
    for (size_t i=1; i< G.comm_chain.size(); i++) {
        tmp_dfg.createEdge(G.comm_chain[i], G.comm_chain[i-1]);
    }
    // use FIFO schedule to regenerate order
    NodeMap<double> priority;
    NodeMap<NodeDuration> new_node_durations;
    calcFIFOPriority(tmp_dfg, tmp_dfg.getNodeExecTimeAsMap(), priority, false);
    CompareNode static_comp_comparator(priority);
    CompareNode static_comm_comparator(priority);
    StaticScheduleGenerator
        static_generator(tmp_dfg, static_comp_comparator, static_comm_comparator);
    static_generator.Run(new_node_durations);
    Nodes new_all_nodes_launch_order;
    Nodes new_comp_launch_order;
    Nodes new_comm_launch_order;
    std::tie(new_all_nodes_launch_order, new_comp_launch_order, new_comm_launch_order) = static_generator.GetNodeOrder();
    G.total_order = new_all_nodes_launch_order;
    G.comp_chain = new_comp_launch_order;
    G.comm_chain = new_comm_launch_order;
    G.node_durations = new_node_durations;
    G.total_order_valid = true;
}

// This function makes sure that the total order stored in ScheduledDFG is consistent with its scheduled_expr
// which could be modified after fusion/partition
void regenerateTotalOrderBasedOnExpr(ScheduledDFG& G) {
    std::unique_ptr<ExplicitLetList> let_list(ExplicitLetList::make(G.scheduled_expr));

    // generate new node_duration
    regenerateTotalOrder(G);

    ExtendedDFG tmp_dfg = G.dfg;

    ExprMap<int> expr_idx;
    // LOG(INFO) << "In next iteration: Comm Expr order:";
    for(int i=0; i<let_list->exprs.size(); i++) {
        auto expr = let_list->exprs[i];
        expr_idx[expr] = i;
        auto node = tmp_dfg.getNodeFromExpr(expr);
        CHECK(node) << "Failed to obtain node from expr in dfg. Expr is " << expr;
        if(i != let_list->exprs.size() - 1) {
            auto next_node = tmp_dfg.getNodeFromExpr(let_list->exprs[i+1]);
            CHECK(next_node) << "Failed to obtain node from expr in dfg: expr is: \n" << ir::AsText(let_list->exprs[i+1]) << ",\nPtr: " << let_list->exprs[i+1].get();
            tmp_dfg.createEdge(next_node, node);
        }
        // if(auto call_node = expr.as<CallNode>()) {
        //     if(IsCollectiveOp(call_node->op)) {
        //         LOG(INFO) << "\t -> " << expr;
        //     }
        // }
    }
    // check the all node order in Expr matches with comp order and comm order
    int comp_last_idx = -1;
    for (size_t i=1; i< G.comp_chain.size(); i++) {
        auto curr_node = G.comp_chain[i];
        auto curr_expr = G.dfg.getExprFromNode(curr_node);
        if(curr_expr.defined()) {
            CHECK(expr_idx.count(curr_expr)) << "Node " << G.dfg.getNodeNameOrDefault(curr_node) << ", expr: " << curr_expr << ", is not in the expr_idx map.";
            int curr_idx = expr_idx.at(curr_expr);
            CHECK_GT(curr_idx, comp_last_idx) << "Operators " << G.dfg.getNodeNameOrDefault(G.comp_chain[comp_last_idx]) << " and " << G.dfg.getNodeNameOrDefault(curr_node) << " violates total order.";
            comp_last_idx = curr_idx;
        }
    }

    int comm_last_idx = -1;
    for (size_t i=1; i< G.comm_chain.size(); i++) {
        auto curr_node = G.comm_chain[i];
        auto curr_expr = G.dfg.getExprFromNode(curr_node);
        if(curr_expr.defined()) {
            CHECK(expr_idx.count(curr_expr)) << "Node " << G.dfg.getNodeNameOrDefault(curr_node) << ", expr: " << curr_expr << ", is not in the expr_idx map.";
            int curr_idx = expr_idx.at(curr_expr);
            CHECK_GT(curr_idx, comm_last_idx) << "Operators " << let_list->exprs[comm_last_idx] << " and " << let_list->exprs[curr_idx] << " violates total order.";
            comm_last_idx = curr_idx;
        }
    }

    for (size_t i=1; i< G.comp_chain.size(); i++) {
        tmp_dfg.createEdge(G.comp_chain[i], G.comp_chain[i-1]);
    }
    for (size_t i=1; i< G.comm_chain.size(); i++) {
        tmp_dfg.createEdge(G.comm_chain[i], G.comm_chain[i-1]);
    }
    // use FIFO schedule to regenerate order
    NodeMap<double> priority;
    NodeMap<NodeDuration> new_node_durations;
    calcFIFOPriority(tmp_dfg, tmp_dfg.getNodeExecTimeAsMap(), priority, false);
    CompareNode static_comp_comparator(priority);
    CompareNode static_comm_comparator(priority);
    StaticScheduleGenerator
        static_generator(tmp_dfg, static_comp_comparator, static_comm_comparator);
    SimulateTimeType makespan = static_generator.Run(new_node_durations);
    Nodes new_all_nodes_launch_order;
    Nodes new_comp_launch_order;
    Nodes new_comm_launch_order;
    std::tie(new_all_nodes_launch_order, new_comp_launch_order, new_comm_launch_order) = static_generator.GetNodeOrder();
    G.total_order = new_all_nodes_launch_order;
    G.total_order_valid = true;
}

void regenerateNodeOrdersBasedOnExpr(ScheduledDFG& G) {
    std::unique_ptr<ExplicitLetList> let_list(ExplicitLetList::make(G.scheduled_expr));

    ExtendedDFG tmp_dfg = G.dfg;

    Nodes comp_chain;
    Nodes comm_chain;
    for(int i=0; i<let_list->exprs.size(); i++) {
        auto expr = let_list->exprs[i];
        auto node = tmp_dfg.getNodeFromExpr(expr);
        CHECK(node) << "Failed to obtain node from expr in dfg.";
        if(tmp_dfg.getNodeType(node) == NodeType::kCompNode) {
            comp_chain.push_back(node);
        } else {
            comm_chain.push_back(node);
        }
    }

    for (size_t i=1; i< comp_chain.size(); i++) {
        tmp_dfg.createEdge(comp_chain[i], comp_chain[i-1]);
    }
    for (size_t i=1; i< comm_chain.size(); i++) {
        tmp_dfg.createEdge(comm_chain[i], comm_chain[i-1]);
    }
    // use FIFO schedule to regenerate order
    NodeMap<double> priority;
    NodeMap<NodeDuration> new_node_durations;
    calcFIFOPriority(tmp_dfg, tmp_dfg.getNodeExecTimeAsMap(), priority, false);
    CompareNode static_comp_comparator(priority);
    CompareNode static_comm_comparator(priority);
    StaticScheduleGenerator
        static_generator(tmp_dfg, static_comp_comparator, static_comm_comparator);
    SimulateTimeType makespan = static_generator.Run(new_node_durations);
    Nodes new_all_nodes_launch_order;
    Nodes new_comp_launch_order;
    Nodes new_comm_launch_order;
    std::tie(new_all_nodes_launch_order, new_comp_launch_order, new_comm_launch_order) = static_generator.GetNodeOrder();
    G.total_order = new_all_nodes_launch_order;
    G.comp_chain = new_comp_launch_order;
    G.comm_chain = new_comm_launch_order;
    G.node_durations = new_node_durations;
    G.total_order_valid = true;
}

void regenerateExprBasedOnTotalOrder(ScheduledDFG& G) {
    std::unique_ptr<ExplicitLetList> ell(ExplicitLetList::make(G.scheduled_expr));
    LetList ll;
    for(auto node: G.total_order) {
        Expr expr = G.dfg.getExprFromNode(node);
        if(expr.defined() && G.expr_var_map.count(expr)) {
            Var var = G.expr_var_map.at(expr);
            ll.Push(var, expr);
        }
    }
    G.scheduled_expr = ll.Get(ell->ret);
}

void DumpTraceToJSON(std::string dump_simulation_prefix, const ScheduledDFG& sched_dfg, const NodeColorMap& color_map, const NodeArgsMap& extra_args) {
    std::string dump_path(dump_simulation_prefix);
    dump_path += "_" + std::to_string(getpid()) + ".json";
    std::ofstream dump_file;
    dump_file.exceptions(std::ofstream::failbit);
    LOG(INFO) << "Dumping simulated schedule result to " << dump_path << ".";
    try {
        dump_file.open(dump_path);
        dump_file << "{" << std::endl;
        dump_file << "    \"traceEvents\": [" << std::endl;
        for(auto it: sched_dfg.node_durations) {
            std::string node_name;
            
            if (sched_dfg.dfg.hasNodeName(it.first)) {
                node_name = sched_dfg.dfg.getNodeNameOrDefault(it.first);
            } else {
                continue;
            }
            dump_file << "{\n\"name\": \"" << node_name << "\",\n"
                << "\"ph\": \"X\",\n"
                << "\"pid\": " << static_cast<int>(sched_dfg.dfg.getNodeType(it.first)) << ",\n"
                << "\"tid\": 0,\n"
                << "\"ts\": " << it.second.start << ",\n"
                << "\"dur\": " << it.second.end - it.second.start << ",\n";
            if (color_map.count(it.first)) {
                dump_file << "\"cname\": \"" << getTraceColor(color_map.at(it.first)) << "\",\n";
            }
            dump_file << "\"args\": {";
                    //   << "\n\t\"blocked_by\": \"" << sched_dfg.dfg.getNodeNameOrDefault(it.second.blocked_by) << "\",\n";
            if(!it.second.critical_sets.empty()) {
                dump_file << "\t\"in_crit_set\": [";
                for(int i=0; i<it.second.critical_sets.size(); i++) {
                    dump_file << it.second.critical_sets[i];
                    if(i != it.second.critical_sets.size()-1) {
                        dump_file << ", ";
                    }
                }
                dump_file << "],\n";
            }
            if (extra_args.count(it.first)) {
                for(auto arg: extra_args.at(it.first)) {
                    dump_file << "\t\"" << arg.first << "\": \"" << arg.second << "\",\n";
                }
            }
            dump_file << "\t\"ready_time\": " << it.second.ready_time << "}\n},\n";
        }
        dump_file << "{ \"name\": \"process_name\",\n"
        << "\"ph\": \"M\",\n"
        << "\"pid\": " << static_cast<int>(NodeType::kCompNode) << ",\n"
        << "\"tid\": 0,\n"
        << "\"args\": {\"name\": \"Computation Stream\"}},\n"
        << "{ \"name\": \"process_name\",\n"
        << "\"ph\": \"M\",\n"
        << "\"pid\": " << static_cast<int>(NodeType::kCommNode) << ",\n"
        << "\"tid\": 0,\n"
        << "\"args\": {\"name\": \"Communication Stream\"}}\n";
        dump_file << "    ]," << std::endl;
        dump_file << "    \"displayTimeUnit\": \"ms\"" << std::endl;
        dump_file << "}" << std::endl;
        dump_file.close();
    } catch (std::ofstream::failure& e) {
        LOG(FATAL) << "Failed to write to dump file " << dump_path << ": " << e.what();
    }
}

void DumpExprToJSON(std::string dump_prefix, const Expr& expr) {
    // serialize and dump module
    auto module_json = ir::serialization::SaveJSON(expr);
    std::string module_dump_path(dump_prefix);
    module_dump_path += ".mod";
    std::ofstream module_dump_file;
    module_dump_file.exceptions(std::ofstream::failbit);
    LOG(INFO) << "Dumping module to " << module_dump_path << ".";
    try {
    module_dump_file.open(module_dump_path);
    module_dump_file << module_json;
    module_dump_file.close();
    } catch (std::ofstream::failure& e) {
        LOG(FATAL) << "Failed to write to dump file " << module_dump_path << ".";
    }
}

dWLabelingResult LabeldWNodes(const ExtendedDFG& G, const Nodes& all_nodes_launch_order) {
    /*
     * A dW node is a node that does not establish partial order with some
     * all-to-all nodes. This function labels all dW nodes in the graph and
     * computes overlappable all-to-all nodes for each dW.
     */

    Nodes all_to_all_nodes;
    // find all all-to-all nodes
    for(auto node: all_nodes_launch_order) {
        auto expr = G.getExprFromNode(node);
        if (IsAllToAll(expr)) {
            all_to_all_nodes.push_back(node);
        }
    }
    // compute reachability matrix between all nodes
    // node_1 in reachability[node_0] if node_1 must be scheduled after node_0
    NodeMap<NodeSet> reachability;
    for(auto node: all_nodes_launch_order) {
        std::queue<const Node*> q;
        q.push(node);
        while(!q.empty()) {
            auto cur_node = q.front();
            q.pop();
            for(auto succ: G.getParents(cur_node)) {
                if (reachability[node].count(succ)) {
                    continue;
                }
                q.push(succ);
                reachability[node].insert(succ);
            }
        }
    }
    // for each node, find overlappable all-to-all nodes
    NodeSet all_to_all_node_set(all_to_all_nodes.begin(), all_to_all_nodes.end());
    NodeMap<Nodes> overlappable_all_to_all_nodes_map;
    for(auto node: all_nodes_launch_order) {
        if (all_to_all_node_set.count(node) || (G.getNodeType(node) != NodeType::kCompNode)) {
            continue;
        }
        // NodeSet overlappable_all_to_all_nodes;
        Nodes overlappable_all_to_all_nodes;
        for(auto all_to_all_node: all_to_all_nodes) {
            if (!reachability[all_to_all_node].count(node) && !reachability[node].count(all_to_all_node)) {
                // overlappable_all_to_all_nodes.insert(all_to_all_node);
                overlappable_all_to_all_nodes.push_back(all_to_all_node);
            }
        }
        if (!overlappable_all_to_all_nodes.empty()) {
            overlappable_all_to_all_nodes_map[node] = overlappable_all_to_all_nodes;
        }
    }
    LOG(INFO) << "Found " << overlappable_all_to_all_nodes_map.size() << " dW nodes out of " << all_nodes_launch_order.size() << " nodes.";
    // eliminate nodes that are not dW
    NodeSet allreduce_nodes;
    for(auto node: all_nodes_launch_order) {
        auto expr = G.getExprFromNode(node);
        if (IsAllReduce(expr)) {
            allreduce_nodes.insert(node);
        }
    }
    NodeSet nodes_to_ignore;
    for (auto node: allreduce_nodes) {
        for (auto it: overlappable_all_to_all_nodes_map) {
            if (reachability[node].count(it.first)) {
                // it is after an allreduce node
                // isDWNode[it.first] = false;
                nodes_to_ignore.insert(it.first);
            }
        }
    }
    for (auto node: all_nodes_launch_order) {
        // ignore init nodes like raf_op_cuda_zeros and fused_strided_slice_strided_slice_cast
        if (G.getNodeNameOrDefault(node).find("raf_op_cuda_zeros") != std::string::npos) {
            nodes_to_ignore.insert(node);
        } else if (G.getNodeNameOrDefault(node).find("fused_strided_slice_strided_slice_cast") != std::string::npos) {
            nodes_to_ignore.insert(node);
        }
    }
    for (auto node: nodes_to_ignore) {
        overlappable_all_to_all_nodes_map.erase(node);
    }
    LOG(INFO) << "After erase non-dW nodes: " << overlappable_all_to_all_nodes_map.size() << " dW nodes out of " << all_nodes_launch_order.size() << " nodes.";

    // calculate the predecessors of each dW node
    NodeMap<NodeSet> dW_node_predecessors;

    for (auto it: overlappable_all_to_all_nodes_map) {
        NodeSet predecessors;
        for (auto other_it: overlappable_all_to_all_nodes_map) {
            if (reachability[other_it.first].count(it.first)) {
                predecessors.insert(other_it.first);
            }
        }
        dW_node_predecessors[it.first] = predecessors;
    }

    return {overlappable_all_to_all_nodes_map, dW_node_predecessors, nodes_to_ignore};
}

NodeMap<MoENodeLabel> LabelMoENodes(const ExtendedDFG& G, const Nodes& all_nodes_launch_order) {
    // compute MoE label for each node. we only care about FW nodes for now.
    NodeMap<MoENodeLabel> moe_label_map;
    const Op& moe_encode_op = Op::Get("raf.op.moe_encode");
    const Op& moe_encode_bpr_op = Op::Get("raf.op.moe_encode_batch_prioritized");
    const Op& moe_decode_op = Op::Get("raf.op.moe_decode");
    int current_moe_layer = 0;
    bool in_moe_layer = false;
    bool between_a2a = false;
    for (auto node: all_nodes_launch_order) {
        auto expr = G.getExprFromNode(node);
        if(IsOp(expr, moe_encode_op) || IsOp(expr, moe_encode_bpr_op)) {
            moe_label_map[node] = {MoENodeType::kMoEDispatch, current_moe_layer};
            in_moe_layer = true;
        } else if (in_moe_layer && IsAllToAll(expr)) {
            moe_label_map[node] = {MoENodeType::kMoEA2A, current_moe_layer};
            between_a2a = !between_a2a;
        } else if (in_moe_layer && IsOp(expr, moe_decode_op)) {
            moe_label_map[node] = {MoENodeType::kMoEGather, current_moe_layer};
            in_moe_layer = false;
            current_moe_layer ++;
        } else if (in_moe_layer) {
            if (between_a2a) {
                moe_label_map[node] = {MoENodeType::kMoEExperts, current_moe_layer};
            } else {
                moe_label_map[node] = {MoENodeType::kMoEOtherCompute, current_moe_layer};
            }
        } else {
            moe_label_map[node] = {MoENodeType::kNonMoECompute, -1};
        }
    }
    return moe_label_map;
}


Nodes GreedyReorderDWNodes(const ExtendedDFG& G, const Nodes& all_nodes_launch_order, const dWLabelingResult& node_labeling_result) {
    /*
     * Reorder dW nodes to maximize overlap with all-to-all nodes.
     */

    // maps each dW node to overlappable a2as
    const auto& dW_to_overlappable_a2as = std::get<0>(node_labeling_result);
    // maps each dW node to its predecessors
    auto dW_node_predecessors = std::get<1>(node_labeling_result);
    // nodes that should be ignored in the output comp order
    // they are rescheduled using priority based scheduler
    const auto& nodes_to_ignore = std::get<2>(node_labeling_result);
    NodeMap<NodeSet> dW_node_successors;

    for (auto it: dW_node_predecessors) {
        auto node = it.first;
        auto& predecessors = it.second;
        for (auto predecessor: predecessors) {
            dW_node_successors[predecessor].insert(node);
        }
    }

    // construct a reverse map from all-to-all nodes to overlappable dW groups
    NodeMap<Nodes> rev_overlappable_nodes_map;
    // the order is not important
    for (auto node = all_nodes_launch_order.begin(); node != all_nodes_launch_order.end(); node++) {
        if (dW_to_overlappable_a2as.count(*node)) {
            for (auto a2a_node: dW_to_overlappable_a2as.at(*node)) {
                rev_overlappable_nodes_map[a2a_node].push_back(*node);
            }
        }
    }


    // calculate node group assignment
    NodeMap<bool> dw_is_available;
    NodeMap<bool> dw_is_used;
    NodeMap<Nodes> all2all_to_overlapped_dw;

    // set initially available dWs
    for (auto it: dW_node_predecessors) {
        if (it.second.empty()) {
            dw_is_available[it.first] = true;
        } else {
            dw_is_available[it.first] = false;
        }
    }

    for(auto node: all_nodes_launch_order) {
        if (IsAllToAll(G.getExprFromNode(node))) {
            SimulateTimeType all2all_time = G.getNodeExecTime(node);
            int n_dws_overlapped = 0;
            while (true) {
                // find best fit available dw
                const Node* best_dw = nullptr;
                SimulateTimeType best_dw_time = std::numeric_limits<SimulateTimeType>::max();
                SimulateTimeType best_fit_gap = std::numeric_limits<SimulateTimeType>::max();
                for (auto overlappable_dw: rev_overlappable_nodes_map[node]) {
                    if (dw_is_used[overlappable_dw] || !dw_is_available[overlappable_dw]) {
                        continue;
                    }
                    SimulateTimeType dw_time = G.getNodeExecTime(overlappable_dw);
                    SimulateTimeType gap = std::abs(dw_time - all2all_time);
                    if (gap < best_fit_gap) {
                        best_fit_gap = gap;
                        best_dw = overlappable_dw;
                        best_dw_time = dw_time;
                    }
                }
                if (best_dw == nullptr || (best_dw_time > 2 * all2all_time && n_dws_overlapped > 0)) {
                    // no suitable dw
                    break;
                }
                // if still all2all to overlap and we are not wasting too much
                // i.e. remaining gap < 2x best fit dw time (or there is only one dw to overlap)
                all2all_time -= best_dw_time;
                all2all_to_overlapped_dw[node].push_back(best_dw);
                dw_is_used[best_dw] = true;
                // unlock other dWs
                if (dW_node_successors.count(best_dw)) {
                    for (auto successor: dW_node_successors[best_dw]) {
                        dW_node_predecessors[successor].erase(best_dw);
                        if (dW_node_predecessors[successor].empty()) {
                            dw_is_available[successor] = true;
                        }
                    }
                }
                n_dws_overlapped ++;
                if (all2all_time <= 0) {
                    break;
                }
            }
            // also schedule all dWs whose last overlappable all2all is this one
            NodeSet must_schedule_dws;
            for (auto it: dW_to_overlappable_a2as) {
                if (it.second.back() == node && !dw_is_used[it.first]) {
                    must_schedule_dws.insert(it.first);
                }
            }
            std::queue<const Node*> q;
            for (auto dw: must_schedule_dws) {
                q.push(dw);
            }
            while (!q.empty()) {
                auto dw = q.front();
                q.pop();
                for (auto pred: dW_node_predecessors[dw]) {
                    if (must_schedule_dws.count(pred)) {
                        continue;
                    }
                    must_schedule_dws.insert(pred);
                    q.push(pred);
                }
            }
            while (true) {
                bool new_dws_scheduled = false;
                NodeSet dws_used_in_this_round;
                for (auto dw: must_schedule_dws) {
                    CHECK(!dw_is_used[dw]);
                    if (!dw_is_available[dw]) {
                        continue;
                    }
                    SimulateTimeType dw_time = G.getNodeExecTime(dw);
                    all2all_to_overlapped_dw[node].push_back(dw);
                    dw_is_used[dw] = true;
                    dws_used_in_this_round.insert(dw);
                    new_dws_scheduled = true;
                    // unlock other dWs
                    if (dW_node_successors.count(dw)) {
                        for (auto successor: dW_node_successors[dw]) {
                            dW_node_predecessors[successor].erase(dw);
                            if (dW_node_predecessors[successor].empty()) {
                                dw_is_available[successor] = true;
                            }
                        }
                    }
                }
                for (auto dw: dws_used_in_this_round) {
                    must_schedule_dws.erase(dw);
                }
                CHECK(new_dws_scheduled || must_schedule_dws.empty());
                if (must_schedule_dws.empty()) {
                    break;
                }
            }
        }
    }

    // reorder ops, put every overlapped dw right after its all2all
    Nodes new_all_nodes_launch_order;
    Nodes new_comp_launch_order;
    for (auto node: all_nodes_launch_order) {
        if (!dW_to_overlappable_a2as.count(node)) {
            new_all_nodes_launch_order.push_back(node);
        } else {
            CHECK(dw_is_used.count(node));
        }
        if (all2all_to_overlapped_dw.count(node)) {
            for(auto overlapped_dw: all2all_to_overlapped_dw[node]) {
                new_all_nodes_launch_order.push_back(overlapped_dw);
            }
        }
    }
    for (auto node: new_all_nodes_launch_order) {
        if (G.getNodeType(node) == NodeType::kCompNode && !nodes_to_ignore.count(node)) {
            new_comp_launch_order.push_back(node);
        }
    }
    return new_comp_launch_order;
}

std::tuple<Nodes, Nodes, Nodes, NodeMap<NodeDuration>> rescheduleBasedOnCompAndCommOrder(
    const ExtendedDFG& G, 
    const Nodes& comp_launch_order,
    const Nodes& comm_launch_order) {
    ExtendedDFG tmp_dfg = G;

    // check resources dependency and add them to dfg if not exists.
    for (size_t i=1; i< comp_launch_order.size(); i++) {
        tmp_dfg.createEdge(comp_launch_order[i], comp_launch_order[i-1]);
    }
    for (size_t i=1; i< comm_launch_order.size(); i++) {
        tmp_dfg.createEdge(comm_launch_order[i], comm_launch_order[i-1]);
    }
    // use FIFO schedule to regenerate order
    // we also delay weight update nodes to the end of the schedule if
    // they are not a part of the specified comp chain
    NodeSet comp_chain_nodes = NodeSet(comp_launch_order.begin(), comp_launch_order.end());
    NodeSet nodes_to_delay;
    for (auto node: G.nodes()) {
        if (IsWeightUpdateFirstOp(G, node)) {
            nodes_to_delay.insert(node);
        }
    }
    // here we first set the exec time of all all-reduce to 0 and obtain
    // timeine without all-reduce. Then we use the timeline to calculate
    // the schedule of all-reduce.
    NodeMap<SimulateTimeType> all_reduce_time_map;
    for (auto node: G.nodes()) {
        if (IsAllReduce(tmp_dfg.getExprFromNode(node))) {
            all_reduce_time_map[node] = tmp_dfg.getNodeExecTime(node);
            tmp_dfg.setNodeExecTime(node, 0);
        }
    }
    NodeMap<double> priority;
    NodeMap<NodeDuration> new_node_durations;
    calcFIFOPriority(tmp_dfg, tmp_dfg.getNodeExecTimeAsMap(), priority, false,
                    /* deprioritize_allreduce= */true, 
                    /* deprioritize_weight_update= */true);
    CompareNode static_comp_comparator(priority);
    CompareNode static_comm_comparator(priority);
    StaticScheduleGenerator
        static_generator(tmp_dfg, static_comp_comparator, static_comm_comparator);
    static_generator.Run(new_node_durations, nodes_to_delay);
    Nodes new_all_nodes_launch_order;
    Nodes new_comp_launch_order;
    Nodes new_comm_launch_order;
    std::tie(new_all_nodes_launch_order, new_comp_launch_order, new_comm_launch_order) = static_generator.GetNodeOrder();
    NodeSet available_all_reduce_nodes;
    Nodes new_comm_launch_order_with_all_reduce;
    SimulateTimeType last_a2a_end = -1;
    for (auto node: new_comm_launch_order) {
        if (IsAllToAll(tmp_dfg.getExprFromNode(node))) {
            if (last_a2a_end > 0 && !new_comm_launch_order_with_all_reduce.empty()) {
                // found all-to-all gap with available all-reduce nodes
                SimulateTimeType a2a_gap = new_node_durations[node].start - last_a2a_end;
                while (!available_all_reduce_nodes.empty()) {
                    // best fit
                    const Node* best_ar_node = nullptr;
                    SimulateTimeType best_ar_time = std::numeric_limits<SimulateTimeType>::max();
                    SimulateTimeType best_fit_gap = std::numeric_limits<SimulateTimeType>::max();
                    for (auto ar_node: available_all_reduce_nodes) {
                        CHECK(all_reduce_time_map.count(ar_node));
                        auto ar_time = all_reduce_time_map.at(ar_node);
                        if (ar_time <= a2a_gap) {
                            auto fit_gap = std::abs(ar_time - a2a_gap);
                            if (fit_gap < best_fit_gap) {
                                best_fit_gap = fit_gap;
                                best_ar_node = ar_node;
                                best_ar_time = ar_time;
                            }
                        }
                    }
                    if (best_ar_node == nullptr) {
                        break;
                    } else {
                        // schedule this all-reduce node
                        new_comm_launch_order_with_all_reduce.push_back(best_ar_node);
                        available_all_reduce_nodes.erase(best_ar_node);
                        a2a_gap -= best_ar_time;
                        if (a2a_gap <= 0) {
                            break;
                        }
                    }
                }
            }
            last_a2a_end = new_node_durations[node].end;
            new_comm_launch_order_with_all_reduce.push_back(node);
        } else if (IsAllReduce(tmp_dfg.getExprFromNode(node))) {
            available_all_reduce_nodes.insert(node);
        }
    }
    // restore all-reduce exec time in the dfg
    for (auto node: G.nodes()) {
        if (IsAllReduce(tmp_dfg.getExprFromNode(node))) {
            tmp_dfg.setNodeExecTime(node, all_reduce_time_map[node]);
        }
    }
    // create new edges for all-reduce nodes
    for (size_t i=1; i< new_comm_launch_order_with_all_reduce.size(); i++) {
        tmp_dfg.createEdge(new_comm_launch_order_with_all_reduce[i], new_comm_launch_order_with_all_reduce[i-1]);
    }
    // make all remaining all-reduce nodes depend on the last all-to-all node
    for (auto node: available_all_reduce_nodes) {
        tmp_dfg.createEdge(node, new_comm_launch_order_with_all_reduce.back());
    }
    // recalculate priority
    calcFIFOPriority(tmp_dfg, tmp_dfg.getNodeExecTimeAsMap(), priority, false,
                    /* deprioritize_allreduce= */false, 
                    /* deprioritize_weight_update= */true);
    CompareNode new_static_comp_comparator = CompareNode(priority);
    CompareNode new_static_comm_comparator = CompareNode(priority);
    StaticScheduleGenerator
        new_static_generator(tmp_dfg, new_static_comp_comparator, new_static_comm_comparator);
    new_static_generator.Run(new_node_durations, nodes_to_delay);
    std::tie(new_all_nodes_launch_order, new_comp_launch_order, new_comm_launch_order) = new_static_generator.GetNodeOrder();
    return {new_all_nodes_launch_order, new_comp_launch_order, new_comm_launch_order, new_node_durations};
}

}  // namespace scheduler_utils
}  // namespace pass
}  // namespace raf