/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file roi_utils.cc
 * \brief Utility functions to find Regions of Interest (ROI).
 */
#include <thread>
#include "roi_utils.h"

namespace raf {
namespace pass {
namespace roi_utils {

using IdentifyCSArgs = std::tuple<const Node*, const Node*, SimulateTimeType>;

// extends the region to include idx
void NodeChainRegion::update(int idx) {
    if(idx < start_index) {
        start_index = idx;
    }
    if(idx > end_index) {
        end_index = idx;
    }
}

// extends the region to cover another region
void NodeChainRegion::update(const NodeChainRegion& region) {
    if(region.start_index < start_index) {
        start_index = region.start_index;
    }
    if(region.end_index > end_index) {
        end_index = region.end_index;
    }
}

RegionOfInterest::RegionOfInterest() {}

RegionOfInterest::RegionOfInterest(const NodeSet& critical_set, const Node* source_node,
                                   int source_chain_start, int source_chain_end,
                                   int target_chain_start, int target_chain_end,
                                   const CPSolution& cp_solution):
                                   critical_set(critical_set), source_node(source_node),
                                   cp_solution(cp_solution) {
    source_chain.start_index = source_chain_start;
    source_chain.end_index = source_chain_end;
    target_chain.start_index = target_chain_start;
    target_chain.end_index = target_chain_end;
}

RegionOfInterest::RegionOfInterest(const NodeSet& critical_set, const Node* source_node,
                                   NodeChainRegion source_chain, NodeChainRegion target_chain,
                                   const CPSolution& cp_solution):
                                   critical_set(critical_set), source_node(source_node),
                                   source_chain(source_chain), target_chain(target_chain),
                                   cp_solution(cp_solution) {}

std::string PrintROI(const ScheduledDFG& sched_dfg, const RegionOfInterest& roi) {
    auto& dfg = sched_dfg.dfg;
    std::ostringstream os;
    os << "\nCritical Set:" << std::endl;
    for(auto node: roi.critical_set) {
        os << "\t-> " << dfg.getNodeNameOrDefault(node) << std::endl;
    }
    auto source_type = dfg.getNodeType(roi.source_node);
    auto& source_chain = type_select(sched_dfg.comp_chain, sched_dfg.comm_chain, source_type);
    auto& target_chain = type_select(sched_dfg.comp_chain, sched_dfg.comm_chain, to_other(source_type));
    os << "Source Chain Region:" << std::endl;
    for(int i=roi.source_chain.start_index; i<= roi.source_chain.end_index; i++) {
        os << "\t-> " << dfg.getNodeNameOrDefault(source_chain[i]) << std::endl;
    }
    os << "Target Chain Region:" << std::endl;
    for(int i=roi.target_chain.start_index; i<= roi.target_chain.end_index; i++) {
        os << "\t-> " << dfg.getNodeNameOrDefault(target_chain[i]) << std::endl;
    }
    return os.str();
}

std::pair<NodeRelationMap, NodeRelationMap> buildLocalReachabilityMap(const ExtendedDFG& dfg, NodeSet nodes) {
    NodeRelationMap reachable_children_map;
    NodeRelationMap reachable_parents_map;

    for(auto node: nodes) {
        reachable_children_map[node] = {};
        reachable_parents_map[node] = {};
        for(auto child: dfg.getNonSourceChildren(node)) {
            if(nodes.count(child)) {
                reachable_children_map[node].insert(child);
            }
        }
        for(auto parent: dfg.getNonSinkParents(node)) {
            if(nodes.count(parent)) {
                reachable_parents_map[node].insert(parent);
            }
        }
    }

    auto local_topo_order = GetTopologicalOrder(dfg, nodes);
    for(auto node: local_topo_order) {
        for(auto child: dfg.getNonSourceChildren(node)) {
            if(nodes.count(child)) {
                if(reachable_children_map.count(child)) {
                    for(auto child_reachable_node: reachable_children_map.at(child)) {
                        reachable_children_map[node].insert(child_reachable_node);
                    }
                }
            }
        }
    }
    // reachable_parents_map should be populated in reverse
    for(auto it = local_topo_order.rbegin(); it != local_topo_order.rend(); ++it) {
        auto node = *it;
        for(auto parent: dfg.getNonSinkParents(node)) {
            if(nodes.count(parent)) {
                if(reachable_parents_map.count(parent)) {
                    for(auto parent_reachable_node: reachable_parents_map.at(parent)) {
                        reachable_parents_map[node].insert(parent_reachable_node);
                    }
                }
            }
        }
    }
    return {reachable_parents_map, reachable_children_map};
}

void recursivelyAddChildrenToCriticalSet(
    const ExtendedDFG& dfg,
    const NodeSet& node_range,
    const NodeMap<int>& source_chain_map,
    const NodeMap<int>& target_chain_map,
    const Node* node,
    NodeSet& critical_set,
    NodeChainRegion& source_region,
    NodeChainRegion& target_region) {
    if(!node_range.count(node) || critical_set.count(node)) {
        return;
    }
    critical_set.insert(node);
    if(source_chain_map.count(node)) {
        source_region.update(source_chain_map.at(node));
    }
    if(target_chain_map.count(node)) {
        target_region.update(target_chain_map.at(node));
    }
    for(auto child: dfg.getNonSourceChildren(node)) {
        recursivelyAddChildrenToCriticalSet(dfg, node_range, source_chain_map, target_chain_map, child,
                                            critical_set, source_region, target_region);
    }
}

bool tryExpandCriticalSet(
    const ExtendedDFG& dfg,
    const NodeSet& node_range,
    const NodeRelationMap& reachability_map,
    const NodeMap<int>& source_chain_map,
    const NodeMap<int>& target_chain_map,
    const Node* node,
    NodeSet& critical_set,
    NodeChainRegion& source_region,
    NodeChainRegion& target_region
) {
    // recursively add nodes into critical set
    if(node_range.count(node) && !critical_set.count(node)) {
        for(auto critical_node: critical_set) {
            if(reachability_map.at(node).count(critical_node)) {
                // reachable to critical set
                recursivelyAddChildrenToCriticalSet(dfg, node_range, source_chain_map, target_chain_map, node,
                                                    critical_set, source_region, target_region);
                return true;
            }
        }
    }
    return false;
}

RegionOfInterest identifyCriticalSet(
    const Node* source_node,
    const Node* target_node,
    const SimulateTimeType idle_time,
    const ScheduledDFG& sched_dfg,
    const NodeMap<int>& comp_chain_map,
    const NodeMap<int>& comm_chain_map,
    int dp_group_size) {

    auto& dfg = sched_dfg.dfg;

    NodeSet candidate_set;
    // identify the set of nodes in the source stream that 
    // overlaps with the idle period
    NodeType source_type = dfg.getNodeType(source_node);
    const NodeMap<int>& source_chain_map = type_select(comp_chain_map, comm_chain_map, source_type);
    const NodeMap<int>& target_chain_map = type_select(comp_chain_map, comm_chain_map, to_other(source_type));
    CHECK(source_chain_map.count(source_node));
    CHECK(target_chain_map.count(target_node));
    int source_chain_idx = source_chain_map.at(source_node);
    int target_chain_idx = target_chain_map.at(target_node);
    auto& source_chain = type_select(sched_dfg.comp_chain, sched_dfg.comm_chain, source_type);
    auto& target_chain = type_select(sched_dfg.comp_chain, sched_dfg.comm_chain, to_other(source_type));


    NodeChainRegion source_region = {source_chain_idx, source_chain_idx};
    NodeChainRegion target_region = {target_chain_idx, target_chain_idx};

    CHECK(sched_dfg.node_durations.count(source_node));
    SimulateTimeType source_node_start_time = sched_dfg.node_durations.at(source_node).start;
    CHECK(sched_dfg.node_durations.count(target_node));
    SimulateTimeType target_node_end_time = sched_dfg.node_durations.at(target_node).end;

    int roi_boundary_factor = 2;
    if(const char* roi_boundary_factor_str = getenv("ROI_BOUNDARY_FACTOR")) {
        roi_boundary_factor = std::stoi(std::string(roi_boundary_factor_str));
    }

    SimulateTimeType source_start_time = source_node_start_time;
    // initialize candidate set on the source stream
    int source_start_idx = source_chain_idx;
    while(true) {
        auto source_stream_node = source_chain[source_start_idx];
        CHECK(sched_dfg.node_durations.count(source_stream_node));
        source_start_time = sched_dfg.node_durations.at(source_stream_node).start;
        if(source_node_start_time - source_start_time >= roi_boundary_factor * idle_time) {
            break;
        }
        if(source_stream_node != dfg.source() && source_stream_node != dfg.sink()) {
            candidate_set.insert(source_stream_node);
            source_region.update(source_start_idx);
        }
        source_start_idx --;
        if(source_start_idx < 0) {
            break;
        }
    }

    SimulateTimeType source_end_time = sched_dfg.node_durations.at(source_node).end;
    int source_end_idx = source_chain_idx;
    while(true) {
        auto source_stream_node = source_chain[source_end_idx];
        CHECK(sched_dfg.node_durations.count(source_stream_node));
        source_end_time = sched_dfg.node_durations.at(source_stream_node).end;
        if(source_end_time - target_node_end_time >= roi_boundary_factor * idle_time) {
            break;
        }
        if(source_stream_node != dfg.source() && source_stream_node != dfg.sink()) {
            candidate_set.insert(source_stream_node);
            source_region.update(source_end_idx);
        }
        source_end_idx ++;
        if(source_end_idx >= source_chain.size()) {
            break;
        }
    }

    // we must also extend target stream start boundary to match source chain, since we are now extending the candidate set
    // before the idle period and additional cross stream dependency may happen there
    SimulateTimeType target_start_time = sched_dfg.node_durations.at(target_node).start;
    int target_start_idx = target_chain_idx;
    while(true) {
        auto target_stream_node = target_chain[target_start_idx];
        CHECK(sched_dfg.node_durations.count(target_stream_node));
        target_start_time = sched_dfg.node_durations.at(target_stream_node).start;
        if(source_node_start_time - target_start_time >= roi_boundary_factor * idle_time) {
            break;
        }
        if(target_stream_node != dfg.source() && target_stream_node != dfg.sink()) {
            candidate_set.insert(target_stream_node);
            target_region.update(target_start_idx);
        }
        target_start_idx --;
        if(target_start_idx < 0) {
            break;
        }
    }

    SimulateTimeType target_end_time = target_node_end_time;
    int target_end_idx = target_chain_idx;
    while(true) {
        auto target_stream_node = target_chain[target_end_idx];
        CHECK(sched_dfg.node_durations.count(target_stream_node));
        target_end_time = sched_dfg.node_durations.at(target_stream_node).end;
        if(target_end_time - target_node_end_time >= roi_boundary_factor * idle_time) {
            break;
        }
        if(target_stream_node != dfg.source() && target_stream_node != dfg.sink()) {
            candidate_set.insert(target_stream_node);
            target_region.update(target_end_idx);
        }
        target_end_idx++;
        if(target_end_idx >= target_chain.size()) {
            break;
        }
    }

    // LOG(INFO) << " ";
    // LOG(INFO) << "-----------------------------------";
    // LOG(INFO) << "Candidate set:";
    // for(auto node: candidate_set) {
    //     LOG(INFO) << "\t -> " << dfg.getNodeNameOrDefault(node);
    // }
    // LOG(INFO) << "-----------------------------------";

    double ratio = 1.3;
    SimulateTimeType diff = 2000;
    if (const char* balance_constraint_params = getenv("BALANCE_CONSTRAINT_PARAMS")) {
      std::string params_str(balance_constraint_params);
      auto ratio_str = params_str.substr(0, params_str.find(";"));
      auto diff_str = params_str.substr(params_str.find(";")+1, std::string::npos);
      ratio = std::stod(ratio_str);
      diff = std::stod(diff_str);
    }
    BalanceConstraint balance_constraint(ratio, diff);
    // LOG(INFO) << "Balance constraint " << PrintBalanceConstraint(balance_constraint);

    auto start = std::chrono::system_clock::now();
    LOG(INFO) << "Started to build pipeline.";
    PipelineBuilder builder(sched_dfg, balance_constraint, dp_group_size);
    SimulateTimeType prune_threshold = 150;
    if (const char* prune_threshold_env_str = getenv("PIPELINE_PRUNE_THRESHOLD")) {
      std::string prune_threshold_str(prune_threshold_env_str);
      prune_threshold = std::stod(prune_threshold_str);
    }
    auto pipeline = builder.Build({source_node, target_node}, candidate_set, prune_threshold);
    auto end = std::chrono::system_clock::now();
    auto elapsed_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG(INFO) << "Building pipeline takes " << elapsed_time / 1000.0f << " ms.";

    auto critical_set = pipeline.GetNodes();

    if(critical_set.size() < 2) {
        // no valid pipeline can be found
        return {};
    }

    // TODO: there is a bug if we directly get solution from builder. why??
    // auto solution = builder.GetSolution();

    Exprs critical_exprs = {};
    auto topo_order = GetTopologicalOrder(sched_dfg.dfg, critical_set);
    for (auto node : topo_order) {
      auto expr = sched_dfg.dfg.getExprFromNode(node);
      critical_exprs.push_back(expr);
    }
    CPSolution solution = SolvePartitionAxes(sched_dfg, critical_exprs, dp_group_size);
    try {
        CheckAllPartitionable(solution);
    } catch (...) {
        LOG(WARNING) << "Failed to partition ROI with critical set: ";
        for(auto expr: critical_exprs) {
            LOG(WARNING) << "\t -> " << expr;
        }
        LOG(FATAL) << "Internal error in building pipeline.";
    }
    return {critical_set, source_node, source_region, target_region, solution};
}

NodeSet mergeNodeSets(const std::vector<const NodeSet*>& node_sets) {
    NodeSet result;
    for(auto node_set: node_sets) {
        for(auto node: (*node_set)) {
            result.insert(node);
        }
    }
    return result;
}

std::vector<RegionOfInterest> findLatencyCriticalROIs(const ScheduledDFG& sched_dfg, int dp_group_size, SimulateTimeType min_idle_time) {
    CHECK(sched_dfg.critical_path_valid) << "findLatencyCriticalROIs expects an ScheduledDFG with valid critical path.";
    auto& dfg = sched_dfg.dfg;
    auto& comp_chain = sched_dfg.comp_chain;
    auto& comm_chain = sched_dfg.comm_chain;
    auto& node_durations = sched_dfg.node_durations;
    // construct node to chain index map. this is used to lookup previous node in the same stream
    NodeMap<int> comp_chain_map;
    for(int i=0; i<comp_chain.size(); i++) {
        comp_chain_map[comp_chain[i]] = i;
    }
    NodeMap<int> comm_chain_map;
    for(int i=0; i<comm_chain.size(); i++) {
        comm_chain_map[comm_chain[i]] = i;
    }
    
    int n_valid_cross_edges = 0;
    std::vector<RegionOfInterest> regions_of_interest;
    std::vector<IdentifyCSArgs> ics_args;
    std::vector<SimulateTimeType> sample_weights;
    // find cross stream edges
    for(int cpid=1; cpid < sched_dfg.critical_path.size(); cpid++) {
        auto source_node = sched_dfg.critical_path[cpid-1];
        auto target_node = sched_dfg.critical_path[cpid];
        if(source_node == sched_dfg.dfg.source() || target_node == sched_dfg.dfg.sink()) {
            continue;
        }
        auto source_type = dfg.getNodeType(source_node);
        auto target_type = dfg.getNodeType(target_node);
        if(source_type != target_type) {
            // this is a cross stream edge
            const Node* prev_node = nullptr;
            int prev_node_idx_in_stream = -1;
            if(target_type == NodeType::kCompNode) {
                CHECK(comp_chain_map.count(target_node)) << "Cannot find " <<sched_dfg.dfg.getNodeNameOrDefault(target_node) << " in comp chain map.";
                prev_node_idx_in_stream = comp_chain_map.at(target_node) - 1;
                if (prev_node_idx_in_stream >= 0) {
                    prev_node = comp_chain[prev_node_idx_in_stream];
                }
            } else {
                CHECK(comm_chain_map.count(target_node)) << "Cannot find " <<sched_dfg.dfg.getNodeNameOrDefault(target_node) << " in comm chain map.";
                prev_node_idx_in_stream = comm_chain_map.at(target_node) - 1;
                if (prev_node_idx_in_stream >= 0) {
                    prev_node = comm_chain[prev_node_idx_in_stream];
                }
            }
            // find the resource idle period length
            SimulateTimeType idle_length;
            if(prev_node_idx_in_stream == -1) {
                idle_length = INFINITY;
            } else {
                CHECK(prev_node && node_durations.count(prev_node));
                SimulateTimeType prev_end_time = node_durations.at(prev_node).end;
                CHECK(node_durations.count(target_node));
                SimulateTimeType curr_start_time = node_durations.at(target_node).start;
                idle_length = curr_start_time - prev_end_time;
            }
            if (idle_length > min_idle_time) {
                n_valid_cross_edges ++;
                ics_args.push_back({source_node, target_node, idle_length});
                sample_weights.push_back(idle_length * idle_length);
            }
        }
    }
    int sample_threshold = 8;
    std::vector<IdentifyCSArgs> sampled_candidates;
    if(n_valid_cross_edges > sample_threshold) {
        // down sample to sample_threshold
        n_valid_cross_edges = sample_threshold;
        std::tie(ics_args, std::ignore) = WeightedSample(ics_args, sample_weights, sample_threshold);
    }
    regions_of_interest.resize(n_valid_cross_edges);
    // run identifyCriticalSet in multithread
    auto thread_f = [&](IdentifyCSArgs args, int idx) {
        RegionOfInterest roi = identifyCriticalSet(
                                std::get<0>(args),
                                std::get<1>(args),
                                std::get<2>(args),
                                sched_dfg,
                                comp_chain_map,
                                comm_chain_map,
                                dp_group_size);
        regions_of_interest[idx] = roi;
    };
    int n_threads = std::thread::hardware_concurrency() - 1;
    if (n_threads <= 1) {
        // just run using current thread
        for(int i=0; i<n_valid_cross_edges; i++) {
            LOG(INFO) << "Identifying Critical Set for cross edge " << i << "/" << n_valid_cross_edges;
            thread_f(ics_args[i], i);
        }
    } else {
        std::vector<std::thread> thread_workers;
        int prev_idx = 0;
        bool finished = false;
        while(!finished) {
            for(int i=0; i<n_threads; i++) {
                int workload_idx = prev_idx + i;
                if(workload_idx < n_valid_cross_edges) {
                    thread_workers.emplace_back(std::move(std::thread(thread_f, ics_args[workload_idx], workload_idx)));
                } else {
                    finished = true;
                    break; // break out of for loop
                }
            }
            for (auto& th: thread_workers) {
                th.join();
            }
            prev_idx += thread_workers.size();
            thread_workers.clear();
        }
    }
    std::vector<RegionOfInterest> valid_regions_of_interest;
    for(auto roi: regions_of_interest) {
        if(roi.critical_set.size() > 1) {
            valid_regions_of_interest.emplace_back(roi);
        }
    }
    return valid_regions_of_interest;
}

Indices GetGroupIndexByCrossStreamCount(const ScheduledDFG& sched_dfg, const Nodes& nodes) {
    CHECK(nodes.size());
    Indices group_index = {0};
    for (int i = 1; i < nodes.size(); ++i) {
        auto curr_type = sched_dfg.dfg.getNodeType(nodes[i]);
        auto prev_type = sched_dfg.dfg.getNodeType(nodes[i - 1]);
        if (curr_type != prev_type) {
            group_index.push_back(group_index.back() + 1);
        } else {
            group_index.push_back(group_index.back());
        }
        prev_type = curr_type;
    }
    return group_index;
}

std::vector<SimulateTimeType> GetGroupExecTime(const ScheduledDFG& sched_dfg, const Indices& group_index, const Nodes& nodes) {
    CHECK(nodes.size());
    std::vector<SimulateTimeType> group_exec_time(group_index.back() + 1, 0);
    for (int i = 0; i < nodes.size(); ++i) {
        group_exec_time[group_index[i]] += sched_dfg.dfg.getNodeExecTime(nodes[i]);
    }
    // Set the last group's exec time to a large value.
    group_exec_time.back() = std::numeric_limits<SimulateTimeType>::max();
    return group_exec_time;
}

Indices GetGroupPrio(const ScheduledDFG& sched_dfg, const Indices& group_index, const Nodes& nodes) {
    auto group_exec_time = GetGroupExecTime(sched_dfg, group_index, nodes);
    std::vector<std::pair<Index, SimulateTimeType>> group_index_and_exec_time = {};
    for (int i = 0; i < group_exec_time.size(); ++i) {
        group_index_and_exec_time.push_back(std::make_pair(i, group_exec_time[i]));
    }
    std::sort(group_index_and_exec_time.begin(), group_index_and_exec_time.end(),
        [](const std::pair<Index, SimulateTimeType>& lhs, const std::pair<Index, SimulateTimeType>& rhs) {
            return lhs.second < rhs.second;
        }
    );
    Indices group_prio(group_exec_time.size(), 0);
    for (int i = 0; i < group_exec_time.size(); ++i) {
        group_prio[group_index_and_exec_time[i].first] = group_exec_time.size() - i;
    }
    return group_prio;
}

// a wrapper function for calculating priority in partitioned pipeline
void calcPipelinePriority(const ScheduledDFG& sched_dfg,
                                const NodeMap<ScheduleLocation>& node_loc_map,
                                NodeMap<double>& output) {
    IndexMap<Nodes> partition_index_nodes = {};
    Index max_partition_index = -1;
    for (auto it : node_loc_map) {
        if (it.second.type == ScheduleType::kPrelude) {
            // prelude node, assign a very high priority
            output[it.first] = 100000;
        } else if (it.second.type == ScheduleType::kEpilogue || it.second.type == ScheduleType::kArbitrary) {
            // epilogue or arbitrary node, assign a very low priority
            output[it.first] = -100000;
        } else {
            // pipeline node, group by part index
            Index partition_index = it.second.partition_index;
            partition_index_nodes[partition_index].push_back(it.first);
            max_partition_index = std::max(max_partition_index, partition_index);
        }
    }
    CHECK(partition_index_nodes.size());
    auto nodes_topo_order = GetTopologicalOrder(sched_dfg.dfg, partition_index_nodes[0]);
    auto group_index = GetGroupIndexByCrossStreamCount(sched_dfg, nodes_topo_order);
    auto group_prio = GetGroupPrio(sched_dfg, group_index, nodes_topo_order);
    int group_count = group_index.back() + 1;
    for (int i = 0; i <= max_partition_index; ++i) {
        auto partitioned_nodes = GetTopologicalOrder(sched_dfg.dfg, partition_index_nodes[i]);
        for (int j = 0; j < partitioned_nodes.size(); ++j) {
            output[partitioned_nodes[j]] = group_prio[group_index[j]] * max_partition_index - i;
        }
    }
    // we assign nodes with zero execution time a high priority
    for (auto it: output) {
        if (sched_dfg.dfg.getNodeExecTime(it.first) == 0) {
            output[it.first] = 100000;
        }
    }
}

void calcPipelinePriorityBasedOnPartitionIndex(const ScheduledDFG& sched_dfg,
                                const NodeMap<ScheduleLocation>& node_loc_map,
                                int n_partitions,
                                NodeMap<double>& output) {
    for (auto it : node_loc_map) {
        if (it.second.type == ScheduleType::kPrelude) {
            // prelude node, assign a very high priority
            output[it.first] = 100000;
        } else if (it.second.type == ScheduleType::kEpilogue || it.second.type == ScheduleType::kArbitrary) {
            // epilogue or arbitrary node, assign a very low priority
            output[it.first] = -100000;
        } else {
            // pipeline node, priority is the inverse partition index
            Index partition_index = it.second.partition_index;
            output[it.first] = (n_partitions - partition_index);
        }
    }
}

NodeSet GetAffectedNodesFromROI(const ScheduledDFG& sched_dfg, const RegionOfInterest& roi) {
    // get the set of all nodes in roi region
    NodeSet roi_nodes;
    auto add_node_in_chain_to_set = [&](const NodeChainRegion region, const Nodes& chain,
                                        NodeSet& node_set) {
        for(int idx=region.start_index; idx <= region.end_index; idx++) {
            CHECK_GT(chain.size(), idx);
            auto node = chain[idx];
            node_set.insert(node);
        }
    };
    if(sched_dfg.dfg.getNodeType(roi.source_node) == NodeType::kCompNode) {
        add_node_in_chain_to_set(roi.source_chain, sched_dfg.comp_chain, roi_nodes);
        add_node_in_chain_to_set(roi.target_chain, sched_dfg.comm_chain, roi_nodes);
    } else {
        add_node_in_chain_to_set(roi.source_chain, sched_dfg.comm_chain, roi_nodes);
        add_node_in_chain_to_set(roi.target_chain, sched_dfg.comp_chain, roi_nodes);
    }
    return roi_nodes;
}

std::pair<Nodes,Nodes> GetScheduleLocationForIrrelevantNodes(
    const ScheduledDFG& sched_dfg, const RegionOfInterest& roi) {

    Nodes prelude_nodes;
    Nodes epilogue_nodes;

    NodeSet affected_nodes = GetAffectedNodesFromROI(sched_dfg, roi);

    NodeRelationMap reachable_parents_map, reachable_children_map;
    std::tie(reachable_parents_map, reachable_children_map) = buildLocalReachabilityMap(sched_dfg.dfg, affected_nodes);

    auto topo_order_affected_nodes = GetTopologicalOrder(sched_dfg.dfg, affected_nodes);

    for(auto node: topo_order_affected_nodes) {
        if(roi.critical_set.count(node)) {
            continue;
        }
        // this node is not partitioned
        bool is_prelude_node = false;
        bool is_epilogue_nodes = false;
        for(auto node_to_partition: roi.critical_set) {
            // DEBUG: we check all nodes anyway to check for correctness
            if(reachable_parents_map.at(node).count(node_to_partition)) {
                // a node to partition is a reachable parent of the current node
                // the current node must be put inside prelude region
                is_prelude_node = true;
            }
            if(reachable_children_map.at(node).count(node_to_partition)) {
                // a node to partition is a reachable children of the current node
                // the current node must be put inside epilogue region
                is_epilogue_nodes = true;
            }
        }
        CHECK(!(is_prelude_node && is_epilogue_nodes))
            << "Cannot find schedule location for irrelevant node " << sched_dfg.dfg.getNodeNameOrDefault(node);
        if(is_prelude_node) {
            prelude_nodes.push_back(node);
        } else {
            // if a node has no dependency with pipelined nodes, put it in epilogue
            epilogue_nodes.push_back(node);
        }
    }
    return std::make_pair(prelude_nodes, epilogue_nodes);
}

bool IsPipelinable(const ScheduledDFG& sched_dfg, const NodeSet& candidate_nodes, const NodeSet& critical_nodes) {
    auto& dfg = sched_dfg.dfg;
    auto local_topo_order = GetTopologicalOrder(dfg, candidate_nodes);
    NodeMap<bool> p2np_sync;
    for (auto node : critical_nodes) {
        for (auto parent: dfg.getNonSinkParents(node)) {
            if (!critical_nodes.count(parent)) {
                p2np_sync[parent] = true;
            }
        }
    }
    for (auto node: local_topo_order) {
        if (p2np_sync[node]) {
            for (auto parent: dfg.getNonSinkParents(node)) {
                p2np_sync[parent] = true;
            }
        }
    }
    for (auto node : critical_nodes) {
        if (p2np_sync[node]) {
            return false;
        }
    }
    return true;
}

bool IsPipelinable(const ScheduledDFG& sched_dfg, const NodeSet& candidate_nodes, const NodeSet& critical_nodes, const Node* node) {
    auto nodes_ = critical_nodes;
    nodes_.insert(node);
    return IsPipelinable(sched_dfg, candidate_nodes, nodes_);
}

// balanced constraint

BalanceConstraint::BalanceConstraint(double ratio, SimulateTimeType diff) :
    ratio(ratio < 1 ? ratio : 1 / ratio),
    diff(diff > 0 ? diff : -diff) {}

std::string PrintBalanceConstraint(const BalanceConstraint& constraint) {
    return std::string("ratio: ") + std::to_string(constraint.ratio) + ", diff: " + std::to_string(constraint.diff);
}

bool IsBalanced_(const BalanceConstraint& constraint, SimulateTimeType comm_time, SimulateTimeType comp_time) {
    // LOG(INFO) << "In IsBalanced_: balance constraint: " << PrintBalanceConstraint(constraint);
    double ratio = (double) comp_time / comm_time;
    ratio = ratio > 1 ? 1 / ratio : ratio;
    SimulateTimeType difference = comp_time - comm_time;
    difference = difference > 0 ? difference : -difference;
    // LOG(INFO) << "[IsBalanced]: new ratio: " << ratio << ", constraint: " << constraint.ratio;
    // LOG(INFO) << "[IsBalanced]: new difference: " << difference << ", constraint:  " << constraint.diff;
    // View the comm and comp time balanced when more than 1 constraint is satisfied.
    // This is beneficial when trying to add a node of non-desired type with desired type of parents and children.
    return ratio < constraint.ratio || difference < constraint.diff;
}

// node priority queue

NodePriorityQueue::NodePriorityQueue(const ScheduledDFG& sched_dfg) :
    sched_dfg_(sched_dfg), queue_(CreateFCompare_()) {}

NodePriorityQueue::FCompare NodePriorityQueue::CreateFCompare_() {
    return [&](const Node* lhs, const Node* rhs) -> bool {
        return sched_dfg_.dfg.getNodeExecTime(lhs) < sched_dfg_.dfg.getNodeExecTime(rhs);
    };
}

void NodePriorityQueue::Push(const Node* node) {
    queue_.push(node);
}

const Node* NodePriorityQueue::Top() const {
    return queue_.top();
}

const Node* NodePriorityQueue::Pop() {
    auto node = queue_.top();
    queue_.pop();
    return node;
}

int NodePriorityQueue::Size() const {
    return queue_.size();
}

bool NodePriorityQueue::Empty() const {
    return Size() == 0;
}

void NodePriorityQueue::Clear() {
    queue_ = std::priority_queue<const Node*, std::deque<const Node*>, FCompare>(CreateFCompare_());
}

// pipeline

Pipeline::Pipeline(const ScheduledDFG& sched_dfg, const BalanceConstraint& constraint) :
    sched_dfg_(sched_dfg), constraint_(constraint){
    // LOG(INFO) << "In pipeline: balance constraint: " << PrintBalanceConstraint(constraint_);
}

void Pipeline::AddNode(const Node* node) {
    CHECK(!all_nodes_.count(node));
    auto exec_time = sched_dfg_.dfg.getNodeExecTime(node);
    auto node_type = sched_dfg_.dfg.getNodeType(node);
    all_nodes_.insert(node);
    type_select(comp_nodes_, comm_nodes_, node_type).insert(node);
    type_select(comp_time_, comm_time_, node_type) += exec_time;
}

bool Pipeline::IsBalanced() const {
    DLOG(INFO) << "Comp_time: " << comp_time_ << ", Comm_time: " << comm_time_;
    return IsBalanced_(constraint_, comp_time_, comm_time_);
}

bool Pipeline::IsBalancedIfAdd(const Node* node) const {
    auto exec_time = sched_dfg_.dfg.getNodeExecTime(node);
    auto node_type = sched_dfg_.dfg.getNodeType(node);
    auto comm_time = comm_time_;
    auto comp_time = comp_time_;
    type_select(comp_time, comm_time, node_type) += exec_time;
    return IsBalanced_(constraint_, comp_time, comm_time);
}

NodeType Pipeline::RequiredNodeType() const {
    return comp_time_ > comm_time_ ? NodeType::kCommNode : NodeType::kCompNode;
}

const NodeSet& Pipeline::GetNodes() const {
    return all_nodes_;
}

int Pipeline::GetCommNodesSize() const {
    return comm_nodes_.size();
}

int Pipeline::GetCompNodesSize() const {
    return comp_nodes_.size();
}

SimulateTimeType Pipeline::GetCommTime() const {
    return comm_time_;
}

SimulateTimeType Pipeline::GetCompTime() const {
    return comp_time_;
}

const ScheduledDFG& Pipeline::GetScheduledDFG() const {
    return sched_dfg_;
}

const BalanceConstraint& Pipeline::GetConstraint() const {
    return constraint_;
}

void Pipeline::Prune(SimulateTimeType threshold) {
    auto& dfg = sched_dfg_.dfg;
    NodeMap<int> children_counter;
    NodeMap<int> parent_counter;

    int orig_size = all_nodes_.size();
    int pruned_nodes = 0;

    NodeSet checked_nodes;
    std::queue<const Node*> peripheral_nodes;
    for(auto node: all_nodes_) {
        parent_counter[node] = 0;
        children_counter[node] = 0;
        for(auto parent: dfg.getNonSinkParents(node)) {
            if(all_nodes_.count(parent)) {
                parent_counter[node] ++;
            }
        }
        if (parent_counter[node] == 0 && !checked_nodes.count(node)) {
            peripheral_nodes.push(node);
            checked_nodes.insert(node);
        }
        for(auto child: dfg.getNonSourceChildren(node)) {
            if(all_nodes_.count(child)) {
                children_counter[node] ++;
            }
        }
        if(children_counter[node] == 0 && !checked_nodes.count(node)) {
            peripheral_nodes.push(node);
            checked_nodes.insert(node);
        }
    }

    while(!peripheral_nodes.empty()) {
        auto node = peripheral_nodes.front();
        peripheral_nodes.pop();
        auto node_exec_time = dfg.getNodeExecTime(node);
        if(node_exec_time < threshold) {
            // prune
            // LOG(INFO) << "Pruning node " << dfg.getNodeNameOrDefault(node);
            pruned_nodes ++;
            all_nodes_.erase(node);
            if(dfg.getNodeType(node) == NodeType::kCompNode) {
                CHECK(comp_nodes_.count(node));
                comp_nodes_.erase(node);
                comp_time_ -= node_exec_time;
            } else {
                CHECK(comm_nodes_.count(node));
                comm_nodes_.erase(node);
                comm_time_ -= node_exec_time;
            }
            for(auto parent: dfg.getNonSinkParents(node)) {
                if(all_nodes_.count(parent)) {
                    children_counter[parent] --;
                    if(children_counter[parent] == 0 && !checked_nodes.count(parent)) {
                        peripheral_nodes.push(parent);
                        checked_nodes.insert(parent);
                    }
                }
            }
            for(auto child: dfg.getNonSourceChildren(node)) {
                if(all_nodes_.count(child)) {
                    parent_counter[child] --;
                    if(parent_counter[child] == 0 && !checked_nodes.count(child)) {
                        peripheral_nodes.push(child);
                        checked_nodes.insert(child);
                    }
                }
            }
        }
    }
    LOG(INFO) << "Pruned " << pruned_nodes << " / " << orig_size << " nodes.";
}

void Pipeline::Clear() {
    all_nodes_.clear();
    comp_nodes_.clear();
    comm_nodes_.clear();
    comp_time_ = 0;
    comm_time_ = 0;
}

bool Pipeline::IsMergeable(const Pipeline& lhs, const Pipeline& rhs) {
    NodeSet depended_nodes = {};
    const auto& sched_dfg = lhs.GetScheduledDFG();
    auto& lhs_nodes = lhs.GetNodes();
    auto& rhs_nodes = rhs.GetNodes();
    for (auto node : lhs_nodes) {
        depended_nodes.insert(node);
        auto parents = sched_dfg.dfg.getNonSinkParents(node);
        depended_nodes.insert(parents.begin(), parents.end());
        auto children = sched_dfg.dfg.getNonSourceChildren(node);
        depended_nodes.insert(children.begin(), children.end());
    }
    for (auto node : rhs_nodes) {
        if (depended_nodes.count(node)) {
            return true;
        }
    }
    return false;
}

Pipeline Pipeline::Merge(const Pipeline& lhs, const Pipeline& rhs) {
    NodeSet all_nodes = {};
    auto& lhs_nodes = lhs.GetNodes();
    auto& rhs_nodes = rhs.GetNodes();
    all_nodes.insert(lhs_nodes.begin(), lhs_nodes.end());
    all_nodes.insert(rhs_nodes.begin(), rhs_nodes.end());
    Pipeline pipeline(lhs.GetScheduledDFG(), lhs.GetConstraint());
    for (auto node : all_nodes) {
        pipeline.AddNode(node);
    }
    return pipeline;
}

std::string PrintPipeline(const ExtendedDFG& dfg, const Pipeline& pipeline) {
    std::ostringstream os;
    os << "Pipeline stats:" << std::endl;
    os << "[comm size: " << pipeline.GetCommNodesSize()
       << ", comp size: " << pipeline.GetCompNodesSize()
       << ", comm_time: " << pipeline.GetCommTime()
       << ", comp_time: " << pipeline.GetCompTime() << "]" << std::endl;
    os << "-------------------" << std::endl;
    os << "Pipeline Nodes" << std::endl;
    for(auto node: pipeline.GetNodes()) {
        os << "\t -> " << dfg.getNodeNameOrDefault(node) << std::endl;
    }
    return os.str();
}

// pipeline builder

PipelineBuilder::PipelineBuilder(const ScheduledDFG& sched_dfg, BalanceConstraint balance_constraint, int dp_group_size) :
    sched_dfg_(sched_dfg), balance_constraint_(balance_constraint), dp_group_size_(dp_group_size),
    comm_queue_(sched_dfg), comp_queue_(sched_dfg), pipeline_(sched_dfg, balance_constraint) {}

CPSolution PipelineBuilder::GetSolution() const {
    return solution_;
}

Pipeline PipelineBuilder::Build(const CrossStreamDependency& dep, const NodeSet& candidates, SimulateTimeType prune_threshold) {
    Setup_(dep, candidates);
    while (!comm_queue_.Empty() || !comp_queue_.Empty()) {
        auto type = pipeline_.RequiredNodeType();
        auto& queue = type_select(comp_queue_, comm_queue_, type);
        auto& other_queue = type_select(comp_queue_, comm_queue_, to_other(type));
        if (auto node = FetchNode_(queue)) {
            // LOG(INFO) << "Add node " << sched_dfg_.dfg.getNodeNameOrDefault(node) << " in required stream.";
            pipeline_.AddNode(node);
            solution_ = temp_solution_;
            FetchCandidates_(node);
        } else {
            // LOG(INFO) << "Fetch node not in required stream.";
            auto node_ = FetchNode_(other_queue);
            if(!node_) {
                // LOG(INFO) << "Both queue is empty, completes expanding pipeline.";
                pipeline_.Prune(prune_threshold);
                return pipeline_;
            }
            while (!pipeline_.IsBalancedIfAdd(node_)) {
                // LOG(INFO) << "Adding node " << sched_dfg_.dfg.getNodeNameOrDefault(node_) << " violates balance constraint."; 
                node_ = FetchNode_(other_queue);
                if(!node_) {
                    // LOG(INFO) << "Both queue is empty, completes expanding pipeline.";
                    pipeline_.Prune(prune_threshold);
                    return pipeline_;
                }
            }
            // LOG(INFO) << "Add node " << sched_dfg_.dfg.getNodeNameOrDefault(node) << " not in required stream.";
            pipeline_.AddNode(node_);
            solution_ = temp_solution_;
            FetchCandidates_(node_);
        }
    }
    // LOG(INFO) << "Build pipeline completed.";
    // LOG(INFO) << "Final pipeline: \n" << PrintPipeline(sched_dfg_.dfg, pipeline_);
    pipeline_.Prune(prune_threshold);
    return pipeline_;
}

void PipelineBuilder::Setup_(const CrossStreamDependency& dep, const NodeSet& candidates) {
    candidates_ = candidates;
    comm_queue_.Clear();
    comp_queue_.Clear();
    pipeline_.Clear();
    // we only add source node since source + target may not be partitionable, but
    // in this case we still hope to partition some nodes since nodes around that region
    // are likely also in the critical path
    // target node will be fetched in FetchCandidates_ since source and target node are 
    // connected by a data edge
    pipeline_.AddNode(dep.source);
    FetchCandidates_(dep.source);
}

void PipelineBuilder::FetchCandidates_(const Node* node) {
    auto& nodes = pipeline_.GetNodes();
    for (auto parent : sched_dfg_.dfg.getNonSinkParents(node)) {
        auto parent_name = sched_dfg_.dfg.getNodeNameOrDefault(parent);
        // LOG(INFO) << "Visiting parent " << sched_dfg_.dfg.getNodeNameOrDefault(parent) << " of node " << sched_dfg_.dfg.getNodeNameOrDefault(node);
        if (parent_name.find("part_") == std::string::npos && candidates_.count(parent) && !nodes.count(parent)) {
            auto& queue = type_select(comp_queue_, comm_queue_, sched_dfg_.dfg.getNodeType(parent));
            queue.Push(parent);
            // LOG(INFO) << "Pushing node " << sched_dfg_.dfg.getNodeNameOrDefault(parent) << " into queue.";
        }
    }
    for (auto child : sched_dfg_.dfg.getNonSourceChildren(node)) {
        auto child_name = sched_dfg_.dfg.getNodeNameOrDefault(child);
        // LOG(INFO) << "Visiting children " << sched_dfg_.dfg.getNodeNameOrDefault(child) << " of node " << sched_dfg_.dfg.getNodeNameOrDefault(node);
        if (child_name.find("part_") == std::string::npos && candidates_.count(child) && !nodes.count(child)) {
            auto& queue = type_select(comp_queue_, comm_queue_, sched_dfg_.dfg.getNodeType(child));
            queue.Push(child);
            // LOG(INFO) << "Pushing node " << sched_dfg_.dfg.getNodeNameOrDefault(child) << " into queue.";
        }
    }
}

static bool debug_print_ = false;
static const std::string node_name_to_print_ = "dontprint";
// static const std::string node_name_to_print_ = "moe_encode";
// static const std::string node_name_to_print_ = "batch_matmul";
// static const std::string node_name_to_print_ = "fn (%p0: Tensor[(32, 128, 768), float32]";

const Node* PipelineBuilder::FetchNode_(NodePriorityQueue& queue) {
    NodeSet nodes = pipeline_.GetNodes();
    while (!queue.Empty()) {
        auto node = queue.Top();
        queue.Pop();
        if(nodes.count(node)) {
            continue;
        }
        if(debug_print_ || sched_dfg_.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
            LOG(INFO) << " ";
            LOG(INFO) << "==============================================================";
            LOG(INFO) << "Testing " << sched_dfg_.dfg.getNodeNameOrDefault(node) << "...";
        }
        if (IsPipelinable(sched_dfg_, candidates_, nodes, node)) {
            bool is_all_partitionable = false;
            std::tie(is_all_partitionable, temp_solution_) = IsAllPartitionable(sched_dfg_, solution_, nodes, node, dp_group_size_);
            if (is_all_partitionable) {
                if(debug_print_ || sched_dfg_.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
                    LOG(INFO) << "Adding SUCCESS: is partitionable.";
                }
                CHECK(IsAllPartitionable(temp_solution_));
                return node;
            } else {
                if(debug_print_ || sched_dfg_.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
                    LOG(INFO) << "Adding FAILED: Not partitionable.";
                }
            }
        } else {
            if(debug_print_ || sched_dfg_.dfg.getNodeNameOrDefault(node).find(node_name_to_print_) != std::string::npos) {
                LOG(INFO) << "Adding FAILED: NotPipelinable.";
            }
        }
    }
    return nullptr;
}


}  // namespace roi_utils
}  // namespace pass
}  // namespace raf