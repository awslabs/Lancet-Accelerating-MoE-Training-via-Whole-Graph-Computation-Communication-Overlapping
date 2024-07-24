/*!
 * Copyright (c) 2022 by Contributors
 * \file roi_utils.h
 * \brief Utility functions to find Regions of Interest (ROI).
 */
#pragma once
#include "./solve_partition_axes.h"
#include "./partition_exprs.h"

namespace raf {
namespace pass {
namespace roi_utils {

using namespace raf::pass::scheduler_utils;
using namespace raf::pass::solve_partition_axes;
using namespace raf::pass::partition_exprs;

using NodePartitionMap = NodeMap<std::vector<const Node*>>;

inline NodeType to_other(const NodeType& type) {
    if (type == NodeType::kCompNode) {
        return NodeType::kCommNode;
    } else {
        return NodeType::kCompNode;
    }
}

template <typename T>
inline const T& type_select(const T& comp_obj, const T& comm_obj, const NodeType& type) {
    if (type == NodeType::kCompNode) {
        return comp_obj;
    } else {
        return comm_obj;
    }
}

template <typename T>
inline T& type_select(T& comp_obj, T& comm_obj, const NodeType& type) {
    if (type == NodeType::kCompNode) {
        return comp_obj;
    } else {
        return comm_obj;
    }
}

struct NodeChainRegion {
    int start_index;
    int end_index;

    // extends the region to include idx
    void update(int idx);
    // extends the region to cover another region
    void update(const NodeChainRegion& region);
};

struct RegionOfInterest {
    RegionOfInterest();
    RegionOfInterest(const NodeSet& critical_set, const Node* source_node,
                     int source_chain_start, int source_chain_end,
                     int target_chain_start, int target_chain_end, 
                     const CPSolution& cp_solution);
    RegionOfInterest(const NodeSet& critical_set, const Node* source_node,
                     NodeChainRegion source_chain, NodeChainRegion target_chain, 
                     const CPSolution& cp_solution);
    NodeSet critical_set;
    const Node* source_node;
    NodeChainRegion source_chain;
    NodeChainRegion target_chain;
    CPSolution cp_solution;
};

std::string PrintROI(const ScheduledDFG& sched_dfg, const RegionOfInterest& roi);

// this function builds a reachability map for nodes in the input nodeset
std::pair<NodeRelationMap, NodeRelationMap> buildLocalReachabilityMap(const ExtendedDFG& dfg, NodeSet nodes);

void recursivelyAddChildrenToCriticalSet(
    const ExtendedDFG& dfg,
    const NodeSet& node_range,
    const NodeMap<int>& source_chain_map,
    const NodeMap<int>& target_chain_map,
    const Node* node,
    NodeSet& critical_set,
    NodeChainRegion& source_region,
    NodeChainRegion& target_region);

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
);

RegionOfInterest identifyCriticalSet(
    const Node* source_node,
    const Node* target_node,
    const SimulateTimeType idle_time,
    const ScheduledDFG& sched_dfg,
    const NodeMap<int>& comp_chain_map,
    const NodeMap<int>& comm_chain_map,
    int dp_group_size);

// returns node sets C and I 
std::vector<RegionOfInterest> findLatencyCriticalROIs(
    const ScheduledDFG& sched_dfg, int dp_group_size, SimulateTimeType min_idle_time = 2000 /* 2 ms */);

// a wrapper function for calculating FIFO priority
void calcPipelinePriority(const ScheduledDFG& sched_dfg,
                                const NodeMap<ScheduleLocation>& node_loc_map,
                                NodeMap<double>& output);
void calcPipelinePriorityBasedOnPartitionIndex(const ScheduledDFG& sched_dfg,
                                const NodeMap<ScheduleLocation>& node_loc_map,
                                int num_partitions,
                                NodeMap<double>& output);

std::pair<Nodes,Nodes> GetScheduleLocationForIrrelevantNodes(
    const ScheduledDFG& sched_dfg, const RegionOfInterest& roi);

bool IsPipelinable(const ScheduledDFG& sched_dfg, const NodeSet& critical_nodes, const NodeSet& candidate_nodes);

// boundary constraint
// For deciding whether node can be a candidate of ROI.
using BoundaryConstraint = std::function<bool(const Node*)>;

struct CrossStreamDependency {
    const Node* source;
    const Node* target;
};

// balance constraint

struct BalanceConstraint {
    BalanceConstraint(double ratio, SimulateTimeType diff);
    
    bool IsBalanced(SimulateTimeType comm_time, SimulateTimeType comp_time) const;

    double ratio;
    SimulateTimeType diff;
};

std::string PrintBalanceConstraint(const BalanceConstraint& constraint);

bool IsBalanced_(const BalanceConstraint& constraint, SimulateTimeType comm_time, SimulateTimeType comp_time);

// priority queue of nodes.

class NodePriorityQueue {
public:
    using FCompare = std::function<bool(const Node*, const Node*)>;

    NodePriorityQueue(const ScheduledDFG& sched_dfg);

    void Push(const Node* node);
    const Node* Top() const;
    const Node* Pop();
    int Size() const;
    bool Empty() const;
    void Clear();

private:
    FCompare CreateFCompare_();

    const ScheduledDFG& sched_dfg_;
    std::priority_queue<const Node*, std::deque<const Node*>, FCompare> queue_;
};

// pipelinable nodes

class Pipeline {
public:
    Pipeline(const ScheduledDFG& sched_dfg, const BalanceConstraint& constraint);

    void AddNode(const Node* node);
    bool IsBalanced() const;
    bool IsBalancedIfAdd(const Node* node) const;
    NodeType RequiredNodeType() const;

    const NodeSet& GetNodes() const;
    int GetCommNodesSize() const;
    int GetCompNodesSize() const;
    SimulateTimeType GetCommTime() const;
    SimulateTimeType GetCompTime() const;

    const ScheduledDFG& GetScheduledDFG() const;
    const BalanceConstraint& GetConstraint() const;

    void Clear();
    void Prune(SimulateTimeType threshold);

    static bool IsMergeable(const Pipeline& lhs, const Pipeline& rhs);
    static Pipeline Merge(const Pipeline& lhs, const Pipeline& rhs);

private:
    NodeSet comp_nodes_;
    NodeSet comm_nodes_;
    SimulateTimeType comp_time_;
    SimulateTimeType comm_time_;

    NodeSet all_nodes_;

    const ScheduledDFG& sched_dfg_;
    BalanceConstraint constraint_;
};

std::string PrintPipeline(const ExtendedDFG& dfg, const Pipeline& pipeline);

// pipeline builder

class PipelineBuilder {
public:
    PipelineBuilder(const ScheduledDFG& sched_dfg, BalanceConstraint balance_constraint, int dp_group_size);

    // candidates should cover consecutive regions on comm and comp chain (similar to NodeChainRegion)
    Pipeline Build(const CrossStreamDependency& dep, const NodeSet& candidates, SimulateTimeType prune_threshold = 150 /* 150us */);

    CPSolution GetSolution() const;

private:
    void Setup_(const CrossStreamDependency& dep, const NodeSet& candidates);
    const Node* FetchNode_(NodePriorityQueue& queue);
    void FetchCandidates_(const Node* node);

    const ScheduledDFG& sched_dfg_;
    BalanceConstraint balance_constraint_;
    int dp_group_size_;

    NodeSet candidates_;

    NodePriorityQueue comm_queue_;
    NodePriorityQueue comp_queue_;

    Pipeline pipeline_;
    CPSolution solution_;
    CPSolution temp_solution_;
};

}  // namespace roi_utils
}  // namespace pass
}  // namespace raf