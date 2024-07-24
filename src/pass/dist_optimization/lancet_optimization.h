/*!
 * Copyright (c) 2021 by Contributors
 * \file lancet_optimization.h
 * \brief Implements joint optimization.
 */
#pragma once
#include <iostream>
#include <math.h>
#include <limits>
#include <unordered_map>
#include <relay/transforms/pass_utils.h>
#include "raf/ir.h"
#include "raf/op_utils.h"
#include "raf/analysis.h"
#include "../../analysis/dependency_graph.h"

#include "../let_list.h"
#include "../stream_schedule.h"
#include "./scheduler_utils.h"
#include "./cost_model_utils.h"
#include "./schedule_generator.h"
#include "./fuse_exprs.h"
#include "./roi_utils.h"

// #define SIMULATOR_EPS 1e-8

namespace raf {
namespace pass {
namespace lancet_optimization {

using namespace raf::analysis;
using namespace raf::analysis::dependency_graph;
using namespace raf::pass::scheduler_utils;
using namespace raf::pass::cost_model_utils;
using namespace raf::pass::schedule_generator;
using namespace raf::pass::fuse_exprs;
using namespace raf::pass::roi_utils;
using stream_schedule::StreamSchedulerBase;
using op::IsCollectiveOp;

using FusionCandidate = std::pair<const Node*, const Node*>;
// the ScheduledDFG in FusionStrategy is already updated
using FusionStrategy = std::tuple<SimulateTimeType, ScheduledDFG>;
// the ScheduledDFG in PartitionStrategy is already updated
using PartitionStrategy = std::tuple<SimulateTimeType, ScheduledDFG>;

// This function returns a copy of the ScheduledDFG where the corresponding nodes are fused
std::tuple<SimulateTimeType, ScheduledDFG> GetFusedExecTime(
    const ScheduledDFG& sched_dfg,
    const Node* comm_a, const Node* comm_b,
    ExtendedOpProfiler& op_profiler);

std::tuple<SimulateTimeType, ScheduledDFG> GetPartitionedExecTime(
    const ScheduledDFG& sched_dfg, int dp_group_size, int n_experts, int number_of_partitions, ExtendedOpProfiler& op_profiler, RegionOfInterest roi);

std::pair<SimulateTimeType, ScheduledDFG> OptimizeScheduledDFG(
    const ScheduledDFG& sched_dfg, TimelineOptAlgo timeline_opt_algo, ExtendedOpProfiler& op_profiler, int max_iterations, int dp_group_size, int n_experts, bool disable_fusion, bool disable_partition);

std::pair<SimulateTimeType, ScheduledDFG> RunOptimization(
    Function func, const ExtendedDFG& dfg, ScheduleHeuristics sched_heuristic,
    TimelineOptAlgo timeline_opt_algo, CommCostModel& comm_cost_model,
    int dp_group_size, int n_experts, int max_iterations,
    bool disable_fusion, bool disable_partition);

std::pair<bool, ScheduledDFG> RunPartitionOnEntireGraph(const ExtendedDFG& dfg, int dp_group_size, int n_experts);

}  // namespace lancet_optimization
}  // namespace pass
}  // namespace raf