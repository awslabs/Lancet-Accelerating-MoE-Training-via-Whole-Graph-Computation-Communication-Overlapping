/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file test_fusion_impl.cc
 * \brief Test fusion impl.
 */
#include <relay/transforms/pass_utils.h>
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/analysis.h"
#include "raf/pass.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "../let_list.h"
#include "../stream_schedule.h"
#include "./scheduler_common.h"
#include "./extended_dfg.h"
#include "./roi_utils.h"
#include "./lancet_optimization.h"
#include "./schedule_test_utils.h"

namespace raf {
namespace pass {
namespace test_fusion_impl {

using namespace raf::analysis;
using namespace raf::analysis::dependency_graph;
using namespace scheduler_common;
using namespace extended_dfg;
using namespace roi_utils;
using namespace lancet_optimization;
using namespace schedule_test_utils;

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

Expr FuseANF(Expr expr) {

    auto prepare_result = PrepareDFG(expr);
    ScheduledDFG sched_dfg = std::get<0>(prepare_result);
    ExtendedOpProfiler op_profiler = std::get<1>(prepare_result);

    auto& dfg = sched_dfg.dfg;
    auto& critical_path = sched_dfg.critical_path;
    
    ScheduledDFG new_sched_dfg = sched_dfg;


    // try to fuse all comm node pairs on critical path.
    for (size_t cpid = 1; cpid < critical_path.size(); cpid++) {
        regenerateTotalOrderBasedOnExpr(new_sched_dfg);
        // create node to chain idx map
        NodeMap<int> op_idx_map;
        for(int idx=0; idx < new_sched_dfg.total_order.size(); idx++) {
            op_idx_map[new_sched_dfg.total_order[idx]] = idx;
        }

        auto cpid_target = cpid;
        auto cpid_source = cpid - 1;
        // for each "edge", we try merge the previous node into the latter one
        const Node* source = critical_path[cpid_source];
        const Node* target = critical_path[cpid_target];
        if(dfg.getNodeType(target) == NodeType::kCompNode || dfg.getNodeType(source) == NodeType::kCompNode) {
            // currently only consider fusing two communications
            continue;
        }
        // check if the two comms are of the same type
        auto source_comm_type = IdentifyCommType(dfg.getCommSize(source));
        auto target_comm_type = IdentifyCommType(dfg.getCommSize(target));
        if(source_comm_type != target_comm_type) {
            // ignore all2all for now
            continue;
        }
        // check if the two nodes can be combined
        if(!canFuse(new_sched_dfg, source, target, op_idx_map)) {
            continue;
        }

        LOG(INFO) << "Trying to fuse node " << dfg.getNodeNameOrDefault(source) << " and " << dfg.getNodeNameOrDefault(target) << cpid;

        auto costs_and_new_sched_dfg = GetFusedExecTime(new_sched_dfg, source, target, op_profiler);
        SimulateTimeType new_cost = std::get<0>(costs_and_new_sched_dfg);
        new_sched_dfg = std::get<1>(costs_and_new_sched_dfg);
        LOG(INFO) << "fused scheduled expr " << ir::AsText(new_sched_dfg.scheduled_expr);
    }

    return new_sched_dfg.scheduled_expr;
}

} // namespace test_fusion_impl 

Pass TestFusionImpl() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(
          tvm::relay::TransformF(test_fusion_impl::FuseANF, f));
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 0, "TestFuseImpl", {});
  PassInfo pass_info(0, "TestFuseImpl", {});
  return RAFSequential({InferType(), ToGraphNormalForm(), func_pass}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.TestFusionImpl").set_body_typed(TestFusionImpl);

} // namespace pass
} // namespace raf