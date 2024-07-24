/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file test_partition_impl.cc
 * \brief Test partition impl.
 */
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
#include "roi_utils.h"
#include "lancet_optimization.h"
#include "./schedule_test_utils.h"

namespace raf {
namespace pass {
namespace test_partition_impl {

using namespace raf::analysis;
using namespace raf::analysis::dependency_graph;
using namespace scheduler_common;
using namespace partition_common;
using namespace solve_partition_axes;
using namespace partition_exprs;
using namespace extended_dfg;
using namespace roi_utils;
using namespace lancet_optimization;
using namespace schedule_test_utils;

Expr PartionANF(Expr expr) {
    auto prepare_result = PrepareDFG(expr);
    ScheduledDFG sched_dfg = std::get<0>(prepare_result);
    ExtendedOpProfiler op_profiler = std::get<1>(prepare_result);
    int n_experts = -1;
    // get n_experts from all to all type
    for (auto node: sched_dfg.dfg.nodes()) {
      auto expr = sched_dfg.dfg.getExprFromNode(node);
      if (expr.defined() && expr.as<CallNode>()) {
        auto call = Downcast<Call>(expr);
        if(IsAllToAll(call->op)) {
          // get shape
          auto shape = call->checked_type_.as<TensorTypeNode>()->shape;
          int inferred_n_experts = -1;
          if (shape.size() == 3) {
            // [E, C, M]
            inferred_n_experts = shape[0].as<IntImmNode>()->value;
          } else {
            CHECK(shape.size() == 4);
            // [G, LE, C, M]
            inferred_n_experts = shape[1].as<IntImmNode>()->value * shape[0].as<IntImmNode>()->value;
          }
          if (n_experts == -1) {
            n_experts = inferred_n_experts;
          } else {
            CHECK_EQ(n_experts, inferred_n_experts);
          }
        }
      }
    }

    // test only partition
    auto rois = findLatencyCriticalROIs(sched_dfg, 2);
    CHECK_GT(rois.size(), 0) << "Cannot find ROI in the input IR.";

    // try to partition the first roi
    auto roi = rois[0];
    LOG(INFO) << "=============== ROI to partition ==============";
    LOG(INFO) << PrintROI(sched_dfg, roi);
    LOG(INFO) << "===============================================";
    auto result = GetPartitionedExecTime(sched_dfg, 1, n_experts, 2, op_profiler, roi);
    SimulateTimeType partitioned_time = std::get<0>(result);
    auto new_sched_dfg = std::get<1>(result);
    regenerateTotalOrder(new_sched_dfg);
    regenerateExprBasedOnTotalOrder(new_sched_dfg);

    LOG(INFO) << "Partitioned time: " << partitioned_time;
    LOG(INFO) << "Partition Result:" << std::endl;
    LOG(INFO) << "DFG:" << std::endl;
    LOG(INFO) << new_sched_dfg.dfg << std::endl;

    return new_sched_dfg.scheduled_expr;
}

} // namespace test_partition_impl 

Pass TestPartitionImpl() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(
          tvm::relay::TransformF(test_partition_impl::PartionANF, f));
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 0, "TestPartitionImpl", {});
  PassInfo pass_info(0, "TestPartitionImpl", {});
  return RAFSequential({InferType(), ToGraphNormalForm(), func_pass}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.TestPartitionImpl").set_body_typed(TestPartitionImpl);

} // namespace pass
} // namespace raf