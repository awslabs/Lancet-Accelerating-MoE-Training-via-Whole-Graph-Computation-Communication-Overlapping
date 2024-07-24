/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file cost_model_utils.cc
 * \brief Cost models used in schedule simulator.
 */

#include "cost_model_utils.h"

namespace raf {
namespace pass {
namespace cost_model_utils {

TVM_REGISTER_NODE_TYPE(CommCostModelParamsNode);

CommCostModelParams CommCostModelParams::make(uint64_t overhead, uint64_t throughput) {
  ObjectPtr<CommCostModelParamsNode> n = make_object<CommCostModelParamsNode>();
  n->overhead = overhead;
  n->throughput = throughput;
  return CommCostModelParams(n);
}

CommCostModelParams::CommCostModelParams(uint64_t overhead, uint64_t throughput) {
  ObjectPtr<CommCostModelParamsNode> n = make_object<CommCostModelParamsNode>();
  n->overhead = overhead;
  n->throughput = throughput;
  data_ = std::move(n);
}

}  // namespace cost_model_utils
}  // namespace pass
}  // namespace raf