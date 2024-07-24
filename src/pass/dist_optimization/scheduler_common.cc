/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file scheduler_common.cc
 * \brief Define data structures used in scheduling and optimization.
 */

#include "scheduler_common.h"

namespace raf {
namespace pass {
namespace scheduler_common {

CommComponents MergeComponents(CommComponents& lhs, CommComponents& rhs) {
    CHECK(lhs.size() == rhs.size());
    CommComponents result = {};
    for (auto it : lhs) {
        result[it.first] = lhs[it.first] + rhs[it.first];
    }
    return result;
}

std::ostream& operator<< (std::ostream& s, CommType comm_type) {
    switch(comm_type) {
        case CommType::kAllReduce:
            s << "raf.op._allreduce";
            break;
        case CommType::kReduceScatter:
            s << "raf.op._reduce_scatter";
            break;
        case CommType::kAllGather:
            s << "raf.op._allgather";
            break;
        case CommType::kAllToAll:
            s << "raf.op._all_to_all";
            break;
    }
    return s;
}

CommType IdentifyCommType(const CommComponents& components) {
    for (auto it : components) {
        if (it.second > 0) {
            return it.first;
        }
    }
    LOG(FATAL) << "Can't identify CommType in components.";
    return CommType::kAllReduce;
}

CommType OpToCommType(const Op& op) {
    static const std::unordered_map<Op, CommType, ObjectPtrHash, ObjectPtrEqual> op2commtype = {
        {Op::Get("raf.op._allreduce"), CommType::kAllReduce},
        {Op::Get("raf.op._reduce_scatter"), CommType::kReduceScatter},
        {Op::Get("raf.op._allgather"), CommType::kAllGather},
        {Op::Get("raf.op._all_to_all"), CommType::kAllToAll}
    };
    auto op_ = op::IsDialectOp(op) ? op::GetBaseOp(op) : op;
    CHECK(op2commtype.count(op_)) << "Encountered unsupported communication op: " << op_;
    return op2commtype.at(op_);
}

std::ostream& operator<<(std::ostream& os, const NodeType& nodetype) {
    switch (nodetype) {
        case NodeType::kCompNode:
            os << "Computation";
            break;
        case NodeType::kCommNode:
            os << "Communication";
            break;
        default:
            break;
    }
    return os;
}

NodeDuration::NodeDuration(Node* blocked_by, SimulateTimeType start,
                 SimulateTimeType end, SimulateTimeType ready_time)
    : blocked_by(blocked_by), start(start), end(end), ready_time(ready_time) {}

NodeDuration::NodeDuration(): blocked_by(nullptr), start(0), end(0), ready_time(0) {};

std::ostream& operator << (std::ostream& os, const NodeDuration& duration) {
    return os << "[" << duration.start << ", " << duration.end << "]";
}

std::string removeFusionPrefix(const std::string& name) {
    if (name.rfind(FUSION_PREFIX, 0) != std::string::npos) {
        return name.substr(std::string(FUSION_PREFIX).length());
    }
    return name;
}

std::string fuseNodeNames(std::string name_a, std::string name_b) {
    name_a = removeFusionPrefix(name_a);
    name_b = removeFusionPrefix(name_b);
    return FUSION_PREFIX + name_a + FUSION_DELIM + name_b;
}

}  // namespace scheduler_common
}  // namespace pass
}  // namespace raf