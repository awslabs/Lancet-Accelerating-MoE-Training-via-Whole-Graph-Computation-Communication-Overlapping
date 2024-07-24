/*!
 * Copyright (c) 2022 by Contributors
 * \file cost_model_utils.h
 * \brief Cost models used in schedule simulator.
 */
#pragma once
#include "./scheduler_utils.h"

namespace raf {
namespace pass {
namespace cost_model_utils {

using namespace raf::pass::scheduler_utils;

class CommCostModelParamsNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  uint64_t overhead;
  uint64_t throughput;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("overhead", &overhead);
    v->Visit("throughput", &throughput);
  }

  static constexpr const char* _type_key = "raf.distributed.CommCostModelParams";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(CommCostModelParamsNode, Object);
};

class CommCostModelParams : public ObjectRef {
 public:
  CommCostModelParams(uint64_t overhead, uint64_t throughput);
  static CommCostModelParams make(uint64_t overhead, uint64_t throughput);
  TVM_DEFINE_OBJECT_REF_METHODS(CommCostModelParams, ObjectRef, CommCostModelParamsNode);
};

using CommCostModelParamsMap = std::unordered_map<CommType, CommCostModelParams>;

}  // namespace cost_model_utils
}  // namespace pass
}  // namespace raf