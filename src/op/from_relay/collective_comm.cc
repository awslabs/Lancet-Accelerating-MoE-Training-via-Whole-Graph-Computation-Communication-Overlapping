/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/from_relay/collective_comm.cc
 * \brief Operators bridged from Relay.
 */
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_GENERIC_ATTR_OP_FROM_RELAY("raf.op._all_to_all", "raf.op._all_to_all");
// alltoallv is not included here since we only use them in our optimization

}  // namespace from_relay
}  // namespace op
}  // namespace raf