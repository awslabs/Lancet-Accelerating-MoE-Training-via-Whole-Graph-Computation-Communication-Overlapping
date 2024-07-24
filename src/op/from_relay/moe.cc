/*!
 * Copyright (c) 2022 by Contributors
 * \file ./src/op/from_relay/moe.cc
 * \brief Operators bridged from Relay.
 */
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY("raf.op.moe_encode", "raf.op.moe_encode",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
        Array<Expr> raf_args = args;
        // capacity_factor
        raf_args.push_back(MakeConstant(FloatValue::make(DataType::Float(32), 1.0)));
        return raf_args;
});

RAF_OP_FROM_RELAY("raf.op.moe_encode_batch_prioritized", "raf.op.moe_encode_batch_prioritized",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
        Array<Expr> raf_args = args;
        // n_partitions
        raf_args.push_back(MakeConstant(IntValue::make(DataType::UInt(64), 1)));
        // partition_id
        raf_args.push_back(MakeConstant(IntValue::make(DataType::UInt(64), 0)));
        // capacity_factor
        raf_args.push_back(MakeConstant(FloatValue::make(DataType::Float(32), 1.0)));
        return raf_args;
});

RAF_GENERIC_ATTR_OP_FROM_RELAY("raf.op.moe_decode", "raf.op.moe_decode");

}  // namespace from_relay
}  // namespace op
}  // namespace raf