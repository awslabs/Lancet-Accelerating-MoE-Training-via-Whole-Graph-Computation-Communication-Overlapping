/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/from_relay/init.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/op_utils.h"
#include "tvm/relay/attrs/transform.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY(
    "zeros", "raf.op.zeros",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
      Array<Expr> raf_args = args;
      const auto* relay_attrs = attrs.as<InitOpAttrs>();
      raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->shape.value_or({}))));
      raf_args.push_back(MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
      return raf_args;
    });

RAF_OP_FROM_RELAY(
    "one_hot", "raf.op.one_hot",
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
      Array<Expr> raf_args = args;
      const auto* relay_attrs = attrs.as<OneHotAttrs>();
      raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->depth)));
      raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
      raf_args.push_back(MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(relay_attrs->dtype))));
      return raf_args;
    });

}  // namespace from_relay
}  // namespace op
}  // namespace raf