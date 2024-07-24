/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/collective_comm.cc
 * \brief Typing of collective communicate operators
 */
#include <algorithm>
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>
#include "raf/dist_context.h"
#include "raf/type.h"
#include "../schema/communication.h"
#include "./utils.h"
#include "../../common/shape_utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
using raf::distributed::DistContext;

template <typename T>
Type IdentityType(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  if (args->x.size() == 1) {
    return GetType(args->x[0]);
  }
  Array<Type> x = {};
  std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
  return TupleType(x);
}

RAF_OP_TYPE("raf.op._allreduce", "NCCLAllReduce", IdentityType<AllreduceArgs>);
RAF_OP_TYPE("raf.op._all_to_all", "NCCLAllToAll", IdentityType<AllToAllArgs>);
RAF_OP_TYPE("raf.op._broadcast", "NCCLBroadcast", IdentityType<BroadcastArgs>);
RAF_OP_TYPE("raf.op._reduce", "NCCLReduce", IdentityType<CommReduceArgs>);


Type AllToAllvInfer(const CallValues& value) {
  const auto* args = value->args.as<AllToAllvArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  Type input_type;
  if (args->x.size() == 1) {
    input_type = GetType(args->x[0]);
  } else {
    Array<Type> x = {};
    std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
    input_type = TupleType(x);
  }
  Type send_counts_type;
  if (args->send_counts.size() == 1) {
    send_counts_type = GetType(args->send_counts[0]);
    CHECK(send_counts_type->IsInstance<TensorTypeNode>());
    DataType dtype = send_counts_type.as<TensorTypeNode>()->dtype;
    CHECK(dtype.is_uint() && dtype.bits() == 64);
  } else {
    Array<Type> x = {};
    std::transform(args->send_counts.begin(), args->send_counts.end(), std::back_inserter(x), GetType);
    for(auto it = x.begin(); it != x.end(); ++it) {
      CHECK((*it)->IsInstance<TensorTypeNode>());
      DataType dtype = (*it).as<TensorTypeNode>()->dtype;
      CHECK(dtype.is_uint() && dtype.bits() == 64);
    }
    send_counts_type = TupleType(x);
  }
  return TupleType({input_type, send_counts_type});
}

RAF_OP_TYPE("raf.op._all_to_allv", "NCCLAllToAllv", AllToAllvInfer);

// Also register relay type for AllToAll
// copied from IdentityRel in tvm src/relay/op/type_relations.cc, but return tensor when input tuple has only one element
bool All2AllRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  Type out_type;
  auto tt = Downcast<TupleType>(types[0]);
  if(tt->fields.size() == 1) {
    out_type = tt->fields[0];
  } else {
    out_type = tt;
  }
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], out_type);
  }
  return true;
}

RELAY_REGISTER_OP("raf.op._all_to_all").add_type_rel("AllToAll", All2AllRel);

Type ReduceScatterInfer(const CallValues& value) {
  const auto* args = value->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  CHECK_GE(args->x.size(), 1U);
  DataType dtype = Downcast<TensorType>(GetType(args->x[0]))->dtype;
  std::vector<int64_t> shapes = args->shapes;
  std::vector<int64_t> shape_indices = args->shape_indices;
  Array<Type> tuple_types;
  size_t index = 0;
  for (int i = 0; i < args->shape_indices.size(); ++i) {
    Array<IndexExpr> shape = {};
    for (auto it = args->shapes.begin() + index;
         it != args->shapes.begin() + args->shape_indices[i]; ++it) {
      shape.push_back(Integer(*it));
    }
    tuple_types.push_back(TensorType(shape, dtype));
    index = shape_indices[i];
  }
  if (tuple_types.size() == 1) {
    return tuple_types[0];
  }
  else {
    return TupleType(tuple_types);
  }
}

RAF_OP_TYPE("raf.op._reduce_scatter", "NCCLReduceScatter", ReduceScatterInfer);

Type SendInfer(const CallValues& value) {
  const auto* args = value->args.as<SendArgs>();
  CHECK(args != nullptr);
  const auto& ty = Downcast<TensorType>(GetType(args->x));
  return TensorType({}, ty->dtype);
}

RAF_OP_TYPE("raf.op._send", "NCCLSend", SendInfer);

Type RecvInfer(const CallValues& value) {
  const auto* args = value->args.as<RecvArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  Array<PrimExpr> shape;
  for (const auto& s : args->shape) {
    shape.push_back(Integer(s));
  }
  return TensorType(shape, DataType(ir::String2DLDataType(args->dtype)));
}

RAF_OP_TYPE("raf.op._recv", "NCCLRecv", RecvInfer);

Type AllGatherInfer(const CallValues& value) {
  const auto* args = value->args.as<AllgatherArgs>();
  CHECK(args != nullptr);
  auto dctx = DistContext::Global();
  Array<Type> fields = {};
  for (int i = 0; i < args->x.size(); ++i) {
    auto ttype = GetType(args->x[i]).as<TensorTypeNode>();
    auto shape = ttype->shape;
    auto new_size = shape[args->axis].as<IntImmNode>()->value * dctx->size;
    shape.Set(args->axis, Integer(new_size));
    fields.push_back(TensorType(shape, DataType(ttype->dtype)));
  }
  if (fields.size() == 1) {
    return fields[0];
  } else {
    return TupleType(fields);
  }
}

RAF_OP_TYPE("raf.op._allgather", "NCCLAllGather", AllGatherInfer);

}  // namespace op
}  // namespace raf
