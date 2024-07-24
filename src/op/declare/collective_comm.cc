/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/collective_comm.cc
 * \brief Declaration of collective communication operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "raf/communicator.h"
#include "../schema/communication.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;
using namespace raf::distributed::communicator;
using tensor::Tensor;

void AllReduce(const CallValues& call) {
  const auto* args = call->args.as<AllreduceArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  auto& tv = args->x;
  const DLTensor* x = tv[0];
  call->device = x->device;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._allreduce", AllReduce)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{0, 0}});

void AllToAll(const CallValues& call) {
  const auto* args = call->args.as<AllToAllArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  auto& tv = args->x;
  const DLTensor* x = tv[0];
  call->device = x->device;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._all_to_all", AllToAll)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.");


void AllToAllv(const CallValues& call) {
  const auto* args = call->args.as<AllToAllvArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> x_ret_fields;
  auto& x_tv = args->x;
  const DLTensor* x = x_tv[0];
  call->device = x->device;
  for (int i = 0; i < x_tv.size(); ++i) {
    const DLTensor* x = x_tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    x_ret_fields.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  Value x_ret;
  if (x_ret_fields.size() == 1) {
    x_ret = x_ret_fields[0];
  } else {
    x_ret = TupleValue::make(ir::Array<Value>(x_ret_fields.begin(), x_ret_fields.end()));
  }
  auto& send_counts_tv = args->send_counts;
  const DLTensor* send_counts = send_counts_tv[0];
  ir::Array<Value> send_counts_ret_fields;
  for (int i = 0; i < send_counts_tv.size(); ++i) {
    const DLTensor* send_counts = send_counts_tv[i];
    std::vector<int64_t> shape(send_counts->shape, send_counts->shape + send_counts->ndim);
    send_counts_ret_fields.push_back(TensorValue::Assemble(/*dev=*/send_counts->device,
                                        /*dtype=*/send_counts->dtype,
                                        /*shape=*/shape));
  }
  Value send_counts_ret;
  if (send_counts_ret_fields.size() == 1) {
    send_counts_ret = send_counts_ret_fields[0];
  } else {
    send_counts_ret = TupleValue::make(ir::Array<Value>(send_counts_ret_fields.begin(), send_counts_ret_fields.end()));
  }
  if (x_ret_fields.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else {
    ir::Array<Value> ret;
    ret.push_back(x_ret);
    ret.push_back(send_counts_ret);
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._all_to_allv", AllToAllv)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("sendcounts", "Tensor", "The tensor specifying the actual "
                                          "number of elements to sent to each "
                                          "destination.");

void Reduce(const CallValues& call) {
  const auto* args = call->args.as<CommReduceArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  auto& tv = args->x;
  const DLTensor* x = tv[0];
  call->device = x->device;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._reduce", Reduce)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{0, 0}});

void Allgather(const CallValues& call) {
  const auto* args = call->args.as<AllgatherArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  std::vector<BaseTensorValue> tvs = args->x;
  const DLTensor* x = tvs[0];
  call->device = x->device;
  int size = CommunicatorManager::Get()->GetCommunicator()->GetSize();
  for (int i = 0; i < tvs.size(); ++i) {
    const DLTensor* x = tvs[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    shape[args->axis] *= size;
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._allgather", Allgather)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void ReduceScatter(const CallValues& call) {
  const auto* args = call->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  std::vector<BaseTensorValue> tvs = args->x;
  CHECK_GE(tvs.size(), 1U);
  const DLTensor* x = tvs[0];
  call->device = x->device;
  std::vector<int64_t> shapes = args->shapes;
  std::vector<int64_t> shape_indices = args->shape_indices;
  ir::Array<Value> ret;
  int64_t start_idx = 0;
  for (int i = 0; i < shape_indices.size(); ++i) {
    std::vector<int64_t> shape(shapes.begin() + start_idx, shapes.begin() + shape_indices[i]);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
    start_idx = shape_indices[i];
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._reduce_scatter", ReduceScatter)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void Broadcast(const CallValues& call) {
  const auto* args = call->args.as<BroadcastArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  auto& tv = args->x;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) call->callee = ir::NullValue<OpValue>();
  const DLTensor* x = tv[0];
  call->device = x->device;
  if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

RAF_OP_DECLARE("raf.op._broadcast", Broadcast)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void Send(const CallValues& call) {
  const auto* args = call->args.as<SendArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>{});
}

RAF_OP_DECLARE("raf.op._send", Send)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void Recv(const CallValues& call) {
  const auto* args = call->args.as<RecvArgs>();
  CHECK(args != nullptr);
  Device dev(DevType::kCUDA(), CommunicatorManager::Get()->GetCommunicator()->GetLocalRank());
  call->device = dev;
  call->out = TensorValue::Assemble(/*ctx=*/dev,
                                    /*dtype=*/ir::String2DLDataType(args->dtype),
                                    /*shape=*/args->shape);
}

RAF_OP_DECLARE("raf.op._recv", Recv)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

}  // namespace declare
}  // namespace op
}  // namespace raf
