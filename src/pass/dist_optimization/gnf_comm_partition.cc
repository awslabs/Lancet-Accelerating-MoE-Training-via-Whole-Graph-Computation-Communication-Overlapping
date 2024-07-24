/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file gnf_comm_partition.cc
 * \brief Partitions large comm operators for better scheduling. Only
 *        needed during lancet profiling.
 */
#include <set>
#include <sstream>
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/dist_context.h"
#include "raf/profiler.h"
#include "raf/stream_pool.h"
#include "../common.h" 
#include "../../common/shape_utils.h"

#ifdef RAF_USE_NCCL
#include "../../op/dialect/nccl/communication_utils.h"
#endif

namespace raf {
namespace pass {
namespace gnf_comm_partition {

using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShapeVecFromTensorType;
using op::ArrayToIntTuple;

class GNFCommPartitioner : public ExprMutator {
public:
  Expr Run(Expr e, size_t partition_size) {
    partition_size_ = partition_size;
    return this->Mutate(e);
  }

  Expr VisitExpr_(const CallNode* call) override {
    if (call->op.as<OpNode>() && (Downcast<Op>(call->op) == allreduce_op_ || Downcast<Op>(call->op) == allgather_op_ 
                                  || Downcast<Op>(call->op) == reduce_scatter_op_)) {
      Array<Expr> visited_args;
      for (auto arg : call->args) {
        visited_args.push_back(this->Mutate(arg));
      }
      CHECK(call->checked_type_.defined()) << "Type of comm operator is not defined.";
      auto tensor_type_node = call->checked_type_.as<TensorTypeNode>();
      CHECK(tensor_type_node) << "Output of comm operator is not a tensor. GNFCommPartitioner expects unfused comm ops only. Op: " << GetRef<Call>(call);
      size_t output_size = BytesCompactTensor(tensor_type_node);
      if (output_size > partition_size_) {
        bool is_reduce_scatter = Downcast<Op>(call->op) == reduce_scatter_op_;
        return PartitionCollective(call, visited_args, is_reduce_scatter);
      }
    }
    return ExprMutator::VisitExpr_(call);
  }

protected:
  Expr PartitionCollective(const CallNode* call, const Array<Expr>& visited_args, bool is_reduce_scatter) {
    auto output_size = BytesCompactTensor(call->checked_type_.as<TensorTypeNode>());
    auto output_shape = GetShapeVecFromTensorType(Downcast<TensorType>(call->checked_type()));
    auto input = visited_args[0];
    CHECK(input->checked_type_.defined()) << "Type of input to comm operator is not defined.";
    CHECK(input->checked_type_.as<TupleTypeNode>() && input->checked_type_.as<TupleTypeNode>()->fields.size() == 1);
    auto input_tensor_type = Downcast<TensorType>(Downcast<TupleType>(input->checked_type())->fields[0]);
    auto input_shape = GetShapeVecFromTensorType(input_tensor_type);
    int expected_n_partition = output_size / partition_size_ + 1;
    CHECK(input_shape.size() == output_shape.size()) << "Input and output dimension of comm operator are not the same (" << input_shape.size() << " vs " << output_shape.size() << ").";
    int partition_axis = input_shape.size() - 1;
    if (partition_axis <= 0) {
      // 0 or 1d tensor, ignore
      return GetRef<Call>(call);
    }
    int last_dim_size_output = output_shape[output_shape.size() - 1];
    int last_dim_size_input = input_shape[input_shape.size() - 1];
    // we partition along the last dimension unless it's one d only.
    // round up to the nearest multiple of expected_n_partition
    while ((last_dim_size_output % expected_n_partition != 0 || last_dim_size_input % expected_n_partition != 0) && expected_n_partition >= 1) {
      expected_n_partition--;
    }
    if (expected_n_partition == 1) {
      return GetRef<Call>(call);
    }
    LOG(INFO) << "Partitioning " << Downcast<Op>(call->op) << " with output size " << output_size << " into " << expected_n_partition << " partitions.";
    // partition input tensors
    Array<Expr> partition_args = {Downcast<Tuple>(visited_args[0])->fields[0],
                               MakeConstant(ScalarValue::make(expected_n_partition)),
                               MakeConstant(ScalarValue::make(partition_axis))};
    auto partition_expr = Call(partition_op_, partition_args);
    Array<Expr> partitioned_comm;
    for (int i = 0; i < expected_n_partition; i++) {
      Array<Expr> comm_args;
      comm_args.push_back(Tuple({TupleGetItem(partition_expr, i)}));
      if (is_reduce_scatter) {
        auto out_shape = GetShapeVecFromTensorType(Downcast<TensorType>(call->checked_type()));
        std::vector<int64_t> shape_indices = {(int64_t)out_shape.size()};
        out_shape[partition_axis] = out_shape[partition_axis] / expected_n_partition;
        comm_args.push_back(MakeConstant(ArrayToIntTuple(out_shape)));
        comm_args.push_back(MakeConstant(ArrayToIntTuple(shape_indices)));
        comm_args.push_back(visited_args[3]);
      } else {
        for (size_t j = 1; j < visited_args.size(); j++) {
          comm_args.push_back(visited_args[j]);
        }
      }
      auto partitioned_comm_i = Call(call->op, comm_args);
      partitioned_comm.push_back(partitioned_comm_i);
    }
    Array<Expr> concat_args = {Tuple(partitioned_comm), MakeConstant(ScalarValue::make(partition_axis))};
    Expr concat_expr = Call(concat_op_, concat_args);
    return concat_expr;
  }

private:
  int64_t partition_size_;
  Op allreduce_op_ = op::GetDialectOp("raf.op.nccl._allreduce");
  Op reduce_scatter_op_ = op::GetDialectOp("raf.op.nccl._reduce_scatter");
  Op allgather_op_ = op::GetDialectOp("raf.op.nccl._allgather");
  Op partition_op_ = op::GetDialectOp("raf.op.tvm.split");
  Op concat_op_ = op::GetDialectOp("raf.op.tvm.concatenate");
};

} // namespace gnf_comm_partition

Pass PartitionCommGNF() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    Bool use_partition = pc->GetConfig("raf.dp_schedule.partition_large_comm", Bool(false)).value();
    if (use_partition) {
      // default partition size 25MB
      size_t partition_size = 25 * 1024 * 1024;
      if(const char* partition_size_str = getenv("LANCET_COMM_PARTITION_SIZE")) {
        partition_size = std::stol(std::string(partition_size_str));
      }
      return Downcast<Function>(gnf_comm_partition::GNFCommPartitioner().Run(f, partition_size));
    }
    return f;
  };
  auto partition_comm = CreateRAFFunctionPass(pass_func, 0, "PartitionCommGNF", {});
  return RAFSequential({InferType(), partition_comm, DeadCodeElimination(), InferType()}, "PartitionCommGNF");
}

RAF_REGISTER_GLOBAL("raf.pass_.PartitionCommGNF").set_body_typed(PartitionCommGNF);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.dp_schedule.partition_large_comm", tvm::Bool);

} // namespace pass
} // namespace raf