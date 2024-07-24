/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/nccl.cc
 * \brief Communication operators implmentated by NCCL
 */
#include <vector>
#include <chrono>
#include <thread>
#include "raf/op_utils.h"
#include "raf/dist_context.h"
#include "../../schema/communication.h"
#include "./communication_utils.h"

namespace raf {
namespace op {
namespace communication {
namespace nccl {

using namespace tvm::runtime;
using namespace distributed;
using namespace distributed::communicator;
using namespace raf::op::schema;
using common::shape_utils::BytesCompactTensor;
using stream_pool::StreamTagEnum;

RAF_REGISTER_DIALECT("nccl").set_enable(DevType::kCUDA());

class NCCLAllreduce : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  DType dtype;
  ncclRedOp_t compute;

  explicit NCCLAllreduce(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._allreduce");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<raf::op::schema::AllreduceArgs>();
    auto& tv = args->x;

    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "Allreduce with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() > 1) {
      RequestWorkspace(&fused_data, cv->device, total_size);
    }
  }

 public:
  ~NCCLAllreduce() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._allreduce"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<AllreduceArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, value::Value output) {
    // `output` is expected to be a TensorValue if input contains only
    // one tensor (TupleValue otherwise) to make it compatible with vm

    // We can use sleep to test communication scheduling locally.
    // using namespace std::this_thread;
    // using namespace std::chrono;
    // sleep_until(system_clock::now() + nanoseconds(200));
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto input_x = Downcast<value::TupleValue>(inputs[0]);

    // Fuse Tensor
    size_t dtype_size = 0;
    // although in IR allreduce takes and outputs tuples, in VM output will be a tensor value
    // if it only have one field
    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      DLTensor* out = output;
      dtype_size = GetSizeInBytes(x->dtype);
      NCCL_CALL(ncclAllReduce(x->data, out->data, total_size / dtype_size, dtype, compute,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      return;
    }
    size_t offset = 0;
    for (int i = 0; i < input_x->fields.size(); ++i) {
      DLTensor* x = input_x->fields[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream));
      offset += tuple_sizes[i];
      dtype_size = GetSizeInBytes(x->dtype);
    }

    // Allreduce
    NCCL_CALL(ncclAllReduce(fused_data, fused_data, total_size / dtype_size, ncclFloat, compute,
                            (ncclComm_t)nccl_comm, (cudaStream_t)stream));
    // UnFuse Tensor
    value::TupleValue out = Downcast<value::TupleValue>(output);
    auto& of = out->fields;
    for (int i = of.size() - 1; i >= 0; --i) {
      DLTensor* x = of[i];
      offset -= tuple_sizes[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      CUDA_CALL(cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream));
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllreduce(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _allreduce, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._allreduce", NCCLAllreduce::make);

class NCCLAllToAll : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  DType dtype;
  void* in_buffer;
  void* out_buffer;
  size_t total_input_size = 0;
  std::vector<size_t> tuple_sizes;

  explicit NCCLAllToAll(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._all_to_all");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<raf::op::schema::AllToAllArgs>();
    auto& tv = args->x;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_input_size += size;
      dtype = x->dtype;
    }
#if NCCL_VERSION_CODE < 20700
    LOG(FATAL) << "AllToAll is not supported in NCCL < 2.7.0";
#endif
    if (tv.size() == 1) return;
    RequestWorkspace(&in_buffer, cv->device, total_input_size);
    RequestWorkspace(&out_buffer, cv->device, total_input_size);
  }

 public:
  ~NCCLAllToAll() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._all_to_all"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<AllToAllArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto input_x = Downcast<value::TupleValue>(inputs[0]);

    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      DLTensor* out = output;

      int nccl_num_ranks;
      NCCL_CALL(ncclCommCount((ncclComm_t)nccl_comm, &nccl_num_ranks));
      CHECK_EQ(reinterpret_cast<Communicator*>(communicator)->GetSize(), nccl_num_ranks)
        << "NCCL communicator world size does not match with Communicator.";

      int64_t size = 1;
      for (int i = 0; i < x->ndim; ++i) {
        size *= x->shape[i];
      }
      CHECK(size % nccl_num_ranks == 0) << "Cannot evenly distribute input tensor to all ranks.";
      int64_t dtype_size_in_bytes = GetSizeInBytes(x->dtype);

      size_t per_rank_bytes = total_input_size / nccl_num_ranks;
      size_t size_per_rank = per_rank_bytes / dtype_size_in_bytes;

      char* send_buffer = (char*) x->data;
      char* recv_buffer = (char*) out->data;
      if (size != 0) {
        NCCL_CALL(ncclGroupStart());
        for(size_t i=0; i<nccl_num_ranks; i++) {
          NCCL_CALL(ncclSend(send_buffer + i * per_rank_bytes, size_per_rank, dtype, i, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
          NCCL_CALL(ncclRecv(recv_buffer + i * per_rank_bytes, size_per_rank, dtype, i, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
        }
        NCCL_CALL(ncclGroupEnd());
      }
    } else {      
      int nccl_num_ranks;
      NCCL_CALL(ncclCommCount((ncclComm_t)nccl_comm, &nccl_num_ranks));
      CHECK_EQ(reinterpret_cast<Communicator*>(communicator)->GetSize(), nccl_num_ranks)
        << "NCCL communicator world size does not match with Communicator.";

      // fuse-reorder tensors into a buffer.
      size_t offset = 0;
      size_t itvl = total_input_size / nccl_num_ranks;
      for (int i = 0; i < input_x->fields.size(); ++i) {
        DLTensor* x = input_x->fields[i];
        size_t size_per_rank = tuple_sizes[i] / nccl_num_ranks;
        for (int j = 0; j < nccl_num_ranks; ++j) {
          void* in = reinterpret_cast<uint8_t*>(in_buffer) + offset + j * itvl;
          void* x_ = reinterpret_cast<uint8_t*>(x->data) + j * size_per_rank;
          CUDA_CALL(cudaMemcpyAsync(in, x_, size_per_rank, cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream));
        }
        offset += size_per_rank;
      }

      // all2all
      DType dtype = ((DLTensor*) input_x->fields[0])->dtype;
      int64_t dtype_size_in_bytes = GetSizeInBytes(dtype);
      size_t total_per_rank_bytes = total_input_size / nccl_num_ranks;
      size_t total_size_per_rank = total_per_rank_bytes / dtype_size_in_bytes;
      char* send_buffer = (char*) in_buffer;
      char* recv_buffer = (char*) out_buffer;
      NCCL_CALL(ncclGroupStart());
      for(size_t i=0; i<nccl_num_ranks; i++) {
        NCCL_CALL(ncclSend(send_buffer + i * total_per_rank_bytes, total_size_per_rank, dtype, i, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
        NCCL_CALL(ncclRecv(recv_buffer + i * total_per_rank_bytes, total_size_per_rank, dtype, i, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      }
      NCCL_CALL(ncclGroupEnd());

      // defuse-reorder tensors
      auto& of = Downcast<value::TupleValue>(output)->fields;
      offset = 0;
      for (int i = 0; i < of.size(); ++i) {
        DLTensor* x = of[i];
        size_t size_per_rank = tuple_sizes[i] / nccl_num_ranks;
        for (int j = 0; j < nccl_num_ranks; ++j) {
          void* out = reinterpret_cast<uint8_t*>(out_buffer) + offset + j * itvl;
          void* x_ = reinterpret_cast<uint8_t*>(x->data) + j * size_per_rank;
          CUDA_CALL(cudaMemcpyAsync(x_, out, size_per_rank, cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream));
        }
        offset += size_per_rank;
      }
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllToAll(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _all_to_all, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._all_to_all", NCCLAllToAll::make);

class NCCLAllToAllv : public raf::op::OpEnv {
  void* comm_stream;
  void* communicator;
  DType dtype;
  size_t total_input_size = 0;
  int nccl_world_size = 0;
  int rank = 0;
  int n_chunks_per_device = 0;
  size_t send_count_buffer_size = 0;
  size_t send_count_buffer_n_elements = 0;
  void* recv_count_buffer_host = nullptr;
  void* recv_count_buffer_device = nullptr;
  void* send_count_buffer_host = nullptr;

  explicit NCCLAllToAllv(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._all_to_allv");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x"), fschema_index[op]("send_counts")};
    RequestStream(&comm_stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<raf::op::schema::AllToAllvArgs>();
    auto& tv = args->x;
    CHECK(tv.size() == 1) << "AllToAllv currently only supports one input tensor.";
    DLTensor* x = tv[0];
    size_t size = BytesCompactTensor(*x);
    total_input_size = size;
    dtype = x->dtype;
    nccl_world_size = DistContext::Global()->size;


#if NCCL_VERSION_CODE < 20700
    LOG(FATAL) << "AllToAll is not supported in NCCL < 2.7.0";
#endif
    // allocate buffers

    // send_counts is a 1d tensor with nccl_world_size * n_chunks_per_device
    // elements, where each chunk is allowed to have a different actual size
    auto& tv_sc = args->send_counts;
    CHECK(tv_sc.size() == 1) << "AllToAllv currently only supports one send_counts tensor.";
    DLTensor* sc = tv_sc[0];
    send_count_buffer_size = BytesCompactTensor(*sc);
    auto sc_dtype = sc->dtype;
    send_count_buffer_n_elements = send_count_buffer_size / (sc_dtype.bits / 8);
    CHECK(send_count_buffer_n_elements % nccl_world_size == 0);
    n_chunks_per_device = send_count_buffer_n_elements / nccl_world_size;
    // send_buffer on each rank has world_size * n_chunks_per_device elements (each is an integer)
    // we exchange the results on each rank during execution
    // We also need to allocate memory on the host
    CUDA_CALL(cudaMallocHost(&send_count_buffer_host, send_count_buffer_size));
    CUDA_CALL(cudaMallocHost(&recv_count_buffer_host, send_count_buffer_size));
  }

 public:
  ~NCCLAllToAllv() {
    // free the pinned memory
    CUDA_CALL(cudaFreeHost(send_count_buffer_host));
    CUDA_CALL(cudaFreeHost(recv_count_buffer_host));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._all_to_allv"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<AllToAllvArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end())),
             TupleValue::make(ir::Array<Value>(args->send_counts.begin(), args->send_counts.end()))}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    rank = reinterpret_cast<Communicator*>(communicator)->GetRank();
    CHECK(inputs.size() == 2);
    auto input_x = Downcast<value::TupleValue>(inputs[0]);
    auto input_send_counts = Downcast<value::TupleValue>(inputs[1]);
    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      DLTensor* send_bytes = input_send_counts->fields[0];
      auto outputs = Downcast<value::TupleValue>(output);
      DLTensor* x_out = outputs->fields[0];
      DLTensor* recv_counts_out = outputs->fields[1];

      void* recv_count_buffer_device = recv_counts_out->data;

      // 1. perform a all-gather on send_bytes and wait for it to finish
      NCCL_CALL(ncclGroupStart());
      for (int i=0; i<nccl_world_size; i++) {
        if (i == rank) {
          continue;
        }
        uint64_t* send_buffer_start = reinterpret_cast<uint64_t*>(send_bytes->data) + (i * n_chunks_per_device);
        uint64_t* recv_buffer_start = reinterpret_cast<uint64_t*>(recv_count_buffer_device) + (i * n_chunks_per_device);
        NCCL_CALL(ncclSend((void*)(send_buffer_start), n_chunks_per_device, ncclUint64, i, (ncclComm_t)nccl_comm, (cudaStream_t)comm_stream));
        NCCL_CALL(ncclRecv((void*)(recv_buffer_start), n_chunks_per_device, ncclUint64, i, (ncclComm_t)nccl_comm, (cudaStream_t)comm_stream));
      }
      NCCL_CALL(ncclGroupEnd());
      uint64_t* self_send_buffer_start = reinterpret_cast<uint64_t*>(send_bytes->data) + (rank * n_chunks_per_device);
      uint64_t* self_recv_buffer_start = reinterpret_cast<uint64_t*>(recv_count_buffer_device) + (rank * n_chunks_per_device);
      CUDA_CALL(cudaMemcpyAsync(self_recv_buffer_start, self_send_buffer_start, n_chunks_per_device * sizeof(uint64_t), cudaMemcpyDeviceToDevice, (cudaStream_t)comm_stream));
      // 2. copy the results to the host and wait for the copy to finish
      CUDA_CALL(cudaMemcpyAsync(send_count_buffer_host, send_bytes->data, send_count_buffer_size, cudaMemcpyDeviceToHost, (cudaStream_t)comm_stream));
      CUDA_CALL(cudaMemcpyAsync(recv_count_buffer_host, recv_count_buffer_device, send_count_buffer_size, cudaMemcpyDeviceToHost, (cudaStream_t)comm_stream));
      CUDA_CALL(cudaStreamSynchronize((cudaStream_t)comm_stream));
      // 3. look at the results and compute the size to send and recv
      size_t x_size_counts = 1;
      for (int i = 0; i < x->ndim; ++i) {
        x_size_counts *= x->shape[i];
      }
      CHECK(x_size_counts % nccl_world_size == 0) << "Cannot evenly distribute input tensor to all ranks.";
      size_t per_rank_x_counts = x_size_counts / nccl_world_size;
      CHECK(per_rank_x_counts % n_chunks_per_device == 0) << "Cannot evenly distribute input tensor to chunks.";
      size_t per_chunk_counts = per_rank_x_counts / n_chunks_per_device;
      int64_t dtype_size_in_bytes = GetSizeInBytes(x->dtype);
      size_t per_chunk_bytes = per_chunk_counts * dtype_size_in_bytes;

      uint64_t* send_counts = reinterpret_cast<uint64_t*>(send_count_buffer_host);
      uint64_t* recv_counts = reinterpret_cast<uint64_t*>(recv_count_buffer_host);
      CUDA_CALL(cudaMemsetAsync(x_out->data, 0, total_input_size, (cudaStream_t)comm_stream));
      // 4. launch the all-to-allv
      char* send_buffer = (char*) x->data;
      char* recv_buffer = (char*) x_out->data;
      NCCL_CALL(ncclGroupStart());
      for(int i=0; i<nccl_world_size; i++) {
        if (i == rank) {
          continue;
        }
        for (int j=0; j< n_chunks_per_device; j++) {
          NCCL_CALL(ncclSend(send_buffer + i * per_chunk_bytes * n_chunks_per_device + j * per_chunk_bytes,
                              send_counts[i * n_chunks_per_device + j], dtype, i, (ncclComm_t)nccl_comm, (cudaStream_t)comm_stream));
          NCCL_CALL(ncclRecv(recv_buffer + i * per_chunk_bytes * n_chunks_per_device + j * per_chunk_bytes,
                              recv_counts[i * n_chunks_per_device + j], dtype, i, (ncclComm_t)nccl_comm, (cudaStream_t)comm_stream));
        }
      }
      NCCL_CALL(ncclGroupEnd());
      // 5. launch memcpy for local send/recvs
      for (int j=0; j < n_chunks_per_device; j++) {
        CUDA_CALL(cudaMemcpyAsync(recv_buffer + rank * per_chunk_bytes * n_chunks_per_device + j * per_chunk_bytes,
                                  send_buffer + rank * per_chunk_bytes * n_chunks_per_device + j * per_chunk_bytes,
                                  send_counts[rank * n_chunks_per_device + j] * dtype_size_in_bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)comm_stream));
      }
    } else {
      LOG(FATAL) << "AllToAllv currently only supports one input tensor.";
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllToAllv(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _all_to_allv, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._all_to_allv", NCCLAllToAllv::make);

class NCCLAllgather : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  void* in_buffer;
  void* out_buffer;
  size_t total_input_size = 0;
  int world_size = 0;
  std::vector<size_t> tuple_sizes;
  bool use_nccl_group = false;

  explicit NCCLAllgather(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._allgather");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<raf::op::schema::AllgatherArgs>();
    auto& tv = args->x;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_input_size += size;
    }
    if (tv.size() == 1) return;
    world_size = DistContext::Global()->size;
    if(const char* use_group_rs_ag_str = getenv("RAF_GROUPED_COLLECTIVE")) {
      auto use_group_rs_ag = std::atoi(use_group_rs_ag_str);
      if (use_group_rs_ag > 0) {
        use_nccl_group = true;
      }
    }
    if (!use_nccl_group) {
      RequestWorkspace(&in_buffer, cv->device, total_input_size);
      RequestWorkspace(&out_buffer, cv->device, total_input_size * world_size);
    }
  }

 public:
  ~NCCLAllgather() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._allgather"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::AllgatherArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto input_x = Downcast<value::TupleValue>(inputs[0]);
    DType dtype = ((DLTensor*) input_x->fields[0])->dtype;
    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      DLTensor* out = output;
      int64_t size = 1;
      for (int i = 0; i < x->ndim; ++i) {
        size *= x->shape[i];
      }
      NCCL_CALL(ncclAllGather(x->data, out->data, size, dtype, (ncclComm_t)nccl_comm,
                              (cudaStream_t)stream));
    } else {
      if (use_nccl_group) {
        auto output_tuple = Downcast<value::TupleValue>(output);
        NCCL_CALL(ncclGroupStart());
        for (int i = 0; i < input_x->fields.size(); ++i) {
          DLTensor* x = input_x->fields[i];
          DLTensor* out = output_tuple->fields[i];
          size_t size = tuple_sizes[i];
          size = size / ((x->dtype.bits + 7) / 8);
          NCCL_CALL(ncclAllGather(x->data, out->data, size, dtype, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
        }
        NCCL_CALL(ncclGroupEnd());
      } else {
        size_t offset = 0;
        int64_t dtype_size_in_bytes = GetSizeInBytes(dtype);
        for (int i = 0; i < input_x->fields.size(); ++i) {
          DLTensor* x = input_x->fields[i];
          void* in = reinterpret_cast<uint8_t*>(in_buffer) + offset;
          CUDA_CALL(cudaMemcpyAsync(in, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream));
          offset += tuple_sizes[i];
        }
        NCCL_CALL(ncclAllGather(in_buffer, out_buffer, total_input_size / dtype_size_in_bytes, dtype, (ncclComm_t)nccl_comm,
                                (cudaStream_t)stream));
        // defuse out buffer
        auto& of = Downcast<value::TupleValue>(output)->fields;
        CHECK(input_x->fields.size() == of.size());
        size_t start_offset = 0;
        for (int i = 0; i < of.size(); ++i) {
          DLTensor* out = of[i];
          for (int j = 0; j < world_size; ++j) {
            char* out_buffer_ = (char*) out_buffer + start_offset + j * total_input_size;
            char* out_ = (char*) out->data + j * tuple_sizes[i];
            CUDA_CALL(cudaMemcpyAsync(out_, out_buffer_, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                      (cudaStream_t)stream));
          }
          start_offset += tuple_sizes[i];
        }
      }
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllgather(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _allgather, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._allgather", NCCLAllgather::make);

class NCCLReduceScatter : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  void* in_buffer;
  void* out_buffer;
  // Total output size
  size_t total_size = 0;
  // Tuple sizes for output tensors in bytes
  std::vector<size_t> tuple_sizes;
  // Split sizes for input tensors in bytes
  std::vector<size_t> split_sizes;
  std::vector<size_t> split_shapes;
  int n_parts;
  ncclRedOp_t compute;
  bool use_nccl_group = false;

  explicit NCCLReduceScatter(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._reduce_scatter");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<ReduceScatterArgs>();
    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "ReduceScatter with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }
    if (args->shape_indices.size() > 1) {
      if(const char* use_group_rs_ag_str = getenv("RAF_GROUPED_COLLECTIVE")) {
        auto use_group_rs_ag = std::atoi(use_group_rs_ag_str);
        if (use_group_rs_ag > 0) {
          use_nccl_group = true;
        }
      }
      // Should defuse output tensor
      auto tv = Downcast<value::TupleValue>(cv->out);
      int64_t nbytes = 0;
      tuple_sizes.clear();
      for (int i = 0; i < tv->fields.size(); ++i) {
        DLTensor* x = tv->fields[i];
        size_t size = BytesCompactTensor(*x);
        tuple_sizes.push_back(size);
        total_size += size;
      }
      DLTensor* x = tv->fields[0];
      if (!use_nccl_group) {
        RequestWorkspace(&out_buffer, cv->device, total_size);
      }
      total_size /= ((x->dtype.bits + 7) / 8);
    }
    else {
      const DLTensor* out = cv->out;
      total_size = BytesCompactTensor(*out) / ((out->dtype.bits + 7) / 8);
    }

    if (args->x.size() > 1) {
      // Should fuse input tensors
      size_t total_input_size = 0;
      for (int i = 0; i < args->x.size(); ++i) {
        DLTensor* x = args->x[i];
        size_t size = BytesCompactTensor(*x);
        split_sizes.push_back(size);
        total_input_size += size;
      }
      DLTensor* x = args->x[0];
      size_t total_input_size_bytes = total_input_size;
      total_input_size /= ((x->dtype.bits + 7) / 8);
      n_parts = DistContext::Global()->size;
      for (int i = 0; i < split_sizes.size(); ++i) {
        CHECK(split_sizes[i] % n_parts == 0) << "Input tensor " << i << " with size " << split_sizes[i] << " is not divisible by " << n_parts;
        split_sizes[i] /= n_parts;
      }
      if (!use_nccl_group) {
        RequestWorkspace(&in_buffer, cv->device, total_input_size_bytes);
      }
    }
    else {
      const DLTensor* x = args->x[0];
      size_t total_input_size = BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8);
      CHECK((total_input_size % total_size) == 0) 
        << "Input tensor cannot be divided to parts of size " << total_size << ".";
    }
  }

 public:
  ~NCCLReduceScatter() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._reduce_scatter"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<ReduceScatterArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    size_t offset = 0;

    auto tv = Downcast<value::TupleValue>(inputs[0]);
    if (tv->fields.size() == 1) {
      DLTensor* x = tv->fields[0];
      DLTensor* out = output;
      NCCL_CALL(ncclReduceScatter(x->data, out->data, total_size, DType(x->dtype), compute, (ncclComm_t)nccl_comm,
                                  (cudaStream_t)stream));
    } else {
      if (use_nccl_group) {
        auto out_tuple = Downcast<value::TupleValue>(output);
        NCCL_CALL(ncclGroupStart());
        for (int i = 0; i < tv->fields.size(); ++i) {
          DLTensor* x = tv->fields[i];
          DLTensor* out = out_tuple->fields[i];
          size_t size = split_sizes[i];
          size = size / ((x->dtype.bits + 7) / 8);
          NCCL_CALL(ncclReduceScatter(x->data, out->data, size, DType(x->dtype), compute, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
        }
        NCCL_CALL(ncclGroupEnd());
      } else {
        DLTensor* x = tv->fields[0];
        DType dtype = x->dtype;
        size_t offset = 0;
        // Fuse reorder tensors.
        for (int i = 0; i < n_parts; ++i) {
          for (int j = 0; j < tv->fields.size(); ++j) {
            DLTensor* x = tv->fields[j];
            void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(in_buffer) + offset;
            void* input = reinterpret_cast<uint8_t*>(x->data) + i * split_sizes[j];
            CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, input, split_sizes[j], cudaMemcpyDeviceToDevice,
                                      (cudaStream_t)stream));
            offset += split_sizes[j];
          }
        }
        if (tuple_sizes.size() == 1) {
          DLTensor* out = output;
          NCCL_CALL(ncclReduceScatter(in_buffer, out->data, total_size, dtype, compute, (ncclComm_t)nccl_comm,
                                      (cudaStream_t)stream));
        }
        else {
          NCCL_CALL(ncclReduceScatter(in_buffer, out_buffer, total_size, dtype, compute, (ncclComm_t)nccl_comm,
                                      (cudaStream_t)stream));
          auto out_tv = Downcast<value::TupleValue>(output)->fields;
          size_t offset = 0;
          for (int i = 0; i < out_tv.size(); ++i) {
            DLTensor* x = out_tv[i];
            void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(out_buffer) + offset;
            void* output = reinterpret_cast<uint8_t*>(x->data);
            CUDA_CALL(cudaMemcpyAsync(output, buffer_data_at_offset, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                      (cudaStream_t)stream));
            offset += tuple_sizes[i];
          }
        }
      }
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLReduceScatter(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _reduce_scatter, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._reduce_scatter", NCCLReduceScatter::make);

class NCCLBroadcast : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  DType dtype;
  int root;

  explicit NCCLBroadcast(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._broadcast");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    auto args = cv->args.as<raf::op::schema::BroadcastArgs>();
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto& tv = args->x;
    root = args->root;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() == 1) return;
    RequestWorkspace(&fused_data, cv->device, total_size);
  }

 public:
  ~NCCLBroadcast() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._broadcast"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::BroadcastArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t dtype_size = 0;
    if (tv->fields.size() == 1) {
      DLTensor* x = tv->fields[0];
      DLTensor* out = output;
      dtype_size = GetSizeInBytes(x->dtype);
      NCCL_CALL(ncclBroadcast(x->data, out->data, total_size / dtype_size, dtype, root,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      return;
    }

    size_t offset = 0;
    for (int i = 0; i < tv->fields.size(); ++i) {
      DLTensor* x = tv->fields[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream));
      offset += tuple_sizes[i];
      CHECK(dtype_size == 0 || dtype_size == GetSizeInBytes(x->dtype))
          << "Broadcast requires tensors to be the same type.";
      dtype_size = GetSizeInBytes(x->dtype);
    }

    NCCL_CALL(ncclBroadcast(fused_data, fused_data, total_size / dtype_size, dtype, root,
                            (ncclComm_t)nccl_comm, (cudaStream_t)stream));

    // UnFuse Tensor
    value::TupleValue out = Downcast<value::TupleValue>(output);
    auto& of = out->fields;
    for (int i = of.size() - 1; i >= 0; --i) {
      DLTensor* x = of[i];
      offset -= tuple_sizes[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      CUDA_CALL(cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream));
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLBroadcast(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _broadcast, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._broadcast", NCCLBroadcast::make);

class NCCLSend : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  int peer;

  explicit NCCLSend(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._send");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    const auto* args = cv->args.as<raf::op::schema::SendArgs>();
    CHECK(args);
    peer = args->peer;
  }

 public:
  ~NCCLSend() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._send"));
  }

  void Execute(const CallValues& cv) {
    const auto* args = cv->args.as<raf::op::schema::SendArgs>();
    CHECK(args);
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    const DLTensor* x = inputs[0];
    NCCL_CALL(ncclSend(x->data, BytesCompactTensor(*x) / (x->dtype.bits / 8), DType(x->dtype), peer,
                       (ncclComm_t)nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLSend(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _send, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._send", NCCLSend::make);

class NCCLRecv : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  int peer;
  std::vector<int64_t> shape;
  DType dtype;

  explicit NCCLRecv(const CallValues& cv) {
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    const auto* args = cv->args.as<raf::op::schema::RecvArgs>();
    CHECK(args);
    peer = args->peer;
    shape = args->shape;
    dtype = ir::String2DLDataType(args->dtype);
  }

 public:
  ~NCCLRecv() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._recv"));
  }

  void Execute(const CallValues& cv) {
    Execute({}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    DLTensor* out = output;
    NCCL_CALL(ncclRecv(out->data, BytesCompactTensor(*out) / (out->dtype.bits / 8),
                       DType(out->dtype), peer, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLRecv(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _recv, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._recv", NCCLRecv::make);

class NCCLReduce : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  ncclRedOp_t compute;
  int root;
  DType dtype;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  void* fused_data;

  explicit NCCLReduce(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._reduce");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<raf::op::schema::CommReduceArgs>();
    root = args->root;
    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "Reduce with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }

    auto& tv = args->x;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() >= 1) {
      RequestWorkspace(&fused_data, cv->device, total_size);
    }
  }

 public:
  ~NCCLReduce() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._reduce"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::CommReduceArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto input_x = Downcast<value::TupleValue>(inputs[0]);
    size_t dtype_size = 0;
    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      DLTensor* out = output;
      dtype_size = GetSizeInBytes(x->dtype);

      size_t dtype_size = GetSizeInBytes(x->dtype);
      NCCL_CALL(ncclReduce(x->data, out->data, total_size / dtype_size, dtype, compute, root,
                           (ncclComm_t)nccl_comm, (cudaStream_t)stream));
    } else {
      size_t offset = 0;
      for (int i = 0; i < input_x->fields.size(); ++i) {
        DLTensor* x = input_x->fields[i];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
        CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                  (cudaStream_t)stream));
        offset += tuple_sizes[i];
        dtype_size = GetSizeInBytes(x->dtype);
      }

      NCCL_CALL(ncclReduce(fused_data, fused_data, total_size / dtype_size, dtype, compute, root,
                           (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      // UnFuse Tensor
      value::TupleValue out = Downcast<value::TupleValue>(output);
      auto& of = out->fields;
      for (int i = of.size() - 1; i >= 0; --i) {
        DLTensor* x = of[i];
        offset -= tuple_sizes[i];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
        CUDA_CALL(cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                                  (cudaStream_t)stream));
      }
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLReduce(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _reduce, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._reduce", NCCLReduce::make);

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace raf