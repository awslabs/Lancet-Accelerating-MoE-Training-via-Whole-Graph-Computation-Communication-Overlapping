/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/context.cc
 * \brief Context of Distributed Settings.
 */
#include "raf/registry.h"
#include "raf/dist_context.h"

namespace raf {
namespace distributed {

using communicator::Communicator;
using communicator::CommunicatorManager;

DistContext DistContext::make() {
  ir::ObjectPtr<DistContextObj> n = ir::make_object<DistContextObj>();
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  n->root_rank = comm->GetRootRank();
  n->rank = comm->GetRank();
  n->size = comm->GetSize();
  n->local_rank = comm->GetLocalRank();
  n->local_size = comm->GetLocalSize();

  return DistContext(n);
}

DistContext DistContext::Global() {
  static DistContext inst = DistContext::make();
  return inst;
}

void EnableDataParallel(bool enable) {
  DistContext::Global()->enable_data_parallel = enable;
}

void OverlapCommForward(bool overlap) {
  DistContext::Global()->overlap_comm_forward = overlap;
}

void ForceSyncAfterComm(bool enable) {
  DistContext::Global()->force_sync_after_comm = enable;
}

void ZeroOpt(int opt_level) {
  DistContext::Global()->zero_opt_level = opt_level;
}

void SetGlobalRank(int rank) {
  DistContext::Global()->rank = rank;
}

void SetGlobalSize(int size) {
  DistContext::Global()->size = size;
}

void SetLocalRankForTuning(int local_rank) {
  DistContext::Global()->local_rank = local_rank;
}

void AutoDPProfilingStartIter(int auto_dp_profiling_start_iter) {
  DistContext::Global()->auto_dp_profiling_start_iter = auto_dp_profiling_start_iter;
}

void AutoDPProfilingEndIter(int auto_dp_profiling_end_iter) {
  DistContext::Global()->auto_dp_profiling_end_iter = auto_dp_profiling_end_iter;
}

void Barrier() {
  CommunicatorManager::Get()->GetCommunicator()->Barrier();
}

RAF_REGISTER_GLOBAL("raf.distributed._make.DistContext").set_body_typed(DistContext::make);
RAF_REGISTER_GLOBAL("raf.distributed.Global").set_body_typed(DistContext::Global);
RAF_REGISTER_GLOBAL("raf.distributed.EnableDataParallel").set_body_typed(EnableDataParallel);
RAF_REGISTER_GLOBAL("raf.distributed.OverlapCommForward").set_body_typed(OverlapCommForward);
RAF_REGISTER_GLOBAL("raf.distributed.ForceSyncAfterComm").set_body_typed(ForceSyncAfterComm);
RAF_REGISTER_GLOBAL("raf.distributed.ZeroOpt").set_body_typed(ZeroOpt);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalRank").set_body_typed(SetGlobalRank);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalSize").set_body_typed(SetGlobalSize);
RAF_REGISTER_GLOBAL("raf.distributed.SetLocalRankForTuning").set_body_typed(SetLocalRankForTuning);
RAF_REGISTER_GLOBAL("raf.distributed.AutoDPProfilingStartIter")
    .set_body_typed(AutoDPProfilingStartIter);
RAF_REGISTER_GLOBAL("raf.distributed.AutoDPProfilingEndIter")
    .set_body_typed(AutoDPProfilingEndIter);
RAF_REGISTER_GLOBAL("raf.distributed.Barrier").set_body_typed(Barrier);

RAF_REGISTER_OBJECT_REFLECT(DistContextObj);

}  // namespace distributed
}  // namespace raf
