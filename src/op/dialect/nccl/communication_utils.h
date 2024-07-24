/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/communication_utils.h
 * \brief Helper functions for communicaton operators
 */
#pragma once
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <nccl.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include "raf/device.h"
#include "raf/op.h"
#include "raf/enum_base.h"
#include "raf/ir.h"
#include "raf/value.h"
#include "raf/tensor.h"
#include "raf/communicator.h"
#include "raf/stream_pool.h"
#include "../../../common/cuda_utils.h"
#include "../../../common/shape_utils.h"

#define NCCL_CALL(cmd)                                                          \
  do {                                                                          \
    ncclResult_t e = cmd;                                                       \
    if (e != ncclSuccess) {                                                     \
      LOG(FATAL) << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << " " \
                 << ncclGetErrorString(e);                                      \
    }                                                                           \
  } while (0)

namespace raf {

template <>
inline DType::operator ncclDataType_t() const {
  switch (code) {
    case kDLInt:
      if (bits == 8) return ncclInt8;
      if (bits == 32) return ncclInt32;
      if (bits == 64) return ncclInt64;
      break;
    case kDLUInt:
      if (bits == 8) return ncclUint8;
      if (bits == 32) return ncclUint32;
      break;
    case kDLFloat:
      if (bits == 16) return ncclFloat16;
      if (bits == 32) return ncclFloat32;
      if (bits == 64) return ncclFloat64;
  }
  LOG(FATAL) << "NotImplementedError: " << c_str();
  throw;
}

}  // namespace raf
