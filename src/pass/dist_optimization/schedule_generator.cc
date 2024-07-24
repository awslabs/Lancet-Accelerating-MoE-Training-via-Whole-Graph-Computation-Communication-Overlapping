/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file schedule_generator.cc
 * \brief Schedule generator.
 */

#include "schedule_generator.h"

namespace raf {
namespace pass {
namespace schedule_generator {

TVM_REGISTER_NODE_TYPE(DynamicScheduleParamsNode);

DynamicScheduleParams DynamicScheduleParams::make(double lambda_comp, double lambda_comm, double gamma, double theta_comp, double theta_comm, double beta) {
  ObjectPtr<DynamicScheduleParamsNode> n = make_object<DynamicScheduleParamsNode>();
  n->lambda_comp = lambda_comp;
  n->lambda_comm = lambda_comm;
  n->gamma = gamma;
  n->theta_comp = theta_comp;
  n->theta_comm = theta_comm;
  n->beta = beta;
  return DynamicScheduleParams(n);
}

DynamicScheduleParams::DynamicScheduleParams(double lambda_comp, double lambda_comm, double gamma, double theta_comp, double theta_comm, double beta) {
  ObjectPtr<DynamicScheduleParamsNode> n = make_object<DynamicScheduleParamsNode>();
  n->lambda_comp = lambda_comp;
  n->lambda_comm = lambda_comm;
  n->gamma = gamma;
  n->theta_comp = theta_comp;
  n->theta_comm = theta_comm;
  n->beta = beta;
  data_ = std::move(n);
}

static std::function<double(double, double, double, double, double, double)> *eval_schedule_ = nullptr;

void SetScheduleEvalFunc(std::function<double(double, double, double, double, double, double)> *func) {
    eval_schedule_ = func;
}

double DynamicEvalCurrentSchedule_(double lamb_comp, double lamb_comm, double gamma, double theta_comp, double theta_comm, double beta) {
    CHECK(eval_schedule_ != nullptr) << "No valid current dfg to evaluate. This function should never be called by users.";
    return (*eval_schedule_)(lamb_comp, lamb_comm, gamma, theta_comp, theta_comm, beta);
}


}  // namespace schedule_generator
}  // namespace pass
}  // namespace raf