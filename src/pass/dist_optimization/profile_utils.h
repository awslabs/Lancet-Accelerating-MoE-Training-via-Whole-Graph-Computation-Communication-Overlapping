/*!
 * Copyright (c) 2022 by Contributors
 * \file profile_utils.h
 * \brief Utils for profiling operators.
 */
#pragma once
#include <unordered_map>
#include "raf/ir.h"
#include "raf/op.h"
#include "raf/executor.h"
#include "raf/profiler.h"
#include "raf/device.h"
#include "raf/value.h"
#include "./scheduler_utils.h"

namespace raf {
namespace pass {
namespace profile_utils {

using namespace raf::analysis;
using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::profiler;
using namespace raf::executor::interpreter;
using raf::pass::scheduler_utils::NodeMap;
using raf::pass::scheduler_utils::NodeSet;
using raf::pass::scheduler_utils::VarMap;
using raf::pass::scheduler_utils::Vars;
using raf::pass::scheduler_utils::ExprMap;
using raf::pass::scheduler_utils::Exprs;
using raf::pass::scheduler_utils::Node;

using ProfileTimeType = uint64_t;
using OpKey = std::string;
using OpKeyProfiledTimeMap = std::unordered_map<OpKey, ProfileTimeType>;

auto tensor_repr = [](std::ostringstream& os, const TensorValueObj* tensor) {
  const DLTensor* t = tensor->tensor.operator->();
  os << "T<";
  for (int i = 0; i < t->ndim; ++i) {
    os << t->shape[i] << "x";
  }
  switch (t->dtype.code) {
    case kDLInt: {
      os << "i" << static_cast<int>(t->dtype.bits);
      break;
    }
    case kDLUInt: {
      os << "u" << static_cast<int>(t->dtype.bits);
      break;
    }
    case kDLFloat: {
      os << "f" << static_cast<int>(t->dtype.bits);
      break;
    }
    case kDLBfloat: {
      os << "bf" << static_cast<int>(t->dtype.bits);
      break;
    }
    default: {
      os << "unk";
      break;
    }
  }
  if (t->dtype.lanes > 1) {
    os << "x" << static_cast<int>(t->dtype.lanes);
  }
  os << ">";
};

inline std::string TESTGenerateArgString(Call& op, Device device) {
  // copied from vm.cc
  // extract the input args and prepare the hash key to query op env
  std::ostringstream os;
  for (int i = 0; i < op->args.size(); i++) {
    auto arg = op->args[i];
    if (arg.as<ConstantNode>()) {
      // Skip constatnts in the hash key
      continue;
    }
    auto arg_value = CreateDummyValueFromType(arg->checked_type(), device);
    if (auto tensor = arg_value.as<TensorValueObj>()) {
      tensor_repr(os, tensor);
    } else if (auto tup = arg_value.as<TupleValueObj>()) {
      os << "(";
      for (auto field : tup->fields) {
        auto t = field.as<TensorValueObj>();
        if (t != nullptr) {
          tensor_repr(os, t);
        }
        os << ",";
      }
      os << ")";
    } else {
      LOG(FATAL) << "Unsupported non-const arg type: " << arg->GetTypeKey();
    }
    os << ",";
  }

  // extract the output
  os << "|";
  if (auto out_tuple_type = op->checked_type().as<TupleTypeNode>()) {
    if (out_tuple_type->fields.size() == 1) {
      auto out_value = CreateDummyValueFromType(out_tuple_type->fields[0], device);
      tensor_repr(os, out_value.as<TensorValueObj>());
    } else {
      os << "(";
      Array<Value> outs;
      for (int i = 0; i < out_tuple_type->fields.size(); i++) {
        auto val = CreateDummyValueFromType(out_tuple_type->fields[i], device);
        tensor_repr(os, val.as<TensorValueObj>());
        os << ",";
      }
      os << ")";
    }
  } else {
    auto out_value = CreateDummyValueFromType(op->checked_type(), device);
    tensor_repr(os, out_value.as<TensorValueObj>());
  }
  return os.str();
}

inline std::string GetExprName_(Expr expr) {
  if (expr->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr);
    if (call->op->IsInstance<OpNode>()) {
      std::stringstream ss;
      ss << call->op;
      return ss.str();
    } else {
      return "func";
    }
  } else if (expr->IsInstance<TupleNode>()) {
    return "tuple";
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    return "tgi";
  } else {
    LOG(WARNING) << "Not supported expr: " << expr->GetTypeKey();
    return "unknown";
  }
}

inline std::string GenerateArgString(Call& op, Device device) {
  return GetExprName_(op) + "|" + TESTGenerateArgString(op, device);
}

inline std::string GenerateArgString(Tuple& tuple, Device device) {
  std::ostringstream os;
  os << "tuple|";
  auto tuple_type = tuple->checked_type().as<TupleTypeNode>();
  if (tuple_type->fields.size() == 1) {
    auto value = CreateDummyValueFromType(tuple_type->fields[0], device);
    tensor_repr(os, value.as<TensorValueObj>());
  } else {
    os << "(";
    Array<Value> outs;
    for (int i = 0; i < tuple_type->fields.size(); i++) {
      auto val = CreateDummyValueFromType(tuple_type->fields[i], device);
      tensor_repr(os, val.as<TensorValueObj>());
      os << ",";
    }
    os << ")";
  }
  return os.str();
}

inline std::string GenerateArgString(TupleGetItem& tgi, Device device) {
  std::ostringstream os;
  os << "tgi|" << tgi->index << "|";
  auto tuple_type = tgi->tuple->checked_type().as<TupleTypeNode>();
  if (tuple_type->fields.size() == 1) {
    auto value = CreateDummyValueFromType(tuple_type->fields[0], device);
    tensor_repr(os, value.as<TensorValueObj>());
  } else {
    os << "(";
    Array<Value> outs;
    for (int i = 0; i < tuple_type->fields.size(); i++) {
      auto val = CreateDummyValueFromType(tuple_type->fields[i], device);
      tensor_repr(os, val.as<TensorValueObj>());
      os << ",";
    }
    os << ")";
  }
  return os.str();
}

inline Device GetCurrentDevice(bool is_simulation) {
  auto device = Device::Current(/*allow_default=*/true);
  if(is_simulation) {
    // if there is a current device, use it
    if (device.device_type() == DevType::kCUDA()) {
      return device;
    }
    // if it is a cpu device, since no actual kernel will be run, 
    // it doesn't matter too much which gpu is used
    return Device(DevType::kCUDA(), 0);
  } else {
    CHECK_NE(device.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
    return device;
  }
}

inline std::string GenerateExprString(Expr& expr) {
  static auto device = GetCurrentDevice(true);
  if (expr->IsInstance<LetNode>()) {
    Expr value = Downcast<Let>(expr)->value;
    return GenerateExprString(value);
  } else if (expr->IsInstance<CallNode>()) {
    Call call = Downcast<Call>(expr);
    return GenerateArgString(call, device);
  } else if (expr->IsInstance<TupleNode>()) {
    Tuple tuple = Downcast<Tuple>(expr);
    return GenerateArgString(tuple, device);
  } else if (expr->IsInstance<TupleGetItemNode>()) {
    TupleGetItem tgi = Downcast<TupleGetItem>(expr);
    return GenerateArgString(tgi, device);
  } else {
    LOG(FATAL) << "Unsupported expr: " << expr->GetTypeKey();
    return "";
  }
}

inline OpKey GenerateOpKeyFromProfileStats(const ProfileStat& stat) {
  auto orig_name = op::GetOrigName(stat.name_);
  // LOG(INFO) << "stat arg string: " << stat.args_string << ".";
  return orig_name + ";" + stat.args_string;
}

inline OpKey GenerateOpKeyFromCallOp(Call op, 
                                     OpKeyProfiledTimeMap& profile_time_map,
                                     bool is_simulation=false) { 
  auto device = GetCurrentDevice(is_simulation);
  op::CallValues call_values;
  call_values = op::CreateDummyCallValues(op, device, /*include_args=*/false);
  // auto op_env = op::Dispatch(call_values);
  auto possible_names = op::GetPossibleDispatchedName(call_values);
  CHECK_GT(possible_names.size(), 0);
  auto arg_str = TESTGenerateArgString(op, device);
  for(auto env_name: possible_names) {
    auto orig_name = op::GetOrigName(env_name);
    auto candidate_key = orig_name + ";" + arg_str;
    if(profile_time_map.count(candidate_key)) {
      return candidate_key;
    }
  }
  // cannot find any candidate key. return the first in possible names
  return op::GetOrigName(possible_names[0]) + ";" + arg_str;
}

inline uint64_t GetExecTime(OpKeyProfiledTimeMap& profile_time_map, Call& op) {
  OpKey key = GenerateOpKeyFromCallOp(op, profile_time_map, false);
  if (profile_time_map.count(key)) {
    return profile_time_map[key];
  } else {
    return 0;
  }
}

inline OpKeyProfiledTimeMap 
GetProfiledTime(const std::vector<ProfileStat>& profile_stats) {
  std::unordered_map<OpKey, std::vector<uint64_t>> profiled_time_vector_map = {};
  for(auto& stat: profile_stats) {
    uint64_t start_time, end_time;
    bool start_time_set = false, end_time_set = false;
    for (size_t i = 0; i < sizeof(stat.items_) / sizeof(stat.items_[0]); ++i) {
      if (stat.items_[i].enabled_) {
        if(stat.items_[i].event_type_ == EventType::kDurationBegin) {
          start_time = stat.items_[i].timestamp_;
          start_time_set = true;
        } else if (stat.items_[i].event_type_ == EventType::kDurationEnd) {
          end_time = stat.items_[i].timestamp_;
          end_time_set = true;
        }
      }
    }
    if (start_time_set && end_time_set) {
      OpKey key = GenerateOpKeyFromProfileStats(stat);
      CHECK_LE(start_time, end_time);
      profiled_time_vector_map[key].push_back(end_time - start_time);
    }
  }
  OpKeyProfiledTimeMap profile_time_map = {};
  for (auto& it : profiled_time_vector_map) {
    auto& time_vec = it.second;
    uint64_t total_time = 0;
    for (auto time : time_vec) {
      total_time += time;
    }
    profile_time_map[it.first] = ((ProfileTimeType) total_time) / time_vec.size();
  }
  return profile_time_map;
}

// <warm_up, repeat>
using ProfileArgs = std::pair<int, int>;

inline std::vector<ProfileStat> 
ProfileExpr(const Expr& e, ProfileArgs profile_args) {
  int prof_level = Profiler::Get()->profile_level();
  Profiler::Get()->set_profile_level(0);
  for (int i = 0; i < profile_args.first; ++i) {
    Interpret(e);
  }
  Profiler::Get()->set_profile_level(1);
  for (int i = 0; i < profile_args.second; ++i) {
    Interpret(e);
  }
  Profiler::Get()->set_profile_level(prof_level);
  return Profiler::Get()->GetProfileStats();
}

inline ExprMap<ProfileTimeType>
GetExprProfiledTimeMap(OpKeyProfiledTimeMap& profiled_time_map,
                       const std::vector<Expr>& calls) {
  ExprMap<ProfileTimeType> expr_profiled_time_map = {};
  for (auto call_expr : calls) {
    Call call = Downcast<Call>(call_expr);
    OpKey key = GenerateOpKeyFromCallOp(call, profiled_time_map);
    expr_profiled_time_map[call_expr] = profiled_time_map[key];
  }
  return expr_profiled_time_map;
}

inline std::pair<VarMap<ProfileTimeType>, ProfileTimeType>
ProfileExpr(const Expr& e,
            const VarMap<Exprs>& var_exprs_map,
            const std::vector<Expr>& overhead_exprs,
            ProfileArgs profile_args) {
  auto profile_stats = ProfileExpr(e, profile_args);
  auto profiled_time_map = GetProfiledTime(profile_stats);
  std::vector<Expr> calls = {};
  for (auto& var_and_exprs : var_exprs_map) {
    auto& exprs = var_and_exprs.second;
    calls.insert(calls.end(), exprs.begin(), exprs.end());
  }
  ExprMap<ProfileTimeType> expr_time_map = GetExprProfiledTimeMap(profiled_time_map, calls);
  VarMap<ProfileTimeType> var_time_map = {};
  for (auto& var_and_exprs : var_exprs_map) {
    Var var = var_and_exprs.first;
    auto& exprs = var_and_exprs.second;
    ProfileTimeType profiled_time = 0;
    for (auto expr : exprs) {
      profiled_time += expr_time_map[expr];
    }
    profiled_time /= exprs.size();
    var_time_map[var] = profiled_time;
  }
  ProfileTimeType overhead_time = 0;
  for (auto& overhead_expr : overhead_exprs) {
    overhead_time += expr_time_map[overhead_expr];
  }
  return std::make_pair(var_time_map, overhead_time);
}

} // namespace profile_utils
} // namespace pass
} // namespace raf