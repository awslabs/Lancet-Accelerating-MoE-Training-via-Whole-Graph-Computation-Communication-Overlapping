/*!
 * Copyright (c) 2022 by Contributors
 * \file extended_op_profiler.h
 * \brief Wrapper of op profiler for getting the execution time for partitioned operartors.
 */
#pragma once
#include "raf/ir_ext.h"
#include "raf/op_profiler.h"
#include "raf/pass.h"
#include "./profile_utils.h"
#include "./scheduler_common.h"
#include "./partition_common.h"

namespace raf {
namespace pass {
namespace extended_op_profiler {

using namespace raf::pass;
using namespace scheduler_common;
using op_profiler::OpProfiler;
using op_profiler::CUDAOpProfiler;

inline Expr UpdateFunctionCall(Expr expr) {
    // Convert the function into GNF,
    // also try to infer the function type.
    CHECK(expr->IsInstance<CallNode>());
    expr = InferType(expr);
    Call call = Downcast<Call>(expr);
    CHECK(call->op->IsInstance<FunctionNode>());
    Function func = Downcast<Function>(call->op);
    CHECK(func->params.size() == call->args.size());
    tvm::Map<Var, Expr> args_map = {};
    Array<Var> vars = {};
    Array<Expr> args_ = {};
    Array<Type> arg_type = {};
    for (int i = 0; i < func->params.size(); ++i) {
        args_map.Set(func->params[i], call->args[i]);
        if (call->args[i]->IsInstance<VarNode>()) {
            // Keep only var args. Some constant args are needed
            // when inferring the type of function body.
            vars.push_back(Downcast<Var>(call->args[i]));
            args_.push_back(call->args[i]);
            arg_type.push_back(call->args[i]->checked_type_);
        }
    }
    // Substitute the func body with all input args.
    auto body = Substitute(func->body, args_map);
    auto func_ = Function(vars, body, func->ret_type, func->type_params, func->attrs);
    auto func_type = FuncType(arg_type, func->ret_type, {}, {});
    func_->checked_type_ = func_type;
    auto mod = IRModule::FromExpr(func_);
    mod = ToGraphNormalForm()(mod);
    mod = InferType()(mod);
    func = Downcast<Function>(mod->Lookup("main"));
    func->checked_type_ = func_->checked_type_;
    Call call_ = Call(func, args_, call->attrs, {});
    call_->checked_type_ = call->checked_type_;
    return call_;
}

class ExtendedOpProfiler {
public:
    ExtendedOpProfiler(const Device& device, const CommCostModel& fcost): 
        profiler_(CUDAOpProfiler::Get(device)), device_(device), fcost_(fcost) {
        Setup_();
    }

    virtual std::pair<SimulateTimeType, std::chrono::milliseconds::rep> GetCompOpExecTime(const Expr& op) {
        if(!op->IsInstance<CallNode>()) {
            return {0, 0};
        }
        auto start = std::chrono::system_clock::now();
        Expr expr = op;
        Call call = Downcast<Call>(expr);
        if (call->op->IsInstance<FunctionNode>()) {
            expr = UpdateFunctionCall(expr);
        }
        Call updated_call = Downcast<Call>(expr);
        // check if op has constant tensor with the wrong device
        Array<Expr> new_args;
        for (auto arg: updated_call->args) {
            if (auto const_node = arg.as<ConstantNode>()) {
                if (const_node->IsTensor()) {
                    auto device = const_node->data->device;
                    if (device.device_type != device_.device_type() || device.device_id != device_.device_id()) {
                        // make fake tensor
                        auto tv = CreateDummyValueFromType(const_node->checked_type(), device_);
                        auto new_const = MakeConstant(tv);
                        new_const->checked_type_ = const_node->checked_type_;
                        new_args.push_back(new_const);
                        continue;
                    }
                }
            }
            new_args.push_back(arg);
        }
        Expr expr_to_profile = Call(updated_call->op, new_args, updated_call->attrs, updated_call->type_args);
        expr_to_profile->checked_type_ = updated_call->checked_type_;
        std::vector<float> profiled_times;
        std::tie(profiled_times, std::ignore) = profiler_->ProfileOp(expr_to_profile, warmup_, exec_number_);
        SimulateTimeType result = 0;
        for(auto time: profiled_times) {
            result += time;
        }
        result /= profiled_times.size();
        auto end = std::chrono::system_clock::now();
        auto elapsed_time = 
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return {result, elapsed_time};
    }

    virtual SimulateTimeType GetCommOpExecTime(const CommComponents& size) {
        return fcost_(size);
    }

private:
    void Setup_() {
        warmup_ = 10;
        exec_number_ = 10;
        if (const char* profile_setting_str = getenv("PROFILE_SETTING")) {
            std::string params_str(profile_setting_str);
            LOG(INFO) << "Profile setting " << params_str;
            auto warmup_str = params_str.substr(0, params_str.find(";"));
            auto exec_number_str = params_str.substr(params_str.find(";") + 1, std::string::npos);
            warmup_ = std::stod(warmup_str);
            exec_number_ = std::stod(exec_number_str);
        }
        LOG(INFO) << "Profiler initialized on device " << device_;
    }

    OpProfiler* profiler_;
    CommCostModel fcost_;
    const Device device_;
    int warmup_;
    int exec_number_;
};

class DummyExtendedOpProfiler: public ExtendedOpProfiler {
public:
    DummyExtendedOpProfiler() : ExtendedOpProfiler(Device::Current(/*allow_default=*/false), [](const CommComponents& any) {return 0.0;}) {}

    std::pair<SimulateTimeType, std::chrono::milliseconds::rep> GetCompOpExecTime(const Expr& op) override {
        return {0, 0};
    }

    SimulateTimeType GetCommOpExecTime(const CommComponents& size) override {
        return 0;
    }
};

} // namespace extended_op_profiler
} // namespace pass
} // namespace raf