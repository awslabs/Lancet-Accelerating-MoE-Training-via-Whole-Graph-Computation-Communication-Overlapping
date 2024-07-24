/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file data_parallel.cc
 * \brief Data Parallel pass
 */
#include <set>
#include <sstream>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/dist_context.h"
#include "raf/profiler.h"
#include "raf/stream_pool.h"
#include "./common.h"

#ifdef RAF_USE_NCCL
#include "../op/dialect/nccl/communication_utils.h"
#endif

namespace raf {
namespace pass {
namespace data_parallel {

using namespace raf::ir;
using namespace raf::op;
using profiler::Profiler;
using profiler::ProfileStat;
using raf::distributed::DistContext;
using raf::value::NoGradValue;
using stream_pool::StreamTagEnum;

struct DataParallel {
  /* =================================================================
  Description:
      Data Parallel Pass will mainly modify the backward closure currently. The modification is
      1) adding communication op after the op which generate the local gradient.
      2) update the returned gradient from local gradient to aggregated global gradient.
      3) adding a stream_sync op before the end of backward closure to ensure communication is done.
  Example:
        Backward closure before DataParallel Pass:
        ```
        let %closure = fn (%dy) {
            let %x1 = raf.op.nll_loss_dtrue(%y_true, %a4);
            let %x2 = raf.op.nll_loss_dpred(%y_true, %a4);
            %0 = raf.op.get_reduce_axis(%x2, %a3);
            %1 = raf.op.get_kept_dims(%x2, %a3);
            let %x3 = raf.op.sum(%x2, %0, %1);
            %2 = raf.op.get_reduce_axis(%x2, %linear1.b);
            %3 = raf.op.get_kept_dims(%x2, %linear1.b);
            let %x4 = raf.op.sum(%x2, %2, %3);
            let %x5 = raf.op.matmul(%x3, %linear1.w);
            let %x6 = raf.op.matmul_tn(%x3, %a2);
            %4 = raf.op.batch_norm_train_dxwb(%x5, %x, %bn1.w, %bn1.b, -114514);
            let %x7 = %4.0;
            let %x8 = %4.1;
            let %x9 = %4.2;
            let %x10 = (%x7, %x1, %x9, -114514, -114514, %x8, %x4, %x6);
            %x10
        };
        ```
        Backward closure after DataParallel Pass:
        ```
        let %closure = fn (%dy) {
            let %x1 = raf.op.nll_loss_dtrue(%y_true, %a4);
            %0 = (%x1,);
            let %g = raf.op._allreduce(%0, -114514);
            let %x2 = raf.op.nll_loss_dpred(%y_true, %a4);
            %1 = raf.op.get_reduce_axis(%x2, %a3);
            %2 = raf.op.get_kept_dims(%x2, %a3);
            let %x3 = raf.op.sum(%x2, %1, %2);
            %3 = raf.op.get_reduce_axis(%x2, %linear1.b);
            %4 = raf.op.get_kept_dims(%x2, %linear1.b);
            let %x4 = raf.op.sum(%x2, %3, %4);
            %5 = (%x4,);
            let %g1 = raf.op._allreduce(%5, -114514);
            let %x5 = raf.op.matmul(%x3, %linear1.w);
            let %x6 = raf.op.matmul_tn(%x3, %a2);
            %6 = (%x6,);
            let %g2 = raf.op._allreduce(%6, -114514);
            %7 = raf.op.batch_norm_train_dxwb(%x5, %x, %bn1.w, %bn1.b, -114514);
            let %x7 = %7.0;
            %8 = (%x7,);
            let %g3 = raf.op._allreduce(%8, -114514);
            let %x8 = %7.1;
            %9 = (%x8,);
            let %g4 = raf.op._allreduce(%9, -114514);
            let %x9 = %7.2;
            %10 = (%x9,);
            let %g5 = raf.op._allreduce(%10, -114514);
            let %null = raf.op.stream_sync(%g5, -114514);
            let %x10 = (%g3, %g, %g5, -114514, -114514, %g4, %g1, %g2);
            %x10
        };
        ```
        After enabling overlap between communication and forward, the IR could(not must) be:
        Backward closure after DataParallel Pass:
        ```
        let %closure = fn (%dy) {
            let %x1 = raf.op.nll_loss_dtrue(%y_true, %a4);
            %0 = (%x1,);
            let %g = raf.op._allreduce(%0, -114514);
            let %x2 = raf.op.nll_loss_dpred(%y_true, %a4);
            %1 = raf.op.get_reduce_axis(%x2, %a3);
            %2 = raf.op.get_kept_dims(%x2, %a3);
            let %x3 = raf.op.sum(%x2, %1, %2);
            %3 = raf.op.get_reduce_axis(%x2, %linear1.b);
            %4 = raf.op.get_kept_dims(%x2, %linear1.b);
            let %x4 = raf.op.sum(%x2, %3, %4);
            %5 = (%x4,);
            let %g1 = raf.op._allreduce(%5, -114514);
            let %x5 = raf.op.matmul(%x3, %linear1.w);
            let %x6 = raf.op.matmul_tn(%x3, %a2);
            %7 = raf.op.batch_norm_train_dxwb(%x5, %x, %bn1.w, %bn1.b, -114514);
            let %x7 = %7.0;
            let %x8 = %7.1;
            let %x9 = %7.2;
            %8 = (%x9,);
            let %g3 = raf.op._allreduce(%8, -114514);
            %9 = (%x8,);
            let %g4 = raf.op._allreduce(%9, -114514);
            %10 = (%x7,);
            let %g5 = raf.op._allreduce(%10, -114514);
            %11 = (%x6,);
            let %g2 = raf.op._allreduce(%11, -114514);
            let %null = raf.op.stream_sync(%g5, -114514);
            let %x10 = (%g3, %g, %g5, -114514, -114514, %g4, %g1, %g2);
            %x10
        };
        ```
   */
 public:
  explicit DataParallel(const FunctionNode* func, const ir::Array<tvm::Bool>& is_expert_parallel)
      : func(func), fp_ell(ExplicitLetList::make(func->body)), is_expert_parallel(is_expert_parallel) {
  }

  Function Run() {
    auto dctx = DistContext::Global();

    size_t fp_n = fp_ell->vars.size();
    auto closure_expr = fp_ell->exprs.at(fp_n - 2);
    Array<Var> bp_params;
    if (const auto* func = closure_expr.as<FunctionNode>()) {
      bp_ell = ExplicitLetList::make(func->body);
      bp_params = func->params;
    }
    size_t bp_n = bp_ell->vars.size();
    auto bp_grads = bp_ell->exprs.at(bp_n - 1);
    std::set<const VarNode*> gradset;  // All the gradients that returned by backward IR.
    if (const auto* tuple = bp_grads.as<TupleNode>()) {
      if(!is_expert_parallel.empty()) {
        CHECK_EQ(is_expert_parallel.size(), tuple->fields.size());
      }
      for (int param_idx = 0; param_idx < tuple->fields.size(); param_idx++) {
        if(!is_expert_parallel.empty()) {
          if(is_expert_parallel[param_idx]) {
            // is ep param, ignore its synchronization
            continue;
          }
        }
        auto g = tuple->fields[param_idx];
        // check g's type. we only reduce float gradients
        CHECK(g->checked_type().defined());
        if (g->checked_type().as<TensorTypeNode>()) {
          CHECK(g->checked_type().as<TensorTypeNode>()) << "Returned gradient contains a tuple.";
          auto g_dtype = g->checked_type().as<TensorTypeNode>()->dtype;

          if(g_dtype.is_float() || g_dtype.is_bfloat16()) {
            if (const auto* var = g.as<VarNode>()) {
              gradset.insert(var);
            }
          }
        } else {
          CHECK(!g->checked_type().as<TupleTypeNode>()) << "Returned gradient contains a tuple.";
        }
      }
    } else if (const auto* var = bp_grads.as<VarNode>()) {
      gradset.insert(var);
    } else {
      LOG(FATAL) << "Return of backward IR must be Var or tuple of Vars in Data Parallel Pass.";
    }
    if (gradset.empty()) {
      return Function(func->params, fp_ell->AsExpr(), {}, {});
    }
    // The map from original local gradient to aggregated global gradient.
    std::map<raf::ir::Expr, raf::ir::Var> var_var_map;
    // We need to explicitly keep local_vars' order to make the pass deterministic
    std::vector<raf::ir::Expr> local_vars;
    // Enlarge the size of bp_ell to fit the allreduce ops.
    // p1 tracks the processing var/expr (from end to begin)
    // p2 tracks the next vacant position to paste the processing var/expr or allreduce op.
    int p1 = bp_n - 1;
    // If NCCL version is 2.10+, then average allreduce is going to be inserted
    // otherwise, a sum allreduce is used and then a divied op is followed
    // So for NCCL version less than 2.10, the let list size is one more item(i.e. divide op)
    // for each gradient

#if defined RAF_USE_NCCL && NCCL_VERSION_CODE >= 21000
    bp_ell->vars.resize(bp_n + 2 * gradset.size());
    bp_ell->exprs.resize(bp_n + 2 * gradset.size());
    int p2 = bp_n - 1 + 2 * gradset.size();
#else
    bp_ell->vars.resize(bp_n + 3 * gradset.size());
    bp_ell->exprs.resize(bp_n + 3 * gradset.size());
    int p2 = bp_n - 1 + 3 * gradset.size();
#endif

    bp_ell->vars[p2] = bp_ell->vars[p1];
    bp_ell->exprs[p2] = bp_ell->exprs[p1];
    --p2;
    --p1;

    for (int i = p1; i >= 0; --i) {
      if (gradset.find(bp_ell->vars[i].operator->()) != gradset.end()) {
        // If the current expr is an op-expr which generate local gradient,
        // we should add a allreduce op after it.
        static Op op_allreduce = Op::Get("raf.op._allreduce");
        auto input_var = raf::ir::MakeVar("allreduce_in", {});

#if defined RAF_USE_NCCL && NCCL_VERSION_CODE >= 21000
        // Here we name the var as 'g'(global gradient), to help us identify it easier.
        bp_ell->vars[p2 - 1] = input_var;
        bp_ell->exprs[p2 - 1] = Tuple({bp_ell->vars[i]});

        bp_ell->vars[p2] = raf::ir::MakeVar("g", {});
        bp_ell->exprs[p2] = Call(op_allreduce, {bp_ell->vars[p2 - 1], MakeConstant(StringValue::make("avg"))});

        var_var_map.insert({bp_ell->vars[i], bp_ell->vars[p2]});
        local_vars.push_back(bp_ell->vars[i]);
        p2 -= 2;
#else
        static Op op_div = Op::Get("raf.op.divide");
        bp_ell->vars[p2 - 2] = input_var;
        bp_ell->exprs[p2 - 2] = Tuple({bp_ell->vars[i]});

        bp_ell->vars[p2 - 1] = raf::ir::MakeVar("g_sum", {});
        bp_ell->exprs[p2 - 1] =
            Call(op_allreduce, {bp_ell->vars[p2 - 2], MakeConstant(StringValue::make("sum"))});

        bp_ell->vars[p2] = raf::ir::MakeVar("g", {});
        auto tt = bp_ell->vars[i]->checked_type().as<TensorTypeNode>();
        if (tt->dtype.code() == kDLFloat) {
          bp_ell->exprs[p2] = Call(
              op_div, {bp_ell->vars[p2 - 1], MakeConstant(ScalarValue::make(float(dctx->size)))});
        } else if (tt->dtype.code() == kDLInt) {
          bp_ell->exprs[p2] = Call(
              op_div, {bp_ell->vars[p2 - 1], MakeConstant(ScalarValue::make(int64_t(dctx->size)))});
        } else if (tt->dtype.code() == kDLUInt) {
          if(tt->dtype.bits() == 8) {
            bp_ell->exprs[p2] = Call(
                op_div, {bp_ell->vars[p2 - 1], MakeConstant(ScalarValue::make(uint8_t(dctx->size)))});
          } else if (tt->dtype.bits() == 32) {
            bp_ell->exprs[p2] = Call(
                op_div, {bp_ell->vars[p2 - 1], MakeConstant(ScalarValue::make(uint32_t(dctx->size)))});
          } else if (tt->dtype.bits() == 64) {
            bp_ell->exprs[p2] = Call(
                op_div, {bp_ell->vars[p2 - 1], MakeConstant(ScalarValue::make(uint64_t(dctx->size)))});
          } else {
            LOG(FATAL) << "Do not support type KDLInt with bits " << tt->dtype.bits() << ". \n";
          }
        } else {
          LOG(FATAL) << "Do not support type other than KDLFloat and KDLInt. Got dtype code: " << tt->dtype.code() << ". \n";
        }
        var_var_map.insert({bp_ell->vars[i], bp_ell->vars[p2]});
        p2 -= 3;
#endif
      }
      bp_ell->vars[p2] = bp_ell->vars[i];
      bp_ell->exprs[p2] = bp_ell->exprs[i];
      --p2;
    }

    // if (!dctx->overlap_comm_forward && dctx->force_sync_after_comm) {
    if (dctx->force_sync_after_comm) {
      static Op op_sync = Op::Get("raf.op.stream_sync");
      Array<Expr> local_gradients;
      Array<Expr> global_gradients;
      for (auto v : local_vars) {
        local_gradients.push_back(v);
        global_gradients.push_back(var_var_map[v]);
      }
      auto args_x = raf::ir::MakeVar("global_gradients", {});
      bp_ell->vars.insert(--bp_ell->vars.end(), args_x);
      bp_ell->exprs.insert(--bp_ell->exprs.end(), Tuple(global_gradients));

      auto args_stream = MakeConstant(value::ScalarValue::make(StreamTagEnum::CudaCommunicate()));

      auto stream_sync_out = raf::ir::MakeVar("stream_sync_out", {});
      bp_ell->vars.insert(--bp_ell->vars.end(), stream_sync_out);
      bp_ell->exprs.insert(--bp_ell->exprs.end(), Call(op_sync, {args_x, args_stream}));

      for (size_t var_idx = 0; var_idx < local_gradients.size(); var_idx++) {
        auto tgi_var = raf::ir::MakeVar("sync_tgi", {});
        auto tgi_expr = TupleGetItem(stream_sync_out, var_idx);
        bp_ell->vars.insert(--bp_ell->vars.end(), tgi_var);
        bp_ell->exprs.insert(--bp_ell->exprs.end(), tgi_expr);
        var_var_map[local_gradients[var_idx]] = tgi_var;
      }

    }
    dctx->iteration++;

    Array<Expr> new_bp_rt;
    if (const auto* tuple = bp_grads.as<TupleNode>()) {
      for (int i = 0; i < tuple->fields.size(); ++i) {
        if (tuple->fields[i]->IsInstance<VarNode>() && var_var_map.count(tuple->fields[i])) {
          auto it = var_var_map.find(tuple->fields[i]);
          new_bp_rt.push_back(it->second);
        } else {
          new_bp_rt.push_back(tuple->fields[i]);
        }
      }
    } else if (bp_grads->IsInstance<VarNode>()) {
      auto it = var_var_map.find(bp_grads);
      new_bp_rt.push_back(it->second);
    } else {
      LOG(FATAL) << "Return of backward IR must be Var or tuple of Vars in Data Parallel Pass.";
    }

    bp_n = bp_ell->vars.size();
    if (new_bp_rt.size() == 1) {
      bp_ell->exprs[bp_n - 1] = new_bp_rt[0];
    } else {
      bp_ell->exprs[bp_n - 1] = Tuple(new_bp_rt);
    }

    fp_ell->exprs[fp_n - 2] = Function(bp_params, bp_ell->AsExpr(), {}, {});
    return Function(func->params, fp_ell->AsExpr(), {}, {});
  }

 private:
  // initialized in constructor
  const FunctionNode* func;
  std::unique_ptr<ExplicitLetList> fp_ell{nullptr};
  // initialized in Run
  std::unique_ptr<ExplicitLetList> bp_ell{nullptr};
  // The comminication operators whose profiling will be collected for scheduling.
  const std::set<std::string> scheduled_communication_ops = {"raf.op._allreduce"};
  // The global gradient set
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> global_grad;
  // whether each param is a expert parallel param
  const ir::Array<tvm::Bool>& is_expert_parallel;
};

}  // namespace data_parallel

Pass AutoDataParallel(ir::Array<tvm::Bool> is_expert_parallel) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return data_parallel::DataParallel(f.operator->(), is_expert_parallel).Run();
  };
  return CreateRAFFunctionPass(pass_func, 0, "AutoDataParallel", {"InferType"});
}

RAF_REGISTER_GLOBAL("raf.pass_.AutoDataParallel").set_body_typed(AutoDataParallel);

}  // namespace pass
}  // namespace raf
