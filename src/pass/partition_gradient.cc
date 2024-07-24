/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file partition_gradient.cc
 * \brief Given a model after AutoDiff and InlineBackward, this pass performs the following:
 * ZeRO-1: Partition the gradients outputed by the given model, so that the later wrapped
 *         optimizer can have a partitioend optimizer status. Note that optimizers must
 *         consider gradient partitioning if applied; otherwise the result will be incorrect.
 * ZeRO-2: Replace the allreduce inserted by AutoDataParallel with reduce_scatter to obtain only
 *         a partition of gradients.
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "raf/pass.h"
#include "raf/ir_ext.h"
#include "raf/op_utils.h"
#include "./common.h"
#include "./let_list.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {
namespace partition_gradient {

class GradientPartitioner : public ExprMutator {
 public:
  GradientPartitioner(int opt_level, int n_part, const Function& func)
      : opt_level_(opt_level), n_part_(n_part), func_(func) {
    // Build the var to expr map for the ANF.
    Map<Var, Expr> var_to_expr;
    auto ell = ExplicitLetList::make(func->body);
    for (size_t i = 0; i < ell->vars.size(); ++i) {
      var_to_expr.Set(ell->vars[i], ell->exprs[i]);
      if (auto call_node = ell->exprs[i].as<CallNode>()) {
        // try to descend into closure
        if (call_node->op->IsInstance<VarNode>()) {
          if (auto func_node = var_to_expr[Downcast<Var>(call_node->op)].as<FunctionNode>()) {
            auto func_ell = ExplicitLetList::make(func_node->body);
            for (size_t j = 0; j < func_ell->vars.size(); ++j) {
              var_to_expr.Set(func_ell->vars[j], func_ell->exprs[j]);
            }
          }
        }
      }
    }

    // Assume output is a tuple of (forward out, (grads, ...))
    auto ret = ell->exprs.back().as<TupleNode>();
    CHECK(ret != nullptr) << "Expected a tuple output, but got " << ell->exprs.back()->GetTypeKey();
    CHECK_EQ(ret->fields.size(), 2U)
        << "Expected the output tuple to be (out, (grad, ...)) with 2 fields, but it has "
        << ret->fields.size() << " fields";

    // Traverse back to find the gradient tuple.
    grad_tuple_var_ = Downcast<Var>(ret->fields[1]);
    Expr grads = var_to_expr[grad_tuple_var_];
    while (!grads->IsInstance<TupleNode>()) {
      if(auto grads_callnode = grads.as<CallNode>()) {
        if(auto grads_funcnode = var_to_expr[Downcast<Var>(grads_callnode->op)].as<FunctionNode>()) {
          // encountered a unapplied backward closure, set grads to the return of that closure
          auto ell_closure = ExplicitLetList::make(grads_funcnode->body);
          CHECK(ell_closure->exprs.back().as<TupleNode>()) << 
            "Expected a tuple as closure output, but got " << ell_closure->exprs.back()->GetTypeKey();
          grad_tuple_var_ = ell_closure->vars.back();
          grads = Downcast<Tuple>(ell_closure->exprs.back());
          break;
        }
      }
      auto tgi = grads.as<TupleGetItemNode>();
      CHECK(tgi != nullptr) << "Expected TupleGetItem, but got " << grads->GetTypeKey();
      auto tuple = Downcast<Tuple>(var_to_expr[Downcast<Var>(tgi->tuple)]);
      grad_tuple_var_ = Downcast<Var>(tuple->fields[tgi->index]);
      grads = var_to_expr[grad_tuple_var_];
    }

    static const Op& divide_op = Op::Get("raf.op.divide");
    for (auto field : Downcast<Tuple>(grads)->fields) {
      if (field->IsInstance<VarNode>()) {
        auto field_expr = var_to_expr[Downcast<Var>(field)];
        if (field_expr->IsInstance<CallNode>()) {
          auto field_call = Downcast<Call>(field_expr);
          if (!IsAllReduceCall(field_expr) && field_call->op != divide_op) {
            continue;
          }
        }
      }
      if (field->IsInstance<VarNode>()) {
        Var field_var = Downcast<Var>(field);
        auto type = Downcast<TensorType>(field_var->checked_type());
        auto dtype = type->dtype;
        if (!dtype.is_float() && !dtype.is_bfloat16() || type->shape.size() == 0) {
          // skip non float field.
          continue;
        }
        grads_.Set(field_var, Expr());
        Var ar_var;
        if (auto field_as_tgi = var_to_expr[field_var].as<TupleGetItemNode>()) {
          ar_var = Downcast<Var>(field_as_tgi->tuple);
          allreduce_tgis_.Set(field_var, var_to_expr[field_var]);
          auto ar_call = var_to_expr.at(ar_var).as<CallNode>();
          CHECK(ar_call);
          allreduces_.Set(ar_var, ar_call->args[0]);
        } else {
          auto expr_call = var_to_expr[field_var].as<CallNode>();
          CHECK(expr_call);
          if(expr_call->op == divide_op) {
            // encountered divide call
            auto tgi_or_ar_var = Downcast<Var>(expr_call->args[0]);
            if(var_to_expr.at(tgi_or_ar_var)->IsInstance<TupleGetItemNode>()) {
              // allreduce -> tgi -> divide
              auto ar_tgi_expr = Downcast<TupleGetItem>(var_to_expr.at(tgi_or_ar_var));
              allreduce_tgis_.Set(tgi_or_ar_var, ar_tgi_expr);
              auto ar_call_var = Downcast<Var>(ar_tgi_expr->tuple);
              CHECK(var_to_expr.at(ar_call_var)->IsInstance<CallNode>());
              auto ar_call_expr = Downcast<Call>(var_to_expr.at(ar_call_var));
              CHECK(IsAllReduceCall(ar_call_expr));
              allreduces_.Set(ar_call_var, ar_call_expr->args[0]);
            } else {
              // allreduce -> divide
              CHECK(var_to_expr.at(tgi_or_ar_var)->IsInstance<CallNode>());
              auto ar_call_expr = Downcast<Call>(var_to_expr.at(tgi_or_ar_var));
              CHECK(IsAllReduceCall(ar_call_expr));
              allreduces_.Set(tgi_or_ar_var, ar_call_expr->args[0]);
            }
          } else {
            // direct allreduce call, do nothing
          }
        }
      }
    }

    scopes_.emplace_back(new LetList);
  }

  /*! \brief Partition the parameters according to the parameter group. */
  Function Partition(int rank) {
    if (grads_.empty()) {  // No gradients to be partitioned.
      return func_;
    }

    rank_ = rank;
    Array<Var> func_params{func_->params};
    auto new_body = this->Mutate(func_->body);
    return Function(func_params, new_body, func_->ret_type, {}, func_->attrs);
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      auto curr_var = node->var;
      auto value = VisitExpr(node->value);
      var_to_expr_.Set(curr_var, value);
      if (grads_.count(curr_var) > 0) {
        // The curr_var is a complete gradient.
        CHECK(!grads_[curr_var].defined());
        auto grad_var = SliceGrad(scope, curr_var, value, opt_level_);
        grads_.Set(curr_var, grad_var);
      } else if (curr_var == grad_tuple_var_) {
        // Replace gradients with sliced ones.
        // If grad is not sliced, then use original fields.
        Array<Expr> fields;
        for (auto field : Downcast<Tuple>(value)->fields) {
          if (auto var_node = field.as<VarNode>()) {
            auto var = GetRef<Var>(var_node);
            if (grads_.count(var) > 0) {
              CHECK(grads_[var].defined())
                  << "Internal error: gradient " << var << " does not map to the sliced one";
              fields.push_back(grads_[var]);
            } else {
              fields.push_back(field);
            }
          } else {
            fields.push_back(field);
          }
        }
        scope->Push(curr_var, Tuple(fields));
      } else if (opt_level_ > 1 && (allreduces_.count(curr_var) || allreduce_tgis_.count(curr_var))) {
        // do nothing
      } else {
        scope->Push(curr_var, value);
      }

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto ret = scopes_.back()->Get(this->Mutate(body));
    scopes_.pop_back();
    return ret;
  }

  /*!\ brief Log identified parameters for debugging. */
  std::string DebugDumpOutParams() {
    std::stringstream ss;
    for (const auto& kv : grads_) {
      ss << kv.first << "(share: " << kv.second << ")" << std::endl;
    }
    return ss.str();
  }

 private:
  /*! \brief Check whether a given expression is a call expression with all_reduce. */
  inline bool IsAllReduceCall(const Expr& expr) {
    static const Op& allreduce_op = Op::Get("raf.op._allreduce");
    if (!expr->IsInstance<CallNode>()) {
      return false;
    }
    auto call = Downcast<Call>(expr);
    if (auto node = call->op.as<OpNode>()) {
      return GetRef<Op>(node) == allreduce_op;
    }
    return false;
  }

  /*! \brief Detect allreduce expr or allreduce followed by divide due to use NCCL version<2.10.
   *  return allreduce expr and divied expr if the pattern match otherwise NullValue*/
  inline std::tuple<Var, Var, Var> GetAllReduceVar(const Var& var) {
    // there are three possiblities for expr:
    // 1. Expr is a allreduce call (master branch). In this case the first element of the tuple is set to
    //    the allreduce call, while others are not defined.
    // 2. Expr is a tgi to allreduce output (schedule branch). In this case only the first and the second
    //    elements of the tuple are set.
    // 3. Expr is a divide call (master branch). In this case there is a chain allreduce -> divide. Only
    //    the first and the third elements of the tuple are set.
    // 4. Expr is a divide call (schedule branch). In this case there exists a allreduce -> tgi -> divide
    //    chain. The elements of the return tuple are set to allreduce, tgi and divide respectively.
    CHECK(var_to_expr_.count(var));
    Expr expr = var_to_expr_.at(var);
    Var allreduce_var = Var();
    Var allreduce_tgi_var = Var();
    Var divide_var = Var();

    if (expr->IsInstance<CallNode>()) {
      static const Op& divide_op = Op::Get("raf.op.divide");
      auto call_node = expr.as<CallNode>();
      auto node = call_node->op.as<OpNode>();
      if (IsAllReduceCall(expr)) {
        // case 1
        allreduce_var = var;
      } else if (node && GetRef<Op>(node) == divide_op) {
        // case 3 or case 4
        Var ar_or_tgi_var = Downcast<Var>(call_node->args[0]);
        CHECK(var_to_expr_.count(ar_or_tgi_var));
        if (IsAllReduceCall(var_to_expr_[ar_or_tgi_var])) {
          // case 3
          allreduce_var = ar_or_tgi_var;
          divide_var = var;
        } else {
          // case 4
          CHECK(var_to_expr_[ar_or_tgi_var]->IsInstance<TupleGetItemNode>());
          auto ar_var = Downcast<Var>(var_to_expr_[ar_or_tgi_var].as<TupleGetItemNode>()->tuple);
          CHECK(var_to_expr_.count(ar_var) && IsAllReduceCall(var_to_expr_[ar_var]));
          allreduce_var = ar_var;
          allreduce_tgi_var = ar_or_tgi_var;
          divide_var = var;
        }
      }
    } else if (expr->IsInstance<TupleGetItemNode>()) {
      // case 2
      auto ar_var = Downcast<Var>(expr.as<TupleGetItemNode>()->tuple);
      CHECK(var_to_expr_.count(ar_var) && IsAllReduceCall(var_to_expr_[ar_var]));
      allreduce_var = ar_var;
      allreduce_tgi_var = var;
    }
    return std::make_tuple(allreduce_var, allreduce_tgi_var, divide_var);
  }

  /*! \brief Return the n'th argument of the given call expr. */
  inline Expr GetNArg(const Expr& expr, int n) {
    CHECK(expr->IsInstance<CallNode>());
    auto call = Downcast<Call>(expr);
    CHECK_GE(call->args.size(), n)
        << "Expected at least " << n << " argument, but got " << raf::ir::AsText(expr);
    return call->args[n];
  }

  std::pair<Expr, Expr> GetReduceScatterArgs(const Var& var) {
    auto ttype = var->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr) << "Expected a tesnor, but got " << var->checked_type();
    auto shape_expr = ttype->shape[0];
    if (!shape_expr->IsInstance<IntImmNode>()) {
      LOG(FATAL) << "Do not support dynamic shape yet";
      throw;
    }
    std::vector<int64_t> shapes;
    for (auto axis : ttype->shape) {
      auto node = axis.as<IntImmNode>();
      CHECK(node != nullptr) << "Axis " << axis << " is not IntImmNode";
      int64_t axis_ = node->value;
      shapes.push_back(axis_);
    }
    int64_t dim0_length = shapes[0];
    if (dim0_length % n_part_ != 0) {
      shapes[0] = dim0_length / n_part_ + 1;
    } else {
      shapes[0] = dim0_length / n_part_;
    }
    std::vector<int64_t> shape_indices;
    shape_indices.push_back(shapes.size());
    return std::make_pair<Expr, Expr>(MakeConstant(raf::op::ArrayToIntTuple(shapes)),
                                      MakeConstant(raf::op::ArrayToIntTuple(shape_indices)));
  }

  /*! \brief Since we always partition the first dimension of gradient tensors, we have to
   * make sure the length of the first dimension is dividable to the total device number.
   * This helper function analyzes the tensor size and generates the pad call if needed.
   * */
  inline Var GenPadCall(LetList* scope, const Var& var) {
    static const Op& pad_op = Op::Get("raf.op.pad");

    // Extract the length of the first dimension.
    auto ttype = var->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr) << "Expected a tesnor, but got " << var->checked_type();
    auto shape_expr = ttype->shape[0];
    if (!shape_expr->IsInstance<IntImmNode>()) {
      LOG(FATAL) << "Do not support dynamic shape yet";
      throw;
    }
    auto dim0_length = tvm::tir::as_const_int(shape_expr)[0];

    // Check if we need to pad the gradient or not.
    auto ret_var = var;
    if (dim0_length % n_part_ != 0) {
      Array<Value> pad_width;
      auto part_dim0_length = dim0_length / n_part_ + 1;
      auto target_dim0_length = part_dim0_length * n_part_;
      pad_width = Array<Value>(ttype->shape.size() * 2, ScalarValue::make(0));
      // Always pad 0s to the end of the first axis.
      pad_width.Set(1, ScalarValue::make(target_dim0_length - dim0_length));

      ret_var = scope->Push(Call(pad_op, {var, MakeConstant(TupleValue::make(pad_width)),
                                          MakeConstant(ScalarValue::make(0)),
                                          MakeConstant(StringValue::make("constant"))}));
    }
    return ret_var;
  }

  /*!
   * \brief Slice gradient based on ZeRO-1 and ZeRO-2.
   * The desired IR for ZeRO-1 is:
   * let %1 = all_reduce(%0); // Could also be a backward op if data_parallel is disabled
   * let %2 = pad(%1, ...);   // %1 is the complete global gradient
   * let %3 = split(%2, ...);
   * let %4 = TupleGetItem(%3, rank);
   * TODO(comaniac): Add %rank to the function argument if rank_ is unknown.
   *
   * The desired IR for ZeRO-2 is:
   * // if NCCL version is >= 2.10
   * let %1 = op(%0);       // A backward op to generate gradient
   * let %2 = pad(%1, ...); // %1 is the complete local gradient
   * let %3 = split(%2, ...);
   * let %4 = reduce_scatter(%3, avg);
   * // else NCCL version is < 2.10
   * let %1 = op(%0);       // A backward op to generate gradient
   * let %2 = pad(%1, ...); // %1 is the complete local gradient
   * let %3 = split(%2, ...);
   * let %4 = reduce_scatter(%3, sum);
   * let %5 = divide(%4, ...)
   */
  Var SliceGrad(LetList* scope, const Var& var, const Expr& value, int opt_level) {
    static const Op& split_op = Op::Get("raf.op.split");
    static const Op& reduce_scatter_op = Op::Get("raf.op._reduce_scatter");
    Var allreduce_var, allreduce_tgi_var, divide_var;
    std::tie(allreduce_var, allreduce_tgi_var, divide_var) = GetAllReduceVar(var);
    if (opt_level_ > 1 && !allreduce_var.defined()) {
      // If this is not an AllReduce, then the gradient was generated locally and
      // no need to apply ZeRO-2.
      opt_level = 1;
    }
    CHECK(var_to_expr_.count(allreduce_var));
    auto allreduce_expr = var_to_expr_.at(allreduce_var);
    CHECK(IsAllReduceCall(allreduce_expr));
    auto grad_var = var;
    if (opt_level > 1) {
      // ZeRO-2: Replace the AllReduce with ReduceScatter.
      auto first_arg = Downcast<Var>(GetNArg(allreduce_expr, 0));
      // The 1st arg of allreduce is a tuple of tensors.
      CHECK(var_to_expr_.count(first_arg));
      auto arg_tuple = Downcast<Tuple>(var_to_expr_.at(first_arg));
      CHECK_EQ(arg_tuple->fields.size(), 1U) << "Not supported yet";
      CHECK(arg_tuple.defined());
      // FIXME(comaniac): This happens when gradients are zeros and are folded.
      // However, we should eliminate zero gradients to reduce communication overheads.
      if (arg_tuple->fields[0]->IsInstance<ConstantNode>()) {
        grad_var = scope->Push(TupleGetItem(first_arg, 0));
        grad_var->checked_type_ = arg_tuple->fields[0]->checked_type();
      } else {
        grad_var = Downcast<Var>(arg_tuple->fields[0]);
      }
    } else {
      // ZeRO-1: Keep AllReduce (or the backward op if data parallel is disabled).
      grad_var = scope->Push(allreduce_var, allreduce_expr);
      if(allreduce_tgi_var.defined()) {
        CHECK(var_to_expr_.count(allreduce_tgi_var));
        grad_var = scope->Push(allreduce_tgi_var, var_to_expr_.at(allreduce_tgi_var));
      }
    }
    auto ttype = grad_var->checked_type().as<TensorTypeNode>();
    CHECK(ttype != nullptr) << "Expected a tesnor, but got " << grad_var->checked_type();
    if (ttype->shape.size()) {
      if (opt_level > 1) {
        // ZeRO-2 eliminate split ops
        auto compute = Downcast<Constant>(GetNArg(allreduce_expr, 1));
        auto reduce_scatter_args = GetReduceScatterArgs(grad_var);
        auto shapes = Downcast<Constant>(reduce_scatter_args.first);
        auto shape_indices = Downcast<Constant>(reduce_scatter_args.second);
        grad_var = GenPadCall(scope, grad_var);
        grad_var = scope->Push(Tuple({grad_var}));
        auto reduce_scatter_var =
            scope->Push(Call(reduce_scatter_op, {grad_var, shapes, shape_indices, compute}));
        if (divide_var.defined()) {
          // update the divide op args
          CHECK(var_to_expr_.count(divide_var));
          auto divide_expr = var_to_expr_.at(divide_var);
          auto divide_call = divide_expr.as<CallNode>();
          return scope->Push(Call(divide_call->op, {reduce_scatter_var, divide_call->args[1]}));
        }
        return reduce_scatter_var;
      } else {
        // ZeRO-1 requires split ops
        grad_var = GenPadCall(scope, grad_var);
        grad_var = scope->Push(Call(split_op, {grad_var, MakeConstant(ScalarValue::make(n_part_)),
                                               MakeConstant(ScalarValue::make(0))}));
        if (divide_var.defined()) {
          // update the divide op args
          CHECK(var_to_expr_.count(divide_var));
          auto divide_expr = var_to_expr_.at(divide_var);
          auto divide_call = divide_expr.as<CallNode>();
          grad_var = scope->Push(TupleGetItem(grad_var, rank_));
          return scope->Push(Call(divide_call->op, {grad_var, divide_call->args[1]}));
        }
      }
    } else {
      // ignore scalar tensor
      if (opt_level > 1) {
        // add back allreduce and gettupleitem
        grad_var = scope->Push(allreduce_var, allreduce_expr);
        if(allreduce_tgi_var.defined()) {
          CHECK(var_to_expr_.count(allreduce_tgi_var));
          grad_var = scope->Push(allreduce_tgi_var, var_to_expr_.at(allreduce_tgi_var));
        }
      }
      return grad_var;
    }
    return scope->Push(TupleGetItem(grad_var, rank_));
  }

  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The optimization level (ZeRO-n). */
  int opt_level_;
  /*! \brief The expected number of partitions. */
  int n_part_;
  /*! \brief The target function. */
  Function func_;
  /*! \brief The rank of the current running device. */
  int rank_;
  /*! \brief Mapping from a gradient to a sliced gradient for this rank. */
  Map<Var, Expr> grads_;
  /*! \brief Mapping from a allreduce all to its input. */
  Map<Var, Expr> allreduces_;
  Map<Var, Expr> allreduce_tgis_;
  /*! \brief The var binding to the gradient tuple. */
  Var grad_tuple_var_;
  /*! \brief Mapping from let-binding var to the expression. */
  Map<Var, Expr> var_to_expr_;
};

}  // namespace partition_gradient

Pass PartitionGradient(int opt_level, int n_part, int rank) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return partition_gradient::GradientPartitioner(opt_level, n_part, f).Partition(rank);
  };
  auto partition_gradient = CreateRAFFunctionPass(pass_func, 0, "PartitionGradientFunc", {});
  return RAFSequential({InferType(), partition_gradient, EraseType(), DeadCodeElimination()},
                       "PartitionGradient");
}

RAF_REGISTER_GLOBAL("raf.pass_.PartitionGradient").set_body_typed(PartitionGradient);

}  // namespace pass
}  // namespace raf
