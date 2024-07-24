/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file liveness_analysis.cc
 * \brief A pass for analyzing tensor liveness.
 */
#include "liveness_analysis.h"
#include <vector>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "tvm/ir/type_functor.h"
#include "./let_list.h"
#include "./common.h"
#include "../common/shape_utils.h"

namespace raf {
namespace pass {

using PackedLiveInMap = Map<Var, Array<Var>>;
using PackedStreamLiveInMap = Map<Var, Map<Integer, Array<Var>>>;

namespace liveness_analysis {

MapStreamVSet LivenessAnalyzer::Run() {
  Expr body;
  FormCheck(func_->body);
  if (failure_) {
    return live_;
  }

  for (const auto& var : func_->params) {
    auto var_type = var->checked_type();
    if (var_type.as<TensorTypeNode>()) {
      Var tvar = CreateTensor("param");
      Init(var, tvar);
    } else if (const auto& tuple_type = var_type.as<TupleTypeNode>()) {
      Array<Var> fields;
      for (const auto& field : tuple_type->fields) {
        CHECK(field.as<TensorTypeNode>())
            << "Expected a tensor in tuple parameter, but got " << field->GetTypeKey();
        Var tvar = CreateTensor("param");
        fields.push_back(tvar);
      }
      Init(var, Merge(fields));
      vtuple_.Set(var, fields);
    } else {
      LOG(FATAL) << "NotImplementedError: Unsupported parameter type: " << var->GetTypeKey();
    }
  }

  // forward analysis
  Forward(func_->body);

  // backward analysis
  Var dummy = CreateNull();
  live_[dummy] = {};
  Backward(func_->body, dummy);

  // init find
  for (const auto& kv : vset_) {
    const Var& var = kv.first;
    const auto& tensors = GetTensorVars(var);
    if (tensors.size() == 1 && *tensors.begin() == var) {
      union_find_forest_[var] = var;
    }
  }

  // init inv
  for (const auto& kv : live_) {
    const Var& k = kv.first;
    const StreamVSet& vs = kv.second;
    for (const auto& stream_kv : vs) {
      for (const auto& v: stream_kv.second) {
        inv_live_[v][stream_kv.first].insert(k);
      }
    }
  }

  // mandatory memory sharing
  CHECK_EQ(var_out_.size(), var_in_.size());
  int m = var_out_.size();
  for (int i = 0; i < m; ++i) {
    Var fout = *GetTensorVars(var_out_[i]).begin();
    Var fin = *GetTensorVars(var_in_[i]).begin();
    CHECK(fout.defined());
    CHECK(fin.defined());
    fout = Find(fout);
    fin = Find(fin);
    if (fout != fin && Intersect(fout, fin)) {
      // the mandatory inplace update is invalid
      // something goes wrong here
      LOG(WARNING) << "Mandatory memory sharing between " << fin << " and " << fout
                   << " is invalid. Such cases cannot be handled by "
                   << "the liveness_analysis pass.";
      failure_ = true;
    } else {
      // the mandatory inplace update is valid
      Unite(fin, fout);
    }
  }

  return live_;
}

void LivenessAnalyzer::FormChecker::VisitExpr_(const CallNode* node) {
  const Array<Expr>& args = node->args;
  Array<Var> vargs;
  for (const auto& arg : node->args) {
    if (arg.as<VarNode>() == nullptr && arg.as<ConstantNode>() == nullptr) {
      // Only support ANF with exceptions of Constant
      analyzer_->failure_ = true;
    }
  }
}

void LivenessAnalyzer::FormChecker::VisitExpr_(const FunctionNode* node) {
  // TODO(hzfan, comaniac): Support liveness analysis in closures.
  // We do not support liveness analysis in closures for now, so we treat closures
  // as ops and will not analyze tensor liveness inside closure body.
  VisitSpan(node->span);
  for (auto param : node->params) {
    ExprVisitor::VisitExpr(param);
  }
}

void LivenessAnalyzer::FormChecker::VisitExpr_(const LetNode* op) {
  // avoid stack overflow
  auto pre_visit = [this](const LetNode* op) {
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
  };
  auto post_visit = [this](const LetNode* op) {
    this->VisitExpr(op->body);
    this->visit_counter_[op] += 1;
  };
  ExpandANormalForm(op, pre_visit, post_visit);
}

void LivenessAnalyzer::FormChecker::VisitExpr_(const IfNode* node) {
  if (node->cond.as<VarNode>() == nullptr) {
    // Only support ANF.
    analyzer_->failure_ = true;
  }
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const VarNode* node) {
  auto vars = analyzer_->GetTensorVars(GetRef<Var>(node));
  CHECK_EQ(vars.size(), 1U);
  analyzer_->Init(let_var_, vars[0]);
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const FunctionNode* node) {
  /*!
   * When a closure is used, the value of the captured variables are required.
   * For example, in
   * fn {
   *   let %closure = {
   *     %b1 = %a1 + %a1
   *     %b1
   *   }
   *   %closure  // here %a1 is used, and thus cannot be inplace rewritten
   * }
   * when the closure is invoked/returned, the value of %a1 (captured variables) is needed.
   */
  Function f = GetRef<Function>(node);
  Array<Var> free_vars = FreeVars(f);
  analyzer_->Init(let_var_, analyzer_->Merge(free_vars));
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const LetNode* op) {
  // avoid stack overflow
  auto pre_visit = [this](const LetNode* op) {
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
  };
  auto post_visit = [this](const LetNode* op) {
    this->VisitExpr(op->body);
    this->visit_counter_[op] += 1;
  };
  ExpandANormalForm(op, pre_visit, post_visit);
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const CallNode* node) {
  if (node->op->IsInstance<OpNode>() && op::IsReshapeOp(Downcast<Op>(node->op))) {
    // Reshape ops does not create a new tensor but just a view, so treat them as a direct assign.
    auto var = node->args[0].as<VarNode>();
    CHECK(var != nullptr) << "Expected the first argument of reshape op to be a Var, but got "
                          << node->args[0]->GetTypeKey();
    this->VisitExpr_(var);

  } else {
    Var dummy = analyzer_->CreateTensorVar(node->checked_type());
    // LOG(INFO) << "Creating variable " << dummy << " for call " << GetRef<Call>(node);
    analyzer_->Init(let_var_, dummy);
  }
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const TupleNode* node) {
  Array<Var> fields;
  for (const auto& field : node->fields) {
    Var var;
    if (field.as<VarNode>()) {
      // Ignore constant fields (e.g., NoGradValue)
      var = Downcast<Var>(field);
    }
    fields.push_back(var);
  }
  auto merged = analyzer_->Merge(fields);
  // LOG(INFO) << "Creating variable " << merged << " for tuple " << GetRef<Tuple>(node);
  analyzer_->Init(let_var_, merged);
  analyzer_->vtuple_.Set(let_var_, fields);
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const TupleGetItemNode* node) {
  auto tuple = Downcast<Var>(node->tuple);
  CHECK(analyzer_->vtuple_.find(tuple) != analyzer_->vtuple_.end());
  Var var = analyzer_->vtuple_.at(tuple)[node->index];
  // LOG(INFO) << "Using variable " << var << " for tuple get item " << GetRef<TupleGetItem>(node);
  analyzer_->Init(let_var_, var);
}

void LivenessAnalyzer::ForwardAnalyzer::VisitExpr_(const IfNode* node) {
  Expr true_branch = node->true_branch;
  Expr false_branch = node->false_branch;
  Var true_ret = analyzer_->Forward(true_branch);
  Var false_ret = analyzer_->Forward(false_branch);
  Var ret = analyzer_->CreateTensorVar(node->checked_type());
  // mandatory memory sharing if condition is true
  Match(ret, true_ret);
  // mandatory memory sharing if condition is false
  Match(ret, false_ret);
  analyzer_->Init(let_var_, ret);
}

void LivenessAnalyzer::ForwardAnalyzer::Match(Var v1, Var v2) {
  if (analyzer_->vtuple_.count(v1) > 0) {
    CHECK(analyzer_->vtuple_.find(v1) != analyzer_->vtuple_.end());
    CHECK(analyzer_->vtuple_.find(v2) != analyzer_->vtuple_.end());
    Array<Var> v1t = analyzer_->vtuple_.at(v1);
    Array<Var> v2t = analyzer_->vtuple_.at(v2);
    Array<Var> fields;
    CHECK_EQ(v1t.size(), v2t.size());
    for (size_t i = 0; i < v1t.size(); ++i) {
      Match(v1t[i], v2t[i]);
    }
  } else {
    analyzer_->var_out_.push_back(v1);
    analyzer_->var_in_.push_back(v2);
  }
}

Var LivenessAnalyzer::ForwardAnalyzer::Run() {
  const auto& vars = ell_->vars;
  const auto& exprs = ell_->exprs;
  CHECK_EQ(vars.size(), exprs.size());
  int n = exprs.size();

  // Forward analysis
  for (int i = 0; i < n; ++i) {
    let_var_ = vars[i];

    // We need to handle OpNode and ConstantNode here, because all these nodes with
    // the same value may point to the same reference, so only the first one will be visited.
    if (exprs[i].as<OpNode>()) {
      analyzer_->Init(let_var_, analyzer_->CreateNull("op"));
    } else if (exprs[i].as<ConstantNode>()) {
      analyzer_->Init(let_var_, analyzer_->CreateNull("const"));
    }
    ExprVisitor::VisitExpr(exprs[i]);
  }
  return ell_->ret;
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const VarNode* node) {
  auto vars = analyzer_->GetTensorVars(GetRef<Var>(node));
  CHECK_EQ(vars.size(), 1U);
  analyzer_->live_[let_var_] = MergeLive(vars[0]);
  // LOG(INFO) << "Setting live of " << let_var_ << "(Vid: " << let_var_->vid << ")";
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const FunctionNode* node) {
  Function f = GetRef<Function>(node);
  Array<Var> free_vars = FreeVars(f);
  analyzer_->live_[let_var_] = MergeLive(let_var_);
  // LOG(INFO) << "Setting live of " << let_var_ << "(Vid: " << let_var_->vid << ")";
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const CallNode* node) {
  const Array<Expr>& args = node->args;
  if (node->op->IsInstance<OpNode>() && op::IsReshapeOp(Downcast<Op>(node->op))) {
    // Reshape ops does not create a new tensor but just a view, so treat them as a direct assign.
    auto var = args[0].as<VarNode>();
    CHECK(var != nullptr) << "Expected the first argument of reshape op to be a Var, but got "
                          << args[0]->GetTypeKey();
    this->VisitExpr_(var);
  } else {
    Array<Var> vargs;
    for (const auto& arg : node->args) {
      if (arg.as<VarNode>()) {
        // use %arg
        vargs.push_back(Downcast<Var>(arg));
      } else if (arg.as<ConstantNode>() || arg.as<OpNode>()) {
        // use nothing
      } else {
        LOG(FATAL) << "NotImplementedError: unsupported args: " << arg->GetTypeKey();
      }
    }
    Var d1 = analyzer_->Merge(vargs);
    analyzer_->live_[let_var_] = MergeLive(d1, let_var_);
    // LOG(INFO) << "Setting live of " << let_var_ << "(Vid: " << let_var_->vid << ")";
  }
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const TupleNode* node) {
  analyzer_->live_[let_var_] = MergeLive(let_var_);
  // LOG(INFO) << "Setting live of " << let_var_ << "(Vid: " << let_var_->vid << ")";
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const TupleGetItemNode* node) {
  analyzer_->live_[let_var_] = MergeLive(let_var_);
  // LOG(INFO) << "Setting live of " << let_var_ << "(Vid: " << let_var_->vid << ")";
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const LetNode* op) {
  // avoid stack overflow
  auto pre_visit = [this](const LetNode* op) {
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
  };
  auto post_visit = [this](const LetNode* op) {
    this->VisitExpr(op->body);
    this->visit_counter_[op] += 1;
  };
  ExpandANormalForm(op, pre_visit, post_visit);
}

void LivenessAnalyzer::BackwardAnalyzer::VisitExpr_(const IfNode* node) {
  CHECK(false) << "If is not supported.";
  // Var free_true = analyzer_->Merge(FreeVars(node->true_branch));
  // Var free_false = analyzer_->Merge(FreeVars(node->false_branch));
  // analyzer_->live_[let_var_] = analyzer_->vset_[MergeLive(
  //     analyzer_->Merge({free_true, free_false, Downcast<Var>(node->cond)}), let_var_)];
  // VisitBranch(node->true_branch, let_var_);
  // VisitBranch(node->false_branch, let_var_);
}

void LivenessAnalyzer::BackwardAnalyzer::VisitBranch(const Expr& branch, const Var& def) {
  CHECK(false) << "Branch is not supported.";
  // Var total_next = analyzer_->CreateTensorVar("if");
  // // get total live-out variables of true_branch
  // analyzer_->vset_[total_next] = analyzer_->live_[next_var_];
  // // remove the tensors defined at this line
  // Var branch_next = analyzer_->Remove(total_next, def);
  // analyzer_->live_[branch_next] = analyzer_->vset_[branch_next];
  // analyzer_->Backward(branch, branch_next);
}

void LivenessAnalyzer::BackwardAnalyzer::Run(Var next_var) {
  static const Op& set_stream_op = Op::Get("raf.op.set_stream");

  const auto& vars = ell_->vars;
  const auto& exprs = ell_->exprs;
  CHECK_EQ(vars.size(), exprs.size());
  int n = exprs.size();

  // map each line to its executing stream
  StdMap<int> line_to_stream;
  for (int i = 0; i < n; ++i) {
    auto var = vars[i];
    auto expr = exprs[i];
    if(expr.as<CallNode>() && Downcast<Call>(expr)->op == set_stream_op) {
      current_stream_ = Downcast<Call>(expr)->args[1].as<ConstantNode>()->value.as<IntValueObj>()->value;
      streams_.insert(current_stream_);
    }
    line_to_stream[var] = current_stream_;
  }
  if (streams_.size() == 0) {
    streams_.insert(-1);
  }

  // Backward analysis
  next_var_ = next_var;
  analyzer_->dummy_output_ = analyzer_->CreateNull();
  analyzer_->live_[analyzer_->dummy_output_] = MergeLive(ell_->ret);
  // LOG(INFO) << "Setting live of dummy " << analyzer_->dummy_output_ << "(Vid: " << analyzer_->dummy_output_->vid << ")";
  for (int i = n - 1; i >= 0; --i) {
    let_var_ = vars[i];
    // LOG(INFO) << "Visiting " << let_var_ << "(Vid: " << let_var_->vid << ") in backward analysis";
    next_var_ = i == n - 1 ? analyzer_->dummy_output_ : vars[i + 1];
    current_stream_ = line_to_stream[let_var_];

    // We need to handle these nodes here, because all these nodes with
    // the same value may point to the same reference, so only the first one will be visited.
    if (exprs[i].as<OpNode>() || exprs[i].as<ConstantNode>() || exprs[i].as<FunctionNode>()) {
      auto dummy_vars = analyzer_->GetTensorVars(let_var_);
      Var d1 = analyzer_->Merge(dummy_vars);
      analyzer_->live_[let_var_] = MergeLive(d1);
      // LOG(INFO) << "Setting live of " << let_var_ << "(Vid: " << let_var_->vid << ")";
    } else {
      CHECK_GT(analyzer_->live_.count(next_var_), 0);
    }
    ExprVisitor::VisitExpr(exprs[i]);
  }
}

Var LivenessAnalyzer::Forward(const Expr& e) {
  return ForwardAnalyzer(e, this).Run();
}

void LivenessAnalyzer::Backward(const Expr& e, const Var& next_var) {
  BackwardAnalyzer(e, this).Run(next_var);
}

void LivenessAnalyzer::FormCheck(const Expr& e) {
  FormChecker(e, this).Run();
}

Var LivenessAnalyzer::CreateTensorVar(const Type& type) {
  return VarCreator(this).Run(type);
}

/*! \brief Calculate the byte compact size of the given type. If the type is a tuple,
 * then the size of each tensor in the tuple will be returned. Note that size 0 means
 * a tensor with dynamic shape.
 */
std::vector<int64_t> CalcBytesCompactSizes(const Type& type) {
  std::vector<const TensorTypeNode*> ttypes;
  std::vector<int64_t> sizes;
  if (auto tuple_type = type.as<TupleTypeNode>()) {
    for (auto field : tuple_type->fields) {
      auto ttype = field.as<TensorTypeNode>();
      CHECK(ttype != nullptr) << "Nested tuple is not supported";
      ttypes.push_back(ttype);
    }
  } else if (auto ttype = type.as<TensorTypeNode>()) {
    ttypes.push_back(ttype);
  } else {
    LOG(FATAL) << "Unsupported type: " << type->GetTypeKey();
    throw;
  }

  for (auto ttype : ttypes) {
    sizes.push_back(common::shape_utils::BytesCompactTensor(ttype));
  }
  return sizes;
}

/*! \brief Dump liveness analysis result statistics. */
void DumpLivenessStat(const MapVSet& live_in) {
  std::stringstream ss;
  ss << "Liveness Analysis Result Statistics: " << std::endl;

  // Peak tensor number.
  int peak_tensor_num = 0;

  // Each tensor (var) to the length of its live.
  StdMap<int> var_to_live_length;
  for (auto it : live_in) {
    peak_tensor_num = (it.second.size() > peak_tensor_num) ? it.second.size() : peak_tensor_num;
    for (auto var : it.second) {
      var_to_live_length[var]++;
    }
  }
  ss << "Peak number of live tensors: " << peak_tensor_num << std::endl;

  // Each appeared live length to nmuber of tensors.
  float avg_length = 0;
  std::unordered_map<int, int> live_length_to_freq;
  for (auto it : var_to_live_length) {
    live_length_to_freq[it.second] += 1;
    avg_length += it.second;
  }
  avg_length /= var_to_live_length.size();
  ss << "Average life length: " << avg_length << std::endl;
  ss << "Detail live length to frequency: " << std::endl;
  for (auto it : live_length_to_freq) {
    ss << std::setw(5) << it.first << std::setw(5) << it.second << std::endl;
  }
  LOG(INFO) << ss.str();
}

}  // namespace liveness_analysis

liveness_analysis::MapStreamVSet LivenessAnalysis(const IRModule& mod) {
  auto entry = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(entry));
  auto la = liveness_analysis::LivenessAnalyzer(func);
  return la.Run();
}

// Put the live in set to an Array as std::unordered_set is not in the object system.
PackedStreamLiveInMap LivenessAnalysisPacked(const IRModule& mod) {
  PackedStreamLiveInMap ret;
  auto res = LivenessAnalysis(mod);
  for (const auto& it : res) {
    Map<Integer, Array<Var>> stream_vars;
    for (const auto& stream_it: it.second) {
      int stream = stream_it.first;
      Array<Var> vars;
      for (const auto& var: stream_it.second) {
        vars.push_back(var);
      }
      stream_vars.Set(stream, vars);
    }
    ret.Set(it.first, stream_vars);
  }
  return ret;
}

RAF_REGISTER_GLOBAL("raf.pass_.LivenessAnalysis").set_body_typed(LivenessAnalysisPacked);

}  // namespace pass
}  // namespace raf
