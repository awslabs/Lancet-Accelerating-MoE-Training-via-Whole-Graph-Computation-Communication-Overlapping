/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file enforce_sync.cc
 * \brief Enforce synchronization between ops in multiple streams.
 */
#include "raf/device.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/dist_context.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/stream_pool.h"
#include "../common.h"
#include "../../common/shape_utils.h"
#include "../let_list.h"

namespace raf {
namespace pass {
namespace enforce_sync {
using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using raf::distributed::DistContext;
using namespace raf::analysis;
using stream_pool::StreamTagEnum;
using common::shape_utils::BytesCompactTensor;

using OpSet = std::unordered_set<Op, ObjectPtrHash, ObjectPtrEqual>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

struct pair_hash {
  std::size_t operator()(const std::pair<int, int>& v) const {
    return std::hash<std::string>{}(std::to_string(v.first) + "," + std::to_string(v.second));
  }
};

struct pair_equal {
  bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};

using DepSet = std::unordered_set<std::pair<int, int>, pair_hash, pair_equal>;

static int64_t compute_stream_idx = StreamTagEnum::CudaCompute();
static int64_t communication_stream_idx = StreamTagEnum::CudaCommunicate();
static int64_t fuse_tensor_stream_idx = StreamTagEnum::MemCudaToCuda1();
static int64_t defuse_tensor_stream_idx = StreamTagEnum::MemCudaToCuda2();
static int64_t unknown_stream_idx = StreamTagEnum::Unknown();

static std::unordered_map<int64_t, std::string> stream_name_hint = {
    {compute_stream_idx, "comp"},
    {communication_stream_idx, "comm"},
    {fuse_tensor_stream_idx, "fuse"},
    {defuse_tensor_stream_idx, "defuse"},
};

int IdentifyStream(const Expr& op) {
  int stream_idx = compute_stream_idx;
  if (op->IsInstance<CallNode>() && IsCollectiveOp(op.as<CallNode>()->op)) {
    stream_idx = communication_stream_idx;
    // if need fuse_tensor before collective op, previous stream should be memory copy stream
  } else if (op->IsInstance<CallNode>() && IsFuseTensorOp(op.as<CallNode>()->op)) {
    // if already have memcpy ops in the graph.
    stream_idx = fuse_tensor_stream_idx;
  } else if (op->IsInstance<CallNode>() && IsDefuseTensorOp(op.as<CallNode>()->op)) {
    stream_idx = defuse_tensor_stream_idx;
  }
  return stream_idx;
}

class FusedCollectivesAnalyzer : public ExprVisitor {
 public:
  virtual void VisitExpr_(const VarNode* var) {
    var_idx_map_[GetRef<Var>(var)] = current_idx_;
  }

  virtual void VisitExpr_(const CallNode* call) {
    auto call_expr = GetRef<Expr>(call);
    UpdateDependencyInfo_(call_expr, call->args);
  }

  virtual void VisitExpr_(const TupleNode* tuple) {
    auto tuple_expr = GetRef<Expr>(tuple);
    UpdateDependencyInfo_(tuple_expr, tuple->fields);
  }

  virtual void VisitExpr_(const TupleGetItemNode* tuple_get_item) {
    auto tuple_get_item_expr = GetRef<Expr>(tuple_get_item);
    UpdateDependencyInfo_(tuple_get_item_expr, {tuple_get_item->tuple});
  }

  void VisitExpr_(const FunctionNode* op) override {
    // currently assumes closures do not contain collectives
    // TODO(@chenyu-jiang): move this pass after lambda lift and enable it
    // to track event ids used across differnt runs
    auto ell = ExplicitLetList::make(op->body);
    for (auto& expr : ell->exprs) {
      CHECK(!(expr.as<CallNode>() && IsCollectiveOp(expr.as<CallNode>()->op)))
          << "Unimplemented: Collectives in closures are currently not supported.";
    }
  }

  virtual void VisitExpr_(const LetNode* op) {
    auto pre_visit = [this](const LetNode* op) {
      Var var = op->var;
      Expr value = op->value;
      expr_idx_map_[value] = current_idx_;
      var_idx_map_[var] = current_idx_;
      idx_expr_map_[current_idx_] = value;
      this->VisitExpr(var);
      this->VisitExpr(value);
      current_idx_++;
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = op->body;
      this->VisitExpr(body);
      current_idx_--;
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  /*! \brief Analyse the predecessors and successors for fused collectives.
   *
   * \returns true if there is any fused collectives in the expr.
   */
  bool Analyse(const Expr& func_body, const Array<Var>& func_params) {
    // populate func_param_to_idx_ map
    for(int i=0; i<func_params.size(); i++) {
      func_param_to_idx_[func_params[i]] = -(func_params.size() - i);
    }
    VisitExpr(func_body);
    // check if any communication collective is found
    if (fused_comm_call_to_fused_size_.empty()) {
      return false;
    }
    return true;
  }

  // get the corresponding communication call from a producer, as well as whether
  // the producer op is the first producer of the fused inputs
  std::vector<std::tuple<Expr, int64_t, bool>> GetCommFromProducer(const Expr& prod) {
    if(producer_to_comm_calls_.count(prod)) {
      std::vector<std::tuple<Expr, int64_t, bool>> result;
      for(auto fused_comm_call_and_offset: producer_to_comm_calls_.at(prod)) {
        Expr fused_comm_call = fused_comm_call_and_offset.first;
        int64_t offset = fused_comm_call_and_offset.second;
        if(fused_comm_call_to_first_producer_.count(fused_comm_call)) {
          if(fused_comm_call_to_first_producer_.at(fused_comm_call) == prod) {
            result.push_back(std::make_tuple(fused_comm_call, offset, true));
          } else {
            result.push_back(std::make_tuple(fused_comm_call, offset, false));
          }
        } else {
          result.push_back(std::make_tuple(fused_comm_call, offset, false));
        }
      }
      return result;
    } else {
      return {};
    }
  }

  ExprMap<std::vector<std::pair<Expr, int64_t>>> GetProducerMap() {
    return producer_to_comm_calls_;
  }

  int64_t GetCommFusedSizeInBytes(const Expr& comm_call) {
    if (!fused_comm_call_to_fused_size_.count(comm_call)) {
      return -1;
    }
    return fused_comm_call_to_fused_size_.at(comm_call);
  }

  DLDataType GetCommDataType(const Expr& comm_call) {
    DLDataType dtype;
    if (!fused_comm_call_to_dtype_.count(comm_call)) {
      return dtype;
    }
    return fused_comm_call_to_dtype_.at(comm_call);
  }

  std::pair<int64_t, int64_t> GetCommArgOffsetAndSizeAtIdx(const Expr& comm_call, int idx) {
    if(!fused_comm_call_to_offset_and_size_.count(comm_call)) {
      return std::make_pair(-1, -1);
    } else {
      auto& args_info = fused_comm_call_to_offset_and_size_.at(comm_call);
      if (idx >= args_info.size()) {
        return std::make_pair(-1, -1);
      } else {
        return args_info[idx];
      }
    }
  }

  bool IsFusedCommExpr(const Expr& expr) {
    return fused_comm_call_to_fused_size_.count(expr);
  }

 private:
  inline virtual int GetIndexFromVar_(const Expr& expr) {
    if(var_idx_map_.count(expr)) {
      return var_idx_map_.at(expr);
    } else {
      return -1;
    }
  }

  inline bool IsFusedCollectiveOpCall(const CallNode* call) {
    // ignore reduce scatter op for now
    if(!IsCollectiveOp(call->op)) {
      return false;
    }
    // assumption: for all collective ops the first argument contains the data tuple
    // UPDATE (temporary?): except all_to_all
    auto data = call->args[0];
    int index = GetIndexFromVar_(data);
    if(index >= 0) {
      // have a actual tuple node in function body
      CHECK(idx_expr_map_.count(index));
      auto input_tuple_node = idx_expr_map_.at(index).as<TupleNode>();
      if(input_tuple_node && input_tuple_node->fields.size() > 1) {
        return true;
      }
    } else {
      // input is in function params
      CHECK(data->checked_type().defined() && data->checked_type().as<TupleTypeNode>());
      auto input_type = data->checked_type().as<TupleTypeNode>();
      if (input_type->fields.size() > 1) {
        return true;
      }
    }
    return false;
  }

  void UpdateDependencyInfo_(const Expr& expr, const Array<Expr>& input_args) {
    if (expr->IsInstance<CallNode>() && IsFusedCollectiveOpCall(expr.as<CallNode>())) {
      // we only need to record dependency for fused collective ops
      auto arg_var = input_args[0];
      int arg_idx = GetIndexFromVar_(arg_var);
      if(arg_idx < 0) {
        // Input of a fused collective op is not in var map, it must be a function argument
        // we don't do OTF copy in this case
        return;
      } else {
        auto arg_expr = idx_expr_map_.at(arg_idx);
        CHECK(arg_expr.defined()) << "Encountered undefined args in " << expr;
        auto arg_tuple_node = arg_expr.as<TupleNode>();
        CHECK(arg_tuple_node) << "Input of a fused collective op is not a tuple: Coll: " << expr;
        // some of the arguments may be in function params. we use first_producer_idx < 0 to denote
        int first_producer_idx = GetIndexFromVar_(arg_tuple_node->fields[0]);
        Expr first_producer_expr = arg_tuple_node->fields[0];
        if(first_producer_idx == -1) {
          // replace with func args
          CHECK(func_param_to_idx_.count(arg_tuple_node->fields[0]));
          first_producer_idx = func_param_to_idx_.at(arg_tuple_node->fields[0]);
        } else {
          first_producer_expr = idx_expr_map_.at(first_producer_idx);
        }
        int64_t fused_size = 0;
        for(auto producer_var: arg_tuple_node->fields) {
          int producer_idx = GetIndexFromVar_(producer_var);
          if(producer_idx == -1) {
            // replace with func args
            CHECK(func_param_to_idx_.count(producer_var));
            producer_idx = func_param_to_idx_.at(producer_var);
          }
          auto producer_expr = Expr();
          if(producer_idx >= 0) {
            CHECK(idx_expr_map_.count(producer_idx));
            producer_expr = idx_expr_map_.at(producer_idx);
          } else {
            // since it's a function argument
            producer_expr = producer_var;
          }
          if(producer_idx < first_producer_idx) {
            first_producer_idx = producer_idx;
            first_producer_expr = producer_expr;
          }
          // calculate tensor size
          CHECK(producer_var->checked_type().defined()) << "Type for node " << producer_var << " is not defined.";
          auto tensor_type_node = producer_var->checked_type().as<TensorTypeNode>();
          CHECK(tensor_type_node) << "Input tuple contains non-tensor element: " << producer_var << ".";
          if(!fused_comm_call_to_dtype_.count(expr)) {
            fused_comm_call_to_dtype_[expr] = tensor_type_node->dtype;
          } else {
            CHECK(DataType(fused_comm_call_to_dtype_[expr]) == tensor_type_node->dtype);
          }
          int64_t field_size = BytesCompactTensor(tensor_type_node);
          // record offset in fused call
          producer_to_comm_calls_[producer_expr].push_back(std::make_pair(expr, fused_size));
          fused_comm_call_to_offset_and_size_[expr].push_back(std::make_pair(fused_size, field_size));
          fused_size += field_size;
        }
        fused_comm_call_to_first_producer_[expr] = first_producer_expr;
        fused_comm_call_to_fused_size_[expr] = fused_size;
      }
    }
  }

  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
  // maps each var to its index
  ExprMap<int> var_idx_map_;
  // maps each expr to its index
  ExprMap<int> expr_idx_map_;
  // index to expr map
  std::unordered_map<int, Expr> idx_expr_map_;

  ExprMap<int64_t> func_param_to_idx_;
  ExprMap<std::vector<std::pair<Expr, int64_t>>> producer_to_comm_calls_;
  ExprMap<int64_t> fused_comm_call_to_fused_size_;
  ExprMap<std::vector<std::pair<int64_t, int64_t>>> fused_comm_call_to_offset_and_size_;
  ExprMap<DLDataType> fused_comm_call_to_dtype_;
  ExprMap<Expr> fused_comm_call_to_first_producer_;
};

class SpreadOutMemcpyFusedCollMutator : ExprMutator {
 public:
  explicit SpreadOutMemcpyFusedCollMutator(const FunctionNode* func, int dp_group_size) : func_(func), dp_group_size_(dp_group_size) {}

  Expr VisitExpr_(const FunctionNode* node) {
    if(!is_main_func) {
      return GetRef<Function>(node);
    }
    // this is the main function
    is_main_func = false;
    auto producer_map = analyzer_.GetProducerMap();
    for(auto param: node->params) {
      HandleFusedCommInputProducer(param, param);
    }
    Array<Var> func_params{node->params};
    auto new_body = this->Mutate(node->body);
    return Function(func_params, new_body, node->ret_type, {}, node->attrs);
  }

  Expr VisitExpr_(const LetNode* node) {
    static auto reduce_scatter_op = Op::Get("raf.op.nccl._reduce_scatter");
    Expr body = node->body;
    do {
      Var var = node->var;
      Expr value = node->value;
      var_idx_map_[var] = current_idx_;
      idx_var_map_[current_idx_] = var;
      idx_expr_map_[current_idx_] = value;
      // 1. first check if this op is a consumer of fused communication output
      // since we need to add zero & memcpy ops before it. Since the output of
      // fused communication op is always a tuple, we only check TGI ops and
      // fused ops
      if(auto tgi_node = value.as<TupleGetItemNode>()) {
        auto tuple_var = Downcast<Var>(tgi_node->tuple);
        
        CHECK(tuple_var.defined());
        if(!var_idx_map_.count(tuple_var)) {
          // the tuple var may be a function argument, directly insert
          scope_.Push(var, value);
        } else {
          int index = var_idx_map_.at(tuple_var);
          CHECK(idx_expr_map_.count(index));
          auto tuple_expr = idx_expr_map_.at(var_idx_map_.at(tuple_var));
          if(analyzer_.IsFusedCommExpr(tuple_expr)) {
            // need to add zero and copy
            int64_t offset, size;
            std::tie(offset, size) = this->analyzer_.GetCommArgOffsetAndSizeAtIdx(tuple_expr, tgi_node->index);
            CHECK(offset >=0 && size >=0);
            DLDataType zeros_type = this->analyzer_.GetCommDataType(tuple_expr);
            CHECK(tgi_node->checked_type().defined() && tgi_node->checked_type().as<TensorTypeNode>());
            Expr zero_expr = CreateZerosOp(Downcast<TensorType>(tgi_node->checked_type()));
            Var zero_var = scope_.Push(zero_expr);
            special_ops_to_stream_[zero_expr] = defuse_tensor_stream_idx;
            // get buffer op
            CHECK(fused_comm_call_to_buffer_var_.count(tuple_expr));
            auto buffer_var = fused_comm_call_to_buffer_var_.at(tuple_expr);
            if (Downcast<Op>(Downcast<Call>(tuple_expr)->op) == reduce_scatter_op) {
              // reduce scatter ops, need to do reverse reorder-partition
              CHECK(offset % dp_group_size_ == 0) << "Reduce scatter offset is not aligned with dp group size.";
              auto tensor_type_node = tgi_node->checked_type().as<TensorTypeNode>();
              CHECK(tensor_type_node) << "Consumer is of non-tensor type: " << value << ".";
              int64_t consumer_size = BytesCompactTensor(tensor_type_node);
              int64_t src_offset = offset / dp_group_size_;
              Expr copy_inplace_expr = CreateCopyInplaceOp(/*dst=*/ zero_var,
                                                /*src=*/ buffer_var, /*size=*/ consumer_size,
                                                /*dst_offset=*/ 0, /*src_offset=*/ src_offset);
              scope_.Push(var, copy_inplace_expr);
              special_ops_to_stream_[copy_inplace_expr] = defuse_tensor_stream_idx;
            } else {
              // allreduce and allgather op, direct copy
              Expr copy_inplace_expr = CreateCopyInplaceOp(/*dst=*/ zero_var,
                                              /*src=*/ buffer_var, /*size=*/ size,
                                              /*dst_offset=*/ 0, /*src_offset=*/ offset);
              // replace the TGI with copy_inplace, reusing its var
              scope_.Push(var, copy_inplace_expr);
              special_ops_to_stream_[copy_inplace_expr] = defuse_tensor_stream_idx;
            }
          } else {
            // it's a normal TGI op, directly insert
            scope_.Push(var, value);
          }
        }
      } else if (value.as<CallNode>() && value.as<CallNode>()->op->IsInstance<FunctionNode>()) {
        // this is a call to fused tuple
        auto call_node = value.as<CallNode>();
        auto func_node = value.as<CallNode>()->op.as<FunctionNode>();
        auto func_expr = Downcast<Function>(value.as<CallNode>()->op);

        // record argument indices which corresponds to a fused comm expr
        std::vector<int> fused_comm_arg_offsets;
        std::vector<Expr> fused_comm_args;
        for(int arg_offset=0; arg_offset < call_node->args.size(); arg_offset++) {
          auto arg_var = call_node->args[arg_offset];
          if(var_idx_map_.count(arg_var)) {
            int index = var_idx_map_.at(arg_var);
            CHECK(idx_expr_map_.count(index));
            auto arg_expr = idx_expr_map_.at(var_idx_map_.at(arg_var));
            if(analyzer_.IsFusedCommExpr(arg_expr)) {
              fused_comm_arg_offsets.push_back(arg_offset);
              fused_comm_args.push_back(arg_expr);
            }
          }
        }

        if(!fused_comm_arg_offsets.empty()) {
          // need to descend into the function body to get tuple index
          std::vector<Expr> fused_func_params;
          ExprMap<int> fused_func_params_map;
          for(auto arg_offset: fused_comm_arg_offsets) {
            fused_func_params.push_back(func_node->params[arg_offset]);
            fused_func_params_map[func_node->params[arg_offset]] = arg_offset;
          }

          ExprMap<std::pair<int, Type>> fused_func_params_to_idx = FusedTGIVisitor(fused_func_params_map).Run(func_node->body);
          CHECK_EQ(fused_func_params_to_idx.size(), fused_func_params.size());
          std::vector<Expr> new_arg_exprs;
          std::vector<Expr> new_arg_vars;
          for(int offset_counter=0; offset_counter<fused_comm_arg_offsets.size(); offset_counter++) {
            int fused_tuple_offset, fused_tuple_size;
            auto fused_comm_tuple_expr = fused_comm_args[offset_counter];
            CHECK(fused_func_params_to_idx.count(fused_func_params[offset_counter]));
            int tgi_index;
            Type tgi_type;
            std::tie(tgi_index, tgi_type) = fused_func_params_to_idx.at(fused_func_params[offset_counter]);
            std::tie(fused_tuple_offset, fused_tuple_size) =
              this->analyzer_.GetCommArgOffsetAndSizeAtIdx(fused_comm_tuple_expr, tgi_index);
            CHECK(fused_tuple_offset >=0 && fused_tuple_size >=0);

            CHECK(tgi_type.as<TensorTypeNode>());
            Expr zero_expr = CreateZerosOp(Downcast<TensorType>(tgi_type));
            Var zero_var = scope_.Push(zero_expr);
            special_ops_to_stream_[zero_expr] = defuse_tensor_stream_idx;
            // get buffer op
            CHECK(fused_comm_call_to_buffer_var_.count(fused_comm_tuple_expr));
            auto buffer_var = fused_comm_call_to_buffer_var_.at(fused_comm_tuple_expr);
            Expr copy_inplace_expr = CreateCopyInplaceOp(/*dst=*/ zero_var,
                                                          /*src=*/ buffer_var, /*size=*/ fused_tuple_size,
                                                          /*dst_offset=*/ 0, /*src_offset=*/ fused_tuple_offset);
            special_ops_to_stream_[copy_inplace_expr] = defuse_tensor_stream_idx;
            Var copy_inplace_var = scope_.Push(copy_inplace_expr, tgi_type);
            new_arg_exprs.push_back(copy_inplace_expr);
            new_arg_vars.push_back(copy_inplace_var);
          }
          auto new_call_node = FusedCallRewriter(fused_comm_arg_offsets, new_arg_vars).Run(value);
          CHECK(new_call_node.defined());
          // replace the original callnode
          scope_.Push(var, new_call_node);
        } else {
          scope_.Push(var, value);
        }
      } else {
        // 2. if this is a fused communication op, replace it with the fused single tensor version
        if(this->analyzer_.IsFusedCommExpr(value)) {
          CHECK(fused_comm_call_to_buffer_var_.count(value)) << "Buffer var not created for fused comm. The ANF may be invalid.";
          auto buffer_var = fused_comm_call_to_buffer_var_.at(value);
          auto comm_call = value.as<CallNode>();
          CHECK(comm_call);

          // construct new comm call
          Array<Expr> comm_args;
          if (Downcast<Op>((Downcast<Call>(value)->op)) == reduce_scatter_op) {
            // static auto reshape_op = Op::Get("raf.op.reshape");
            CHECK(fused_comm_call_to_buffer_size_.count(value)) << "Buffer size not created for fused comm. The ANF may be invalid.";
            int64_t buffer_size_in_bytes = fused_comm_call_to_buffer_size_.at(value);
            CHECK(buffer_size_in_bytes % dp_group_size_ == 0) << "Buffer size must be divisible by dp group size.";
            DLDataType buffer_type = this->analyzer_.GetCommDataType(value);
            int64_t n_buffer_elements = buffer_size_in_bytes * 8 / buffer_type.bits;
            std::vector<int64_t> fused_comm_shape = {n_buffer_elements / dp_group_size_};
            std::vector<int64_t> fused_comm_shape_indices = {1};
            auto fused_comm_buffer_tuple_expr = Tuple({buffer_var});
            auto fused_comm_buffer_tuple_var = scope_.Push(fused_comm_buffer_tuple_expr);
            special_ops_to_stream_[fused_comm_buffer_tuple_expr] = communication_stream_idx;
            // reduce scatter has some extra args
            comm_args.push_back(fused_comm_buffer_tuple_var);
            comm_args.push_back(MakeConstant(ArrayToIntTuple(fused_comm_shape)));
            comm_args.push_back(MakeConstant(ArrayToIntTuple(fused_comm_shape_indices)));
            comm_args.push_back(comm_call->args[3]);
          } else {
            // insert a make tuple for buffer var in comm stream (so no dependency will be added)
            auto fused_comm_buffer_tuple_expr = Tuple({buffer_var});
            auto fused_comm_buffer_tuple_var = scope_.Push(fused_comm_buffer_tuple_expr);
            special_ops_to_stream_[fused_comm_buffer_tuple_expr] = communication_stream_idx;
            comm_args.push_back(fused_comm_buffer_tuple_var);
            for (int i = 1; i < comm_call->args.size(); ++i) {
              comm_args.push_back(comm_call->args[i]);
            }
          }
          auto new_comm_call = Call(comm_call->op, comm_args);
          auto new_comm_call_var = scope_.Push(new_comm_call);
          // since now the comm takes a single input, output is a tensor instead of tuple
          // substitute the buffer var with comm output directly
          fused_comm_call_to_buffer_var_[value] = new_comm_call_var;
        } else {
          // if this is a normal op, directly push
          scope_.Push(var, value);
        }
      }
      // after the above parts we have already pushed the original / substitute ops into the scope

      // 3. check if this op's output is used by any fused comm
      HandleFusedCommInputProducer(var, value);
      current_idx_ ++;

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto ret = scope_.Get(this->Mutate(body));
    return ret;
  }

  Function Run() {
    static auto* dev2str = tvm::runtime::Registry::Get("raf._core.core_utils.dev2str");
    device_ = Device::Current(/*allow_default=*/false);
    CHECK_NE(device_.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
    tvm::String device_str = (*dev2str)(device_);
    dev_string_ = device_str;

    if (!analyzer_.Analyse(func_->body, func_->params)) {
      // no fused collectives found in expr. do nothing.
      return GetRef<Function>(func_);
    }

    return Downcast<Function>(this->VisitExpr_(func_));
  }

  bool HandleFusedCommInputProducer(Var& var, Expr& expr) {
    static auto reduce_scatter_op = Op::Get("raf.op.nccl._reduce_scatter");
    auto successor_comms = this->analyzer_.GetCommFromProducer(expr);
    if (!successor_comms.empty()) {
      for(auto tuple: successor_comms) {
        Expr comm_expr;
        int64_t offset;
        bool is_first_producer;
        std::tie(comm_expr, offset, is_first_producer) = tuple;
        CHECK(comm_expr.defined() && offset >=0) << "Producer info polulated incorrectly in FusedCollectivesAnalyzer.";
        if(is_first_producer) {
          // insert zero op
          int64_t zeros_size = this->analyzer_.GetCommFusedSizeInBytes(comm_expr);
          DLDataType zeros_type = this->analyzer_.GetCommDataType(comm_expr);
          Expr zero_expr = CreateZerosOp(zeros_size, zeros_type);
          Var zero_var = scope_.Push(zero_expr);
          fused_comm_call_to_buffer_var_[comm_expr] = zero_var;
          fused_comm_call_to_buffer_size_[comm_expr] = zeros_size;
          // assign it to fuse tensor stream to reduce the number of syncs
          special_ops_to_stream_[zero_expr] = fuse_tensor_stream_idx;
        } else {
        }
        // insert the copy
        // since we are visiting the let list in order, we must have already visited
        // the first producer.
        CHECK(fused_comm_call_to_buffer_var_.count(comm_expr)) << "Cannot find buffer var for " << comm_expr << " when processing " << expr;
        CHECK(fused_comm_call_to_buffer_size_.count(comm_expr)) << "Cannot find buffer size for " << comm_expr << " when processing " << expr;
        CHECK(expr->checked_type().defined()) << "Producer type is undefined: " << expr << ".";
        auto tensor_type_node = expr->checked_type().as<TensorTypeNode>();
        CHECK(tensor_type_node) << "Producer is of non-tensor type: " << expr << ".";
        int64_t producer_size = BytesCompactTensor(tensor_type_node);
        if (Downcast<Op>(Downcast<Call>(comm_expr)->op) == reduce_scatter_op) {
          // reduce scatter ops, need to do reorder-partition
          CHECK(offset % dp_group_size_ == 0) << "Reduce scatter offset is not aligned with dp group size.";
          int64_t buffer_per_partition_size = fused_comm_call_to_buffer_size_.at(comm_expr) / dp_group_size_;
          int64_t producer_per_partition_size = producer_size / dp_group_size_;
          int64_t per_partition_offset = offset / dp_group_size_;
          for (int partition_id = 0; partition_id < dp_group_size_; partition_id ++) {
            int64_t src_offset = partition_id * producer_per_partition_size;
            int64_t dst_offset = per_partition_offset + partition_id * buffer_per_partition_size;
            Expr copy_inplace_expr = CreateCopyInplaceOp(/*dst=*/ fused_comm_call_to_buffer_var_.at(comm_expr),
                                              /*src=*/ var, /*size=*/ producer_per_partition_size,
                                              /*dst_offset=*/ dst_offset, /*src_offset=*/ src_offset);
            Var copy_inplace_var = scope_.Push(copy_inplace_expr);
            special_ops_to_stream_[copy_inplace_expr] = fuse_tensor_stream_idx;
            fused_comm_call_to_buffer_var_[comm_expr] = copy_inplace_var;
          }
        } else {
          // allreduce and allgather ops, directly copy the tensor to the buffer at offset
          Expr copy_inplace_expr = CreateCopyInplaceOp(/*dst=*/ fused_comm_call_to_buffer_var_.at(comm_expr),
                                                        /*src=*/ var, /*size=*/ producer_size,
                                                        /*dst_offset=*/ offset, /*src_offset=*/ 0);
          Var copy_inplace_var = scope_.Push(copy_inplace_expr);
          special_ops_to_stream_[copy_inplace_expr] = fuse_tensor_stream_idx;
          // replace the buffer with copy output
          fused_comm_call_to_buffer_var_[comm_expr] = copy_inplace_var;
        }
      }
      return true;
    }
    return false;
  }

  int64_t GetStreamForSpecialOps(const Expr& expr) {
    if(special_ops_to_stream_.count(expr)) {
      return special_ops_to_stream_.at(expr);
    } else {
      return unknown_stream_idx;
    }
  }

  ExprMap<int64_t> GetSpecialStreamMapping() {
    return special_ops_to_stream_;
  }

 protected:
  class FusedTGIVisitor : public ExprVisitor {
  public:
    FusedTGIVisitor(ExprMap<int>& fused_params) : fused_params_(fused_params) {}

    void VisitExpr_(const TupleGetItemNode* node) {
      ExprVisitor::VisitExpr_(node);
      if(fused_params_.count(node->tuple)) {
        int tgi_index = node->index;
        CHECK(!(results_.count(node->tuple) && results_.at(node->tuple).first != tgi_index))
          << "Not Implemented: Fused param is used more than once in prim func.";
        CHECK(node->checked_type().defined()) << "TGI does not have a checked type in prim func.";
        results_[node->tuple] = std::make_pair(tgi_index, node->checked_type());
      }
    }

    ExprMap<std::pair<int, Type>> Run(const Expr& expr) {
      VisitExpr(expr);
      return results_;
    }

    ExprMap<int>& fused_params_;
    ExprMap<std::pair<int, Type>> results_;
  };

  class FusedCallRewriter : public ExprMutator {
  public:
    FusedCallRewriter(std::vector<int>& arg_indices_to_rewrite, std::vector<Expr>& new_args): 
      arg_indices_to_rewrite_(arg_indices_to_rewrite), new_args_(new_args) {
        CHECK_EQ(arg_indices_to_rewrite.size(), new_args.size());
      }

    Expr VisitExpr_(const VarNode* node) {
      if(params_info_map.count(GetRef<Var>(node))) {
        // rewrite its shape
        auto args_idx = params_info_map.at(GetRef<Var>(node));
        CHECK(args_idx_to_input_idx.count(args_idx));
        auto new_type = Downcast<Var>(new_args_[args_idx_to_input_idx.at(args_idx)])->type_annotation;
        CHECK(new_type.defined()) << "New args type must be specified.";
        auto new_var = raf::ir::MakeVar(node->name_hint(), new_type);
        return new_var;
      }
      return GetRef<Var>(node);
    }

    Expr VisitExpr_(const TupleGetItemNode* node) {
      if(params_info_map.count(node->tuple)) {
        // directly pass the old tuple var (which corresponds to the new input)
        CHECK(node->tuple->IsInstance<VarNode>());
        auto ret = VisitExpr(node->tuple);
        CHECK(ret.defined());
        return ret;
      }
      return ExprMutator::VisitExpr_(node);
    }

    Expr VisitExpr_(const FunctionNode* node) {
      for(auto idx: arg_indices_to_rewrite_) {
        CHECK(idx < node->params.size());
        params_info_map[node->params[idx]] = idx;
      }
      Array<Var> new_params;
      for(auto param: node->params) {
        new_params.push_back(Downcast<Var>(this->Mutate(param)));
      }
      auto new_body = this->Mutate(node->body);
      return Function(new_params, new_body, Type(), {}, node->attrs, node->span);
    }

    Expr VisitExpr_(const CallNode* node) {
      if(GetRef<Expr>(node) == main_expr) {
        Array<Expr> new_args;
        int next_rewrite_arg_idx = 0;
        for(int i=0; i<node->args.size(); i++) {
          if(next_rewrite_arg_idx < arg_indices_to_rewrite_.size() && i == arg_indices_to_rewrite_[next_rewrite_arg_idx]) {
            CHECK(new_args_[next_rewrite_arg_idx].defined());
            new_args.push_back(new_args_[next_rewrite_arg_idx]);
            next_rewrite_arg_idx++;
          } else {
            new_args.push_back(node->args[i]);
          }
        }
        auto new_op = this->Mutate(node->op);
        return Call(new_op, new_args);
      }
      return ExprMutator::VisitExpr_(node);
    }

    Expr Run(Expr& expr) {
      main_expr = expr;
      for(int i=0; i< arg_indices_to_rewrite_.size(); i++) {
        args_idx_to_input_idx[arg_indices_to_rewrite_[i]] = i;
      }
      return this->Mutate(expr);
    }

    std::vector<int>& arg_indices_to_rewrite_;
    std::unordered_map<int, int> args_idx_to_input_idx;
    std::vector<Expr>& new_args_;
    ExprMap<int> params_info_map;
    Expr main_expr;
  };

  inline Expr CreateZerosOp(int64_t size_in_bytes, DLDataType dtype) {
    std::vector<int64_t> one_d_shape = {(size_in_bytes * 8) /  dtype.bits};
    return CreateZerosOp(one_d_shape, dtype);
  }

  inline Expr CreateZerosOp(TensorType type) {
    std::vector<int64_t> shape;
    for (auto axis : type->shape) {
      auto node = axis.as<IntImmNode>();
      CHECK(node != nullptr) << "Axis " << axis << " is not IntImmNode";
      int64_t axis_ = node->value;
      shape.push_back(axis_);
    }
    return CreateZerosOp(shape, type->dtype);
  }

  inline Expr CreateZerosOp(std::vector<int64_t>& shape, DLDataType dtype) {
    static auto zeros = Op::Get("raf.op.zeros");
    Array<Expr> args({MakeConstant(ArrayToIntTuple(shape)),
                      MakeConstant(StringValue::make(tvm::runtime::DLDataType2String(dtype))),
                      MakeConstant(StringValue::make(dev_string_))});
    return Call(zeros, args);
  }

  inline Expr CreateCopyInplaceOp(Var& dst_tensor, Var& src_tensor, int64_t size,
                            int64_t dst_offset, int64_t src_offset) {
    static auto copy_inplace_op = Op::Get("raf.op.copy_inplace");
    Array<Expr> args({dst_tensor, src_tensor, MakeConstant(IntValue::make(DataType::Int(64), size)),
                      MakeConstant(IntValue::make(DataType::Int(64), dst_offset)), MakeConstant(IntValue::make(DataType::Int(64), src_offset))});
    return Call(copy_inplace_op, args);
  }

 private:
  Device device_;
  int dp_group_size_;
  std::string dev_string_;
  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
  // maps each var to its index
  ExprMap<int> var_idx_map_;
  ExprMap<int> expr_idx_map_;
  std::map<int, Var> idx_var_map_;
  std::map<int, Expr> idx_expr_map_;

  ExprMap<Var> fused_comm_call_to_buffer_var_;
  ExprMap<int64_t> fused_comm_call_to_buffer_size_;
  ExprMap<int64_t> special_ops_to_stream_;

  bool is_main_func = true;
  const FunctionNode* func_;
  LetList scope_;
  FusedCollectivesAnalyzer analyzer_;
}; // class SpreadOutMemcpyFusedCollMutator

class SyncAnalyzer : ExprVisitor {
 public:
  /*! \brief Analyse the needed dependency edges between ops. It stores the analysis result in
   * add_event_after_op, wait_event_before_op and set_stream_before_op.
   *
   * \returns true if we need to add any dependency edge (i.e. if collective communication ops
   * are found in expr).
   */
  bool Analyse(const Expr& expr, const ExprMap<int64_t>& stream_override_map = {}) {
    stream_override_map_ = stream_override_map;

    VisitExpr(expr);

    DepSet dep_set;
    GetDepSet_(dep_set);
    CreateEventsUsingDepSet_(dep_set);

    return !dep_set.empty();
  }

  void VisitExpr_(const VarNode* var) {
    var_idx_map_[GetRef<Var>(var)] = current_idx_;
  }

  void VisitExpr_(const CallNode* call) {
    auto call_expr = GetRef<Expr>(call);
    UpdateStreamInfo_(call_expr);
    UpdateDependencyInfo_(call_expr, call->args);
  }

  void VisitExpr_(const TupleNode* tuple) {
    auto tuple_expr = GetRef<Expr>(tuple);
    UpdateStreamInfo_(tuple_expr);
    UpdateDependencyInfo_(tuple_expr, tuple->fields);
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item) {
    auto tuple_get_item_expr = GetRef<Expr>(tuple_get_item);
    UpdateStreamInfo_(tuple_get_item_expr);
    UpdateDependencyInfo_(tuple_get_item_expr, {tuple_get_item->tuple});
  }

  void VisitExpr_(const LetNode* op) {
    auto pre_visit = [this](const LetNode* op) {
      Var var = op->var;
      Expr value = op->value;
      max_idx_ = std::max(current_idx_, max_idx_);
      expr_idx_map[value] = current_idx_;
      var_idx_map_[var] = current_idx_;
      this->VisitExpr(var);
      this->VisitExpr(value);
      current_idx_++;
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = op->body;
      max_idx_ = std::max(current_idx_, max_idx_);
      this->VisitExpr(body);
      current_idx_--;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  int GetUniqueEventId() {
    return next_unique_event_id_++;
  }

  // maps expr to id
  ExprMap<int> expr_idx_map;

  std::unordered_map<int, int> idx_stream_map;

  // maps expr id to added event_id
  std::unordered_map<int, std::vector<int>> add_event_after_op;
  // maps expr id to event_id to wait for
  std::unordered_map<int, std::vector<int>> wait_event_before_op;
  // maps expr id to whether it needs an set stream op before it
  std::unordered_map<int, bool> set_stream_before_op;

 private:
  // Determines if a set_stream op should be added before the op.
  // a set_stream op is needed if the executing stream of the previous op
  // (previous_op_stream_idx_) is different from the executing stream
  // of the current op. It also updates previous_op_stream_idx_.
  void UpdateStreamInfo_(const Expr& op) {
    int expected_stream_idx;
    if(stream_override_map_.count(op)) {
      expected_stream_idx = stream_override_map_.at(op);
    } else {
      expected_stream_idx = IdentifyStream(op);
    }
    idx_stream_map[current_idx_] = expected_stream_idx;
    if (previous_op_stream_idx_ == -1 || previous_op_stream_idx_ != expected_stream_idx) {
      // if the current op is the first op or last op's stream is not the stream for current op
      set_stream_before_op[current_idx_] = true;
      previous_op_stream_idx_ = expected_stream_idx;
    }
  }

  // This function is called once for every op during ANF expansion. It updates the
  // last input producer and first output consumer in each stream for each op.
  // The last input producer map is updated as we iterate through the input
  // arguments. The last input producer map will contain the correct value after
  // we iterate through every op, since we must have called the function on
  // the true "first consumer" once.
  void UpdateDependencyInfo_(const Expr& op, const Array<Expr>& input_args) {
    for (auto arg : input_args) {
      if (var_idx_map_.count(arg)) {
        // update producer map and consumer map
        int arg_idx = var_idx_map_[arg];
        int expr_idx = current_idx_;
        UpdateLastProducerMap_(expr_idx, arg_idx);
        UpdateFirstConsumerMap_(arg_idx, expr_idx);
      }
    }
  }

  void UpdateLastProducerMap_(int idx, int producer) {
    int producer_stream = idx_stream_map[producer];
    if (producer_stream == idx_stream_map[idx]) {
      // neglect producers in the same stream
      return;
    }
    if (!last_producer_idx_map_.count(idx)) {
      last_producer_idx_map_.emplace(idx, std::unordered_map<int, int>{});
    }
    if (!last_producer_idx_map_[idx].count(producer_stream)) {
      last_producer_idx_map_[idx].emplace(producer_stream, producer);
    }
    if (producer > last_producer_idx_map_[idx][producer_stream]) {
      last_producer_idx_map_[idx][producer_stream] = producer;
    }
  }

  void UpdateFirstConsumerMap_(int idx, int consumer) {
    int consumer_stream = idx_stream_map[consumer];
    if (consumer_stream == idx_stream_map[idx]) {
      // neglect consumers in the same stream
      return;
    }
    if (!first_consumer_idx_map_.count(idx)) {
      first_consumer_idx_map_.emplace(idx, std::unordered_map<int, int>{});
    }
    if (!first_consumer_idx_map_[idx].count(consumer_stream)) {
      first_consumer_idx_map_[idx].emplace(consumer_stream, consumer);
    }
    if (consumer < first_consumer_idx_map_[idx][consumer_stream]) {
      first_consumer_idx_map_[idx][consumer_stream] = consumer;
    }
  }

  void GetDepSet_(DepSet& dep_set) {
    dep_set.clear();
    for (int call = 0; call < max_idx_ + 1; ++call) {
      for (auto stream_and_producer : last_producer_idx_map_[call]) {
        CHECK(stream_and_producer.first != unknown_stream_idx) << "Still have unknown stream.";
        dep_set.insert(std::make_pair(stream_and_producer.second, call));
      }
      for (auto stream_and_consumer : first_consumer_idx_map_[call]) {
        CHECK(stream_and_consumer.first != unknown_stream_idx) << "Still have unknown stream.";
        dep_set.insert(std::make_pair(call, stream_and_consumer.second));
      }
    }
    PruneReduntantDependency_(dep_set);
  }

  void PruneReduntantDependency_(DepSet& dep_set) {
    if (dep_set.empty()) {
      return;
    }

    // In multi-stream dependency analysis, we consider these two kinds of dependencies:
    // (1) dependencies between ops in different streams
    // (2) dependencies between ops in the same stream
    // Before pruning, we add back dependencies of the second kind.
    std::unordered_map<int, std::vector<int>> stream_id_map;
    for (int i = 0; i < max_idx_; ++i) {
      int stream = idx_stream_map[i];
      if (!stream_id_map.count(stream)) {
        stream_id_map.emplace(stream, std::vector<int>{});
      }
      stream_id_map[stream].push_back(i);
    }
    for (auto& stream_and_ids : stream_id_map) {
      auto& ids = stream_and_ids.second;
      std::sort(ids.begin(), ids.end());
      for (int i = 0; i < ids.size() - 1; ++i) {
        dep_set.insert(std::make_pair(ids[i], ids[i + 1]));
      }
    }

    std::unordered_map<int, std::unordered_set<int>> direct_children_map = {};
    std::unordered_map<int, std::unordered_set<int>> direct_parent_map = {};
    std::unordered_set<int> roots = {};
    for (auto& dep_pair : dep_set) {
      roots.insert(dep_pair.first);
      direct_children_map[dep_pair.first].insert(dep_pair.second);
      direct_parent_map[dep_pair.second].insert(dep_pair.first);
    }
    for (auto& dep_pair : dep_set) {
      auto it = roots.find(dep_pair.second);
      if (it != roots.end()) {
        roots.erase(it);
      }
    }

    std::vector<int> post_dfs_order = {};
    std::unordered_map<int, bool> visited = {};
    for (auto root : roots) {
      DFS_(direct_children_map, visited, root, post_dfs_order);
    }

    // Prune redundant dependencies. One dependency (u, v) is redundant if and only if
    // there exists a path from u to v that does not go through the dependency (u, v).
    // Therefore, if dependency (u, v) is redundant, node v should be both direct and
    // indirect child of node u. When pruning, we firstly iterate through the dependency
    // graph in post DFS order to collect direct and indirect children of each node,
    // and then remove redundant edges from the graph.
    std::unordered_map<int, std::unordered_set<int>> indirect_children_map = {};
    for (auto node : post_dfs_order) {
      for (auto parent : direct_parent_map[node]) {
        indirect_children_map[parent].insert(direct_children_map[node].begin(),
                                             direct_children_map[node].end());
        indirect_children_map[parent].insert(indirect_children_map[node].begin(),
                                             indirect_children_map[node].end());
      }
    }
    DepSet deps_to_remove = {};
    for (auto node : post_dfs_order) {
      for (auto direct_child : direct_children_map[node]) {
        if (indirect_children_map[node].count(direct_child)) {
          deps_to_remove.insert(std::make_pair(node, direct_child));
        }
      }
    }

    for (auto& dep_to_remove : deps_to_remove) {
      if (dep_set.count(dep_to_remove)) {
        dep_set.erase(dep_set.find(dep_to_remove));
      }
    }

    // The pruned graph should only contains dependencies of the first kind.
    // Here we remove dependencies of the second kind.
    for (auto it = dep_set.begin(); it != dep_set.end();) {
      if (InSameStream_(it->first, it->second)) {
        it = dep_set.erase(it);
      } else {
        ++it;
      }
    }
  }

  void DFS_(std::unordered_map<int, std::unordered_set<int>>& direct_children_map,
            std::unordered_map<int, bool>& visited, int node, std::vector<int>& post_dfs_order) {
    if (!visited[node]) {
      for (auto child : direct_children_map[node]) {
        DFS_(direct_children_map, visited, child, post_dfs_order);
      }
      post_dfs_order.push_back(node);
      visited[node] = true;
    }
  }

  bool InSameStream_(int i, int j) {
    return idx_stream_map[i] == idx_stream_map[j];
  }

  void CreateEventsUsingDepSet_(DepSet& dep_set) {
    for (auto& dep_pair : dep_set) {
      add_event_after_op[dep_pair.first].push_back(GetUniqueEventId());
      wait_event_before_op[dep_pair.second].push_back(add_event_after_op[dep_pair.first].back());
    }
  }

  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
  // max "depth" of expr
  int max_idx_ = 0;
  // stream index of the previous op
  int previous_op_stream_idx_ = -1;
  // maps each var & expr to its index
  ExprMap<int> var_idx_map_;

  // custom expr to stream assignment
  ExprMap<int64_t> stream_override_map_;

  // counter to generate unique event ids.
  int next_unique_event_id_ = 1;

  // index refers to the order of an op in the input ANF expression.
  // map each Call, Tuple or TupleGetItem op to the index of the last producer of the op's inputs
  std::unordered_map<int, std::unordered_map<int, int>> last_producer_idx_map_;
  // map each Call, Tuple or TupleGetItem op to the index (in the input ANF expr) of the first
  // consumer of the op's outputs
  std::unordered_map<int, std::unordered_map<int, int>> first_consumer_idx_map_;
};

class SyncEnforcer : ExprVisitor {
 public:
  /*!
   * This pass works in ANF and adds necessary synchronization ops (i.e., set_stream, add_event,
   * wait_event) between communication ops and computation ops to ensure correctness. It does not
   * alter the execution order of ops and assumes single stream computation execution (i.e.
   * sequential stream schedule)
   *
   * Specifically,
   *    1. It inserts a set_stream(device_id, stream_idx) if the op requires the stream be switched
   *       before its execution.
   *
   *    2. It inserts an add_event(unique_event_id, stream_idx) after an op if the following op
   * depends on it executes on a different stream.
   *
   *    3. It inserts a wait_event(unique_event_id, stream_idx) before an op if the previous op it
   * depends on executes on a different stream.
   *
   * Example 1:
   *    Input data flow graph (number represent order in ANF):
   *
   *      atan (1) -> allreduce (2) -> atan(3)
   *
   *    Corresponding IR:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %a1 = raf.op.atan(%x);
   *        let %a2 = (%a1,);
   *        let %a3 = raf.op._allreduce(%a2, str"sum");
   *        let %a4 = raf.op.atan(%a3);
   *        %a4
   *      }
   *
   *    After Transformation:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %a1 = raf.op.atan(%x);
   *        let %a2 = (%a1,);
   *        let %add_event_comp = raf.op.add_event(int64(1), int64(1));
   *        let %set_stream_comm = raf.op.set_stream(int64(0), int64(5));
   *        let %wait_for_comp = raf.op.wait_event(int64(1), int64(5));
   *        let %a3 = raf.op._allreduce(%a2, str"sum");
   *        let %add_event_comm = raf.op.add_event(int64(2), int64(5));
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %wait_for_comm = raf.op.wait_event(int64(2), int64(1));
   *        let %a4 = raf.op.atan(%a3);
   *        %a4
   *      }
   *
   * Example 2:
   *    Input data flow graph (number represent order in ANF):
   *
   *     atan (1) -> atan (2) -> allreduce (3) -> mul (5) -> concat (6)
   *        \                                           /
   *         -------------->  allreduce (4) ----------->
   *
   *    Here since allreduce(3) is executed before allreduce(4), and
   *    allreduce(3) depends on atan(2) which is guarenteed to executed
   *    after atan(1). Therefore the dependency atan(1) -> allreduce(4)
   *    is not necessary and we will not add the corresponding events.
   *
   *    Corresponding IR:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %v = raf.op.atan(%x);
   *        let %v1 = (%v,);
   *        let %v2 = raf.op.atan(%v);
   *        let %v3 = (%v2,);
   *        let %v4 = raf.op._allreduce(%v3, str"sum");
   *        let %v5 = raf.op._allreduce(%v1, str"sum");
   *        let %v6 = raf.op.atan(%v4);
   *        let %v7 = (%v6, %v5);
   *        let %v8 = raf.op.concatenate(%v7, int64(0));
   *        %v8
   *      }
   *
   *    After Transformation:
   *
   *      fn (%x: Tensor[(64, 128), float32]) {
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %v = raf.op.atan(%x);
   *        let %v1 = (%v,);
   *        let %v2 = raf.op.atan(%v);
   *        let %v3 = (%v2,);
   *        let %add_event_comp = raf.op.add_event(int64(1), int64(1));
   *        let %set_stream_comm = raf.op.set_stream(int64(0), int64(5));
   *        let %wait_for_comp = raf.op.wait_event(int64(1), int64(5));
   *        let %v4 = raf.op._allreduce(%v3, str"sum");
   *        let %add_event_comm = raf.op.add_event(int64(2), int64(5));
   *        let %v5 = raf.op._allreduce(%v1, str"sum");
   *        let %add_event_comm1 = raf.op.add_event(int64(3), int64(5));
   *        let %set_stream_comp = raf.op.set_stream(int64(0), int64(1));
   *        let %wait_for_comm = raf.op.wait_event(int64(2), int64(1));
   *        let %v6 = raf.op.atan(%v4);
   *        let %wait_for_comm1 = raf.op.wait_event(int64(3), int64(1));
   *        let %v7 = (%v6, %v5);
   *        let %v8 = raf.op.concatenate(%v7, int64(0));
   *        %v8
   *      }
   */
  explicit SyncEnforcer(const FunctionNode* func) : func_(func) {
  }

  void VisitExpr_(const LetNode* op) {
    Var orig_var = op->var;
    Expr orig_value = op->value;
    int orig_value_idx = analyzer_.expr_idx_map[orig_value];

    SetStreamAndWaitEvent(orig_value_idx);
    ell_->Push(orig_var, orig_value);
    AddEvent(orig_value_idx);

    ell_->ret = ell_->vars.back();
    VisitExpr(op->body);
  }

  Function Run(const ExprMap<int64_t>& stream_override_map = {}) {
    auto device = Device::Current(/*allow_default=*/false);
    CHECK_NE(device.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
    device_id_ = device.device_id();
    CHECK_EQ(device_id_, DistContext::Global()->local_rank) << "Current device id != local rank.";

    if (!analyzer_.Analyse(func_->body, stream_override_map)) {
      // no collectives found in expr. do nothing.
      return GetRef<Function>(func_);
    }

    ell_ = std::make_unique<ExplicitLetList>();
    VisitExpr(func_->body);

    return Function(func_->params, ell_->AsExpr(), {}, {});
  }

 protected:
  Expr CreateSetStreamOp(int64_t device_id, int64_t stream_id) {
    static Op set_stream_op = Op::Get("raf.op.set_stream");
    return CreateSetStreamOrEventOp_(set_stream_op, device_id, stream_id);
  }

  Expr CreateAddEventOp(int64_t event_id, int64_t stream_id) {
    static Op add_event_op = Op::Get("raf.op.add_event");
    return CreateSetStreamOrEventOp_(add_event_op, event_id, stream_id);
  }

  Expr CreateWaitEventOp(int64_t event_id, int64_t stream_id) {
    static Op wait_event_op = Op::Get("raf.op.wait_event");
    return CreateSetStreamOrEventOp_(wait_event_op, event_id, stream_id);
  }

  inline void AddEvent(int idx) {
    if (analyzer_.add_event_after_op.count(idx)) {
      int stream = analyzer_.idx_stream_map[idx];
      std::string add_event_var_name = stream_name_hint[stream] + "_add_event";
      for (auto event_id : analyzer_.add_event_after_op[idx]) {
        Var add_event_var = raf::ir::MakeVar(add_event_var_name, {});
        Expr add_event_value = CreateAddEventOp(event_id, stream);
        ell_->Push(add_event_var, add_event_value);
      }
    }
  }

  inline void SetStreamAndWaitEvent(int idx) {
    int stream = analyzer_.idx_stream_map[idx];
    std::string var_name_hint = stream_name_hint[stream];
    if (analyzer_.set_stream_before_op[idx]) {
      std::string set_stream_var_name = var_name_hint + "_set_stream";
      Var set_stream_var = raf::ir::MakeVar(set_stream_var_name, {});
      Expr set_stream_value = CreateSetStreamOp(device_id_, stream);
      ell_->Push(set_stream_var, set_stream_value);
    }
    if (analyzer_.wait_event_before_op.count(idx)) {
      std::string wait_event_var_name = var_name_hint + "_wait_event";
      for (auto event_id : analyzer_.wait_event_before_op[idx]) {
        Var wait_event_var = raf::ir::MakeVar(wait_event_var_name, {});
        Expr wait_event_value = CreateWaitEventOp(event_id, stream);
        ell_->Push(wait_event_var, wait_event_value);
      }
    }
  }

 private:
  Expr CreateSetStreamOrEventOp_(Op& op, int64_t first_arg, int64_t second_arg) {
    Expr first_arg_expr = MakeConstant(value::ScalarValue::make(first_arg));
    Expr second_arg_expr = MakeConstant(value::ScalarValue::make(second_arg));
    Array<Expr> args({first_arg_expr, second_arg_expr});
    return Call(op, args);
  }

  int device_id_ = -1;
  const FunctionNode* func_;
  SyncAnalyzer analyzer_;
  std::unique_ptr<ExplicitLetList> ell_;
};
}  // namespace enforce_sync

TVM_REGISTER_PASS_CONFIG_OPTION("raf.enforce_sync.on_the_fly_gradient_copy", Bool);

Pass EnforceSync(int dp_group_size) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    bool use_otf_gradient_copy = static_cast<bool>(
        pc->GetConfig("raf.enforce_sync.on_the_fly_gradient_copy", Bool(true)).value());
    if(const char* use_group_rs_ag_str = getenv("RAF_GROUPED_COLLECTIVE")) {
      auto use_group_rs_ag = std::atoi(use_group_rs_ag_str);
      if (use_group_rs_ag > 0) {
        use_otf_gradient_copy = false;
        LOG(INFO) << "RAF_GROUPED_COLLECTIVE is set to true, otf gradient copy disabled.";
      }
    }
    if (use_otf_gradient_copy) {
      enforce_sync::SpreadOutMemcpyFusedCollMutator mutator(f.operator->(), dp_group_size);
      auto fuse_mutated_func = Downcast<Function>(DeadCodeElimination(mutator.Run()));
      auto stream_mapping = mutator.GetSpecialStreamMapping();
      auto new_func = enforce_sync::SyncEnforcer(fuse_mutated_func.operator->()).Run(stream_mapping);
      return new_func;
    }
    return enforce_sync::SyncEnforcer(f.operator->()).Run();
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 0, "EnforceSync", {});
  PassInfo pass_info(0, "EnforceSync", {});
  return RAFSequential({InferType(), func_pass, EraseType()}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.EnforceSync").set_body_typed(EnforceSync);

}  // namespace pass
}  // namespace raf
