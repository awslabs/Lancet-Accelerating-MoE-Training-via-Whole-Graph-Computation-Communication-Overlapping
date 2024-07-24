/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file memory_plan.cc
 * \brief Optimized allocated memory in the IR.
 */
#include <algorithm>
#include <random>
#include <vector>

#include "raf/op.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "raf/pass.h"
#include "raf/stream_pool.h"
#include "./let_list.h"
#include "./liveness_analysis.h"
#include "tvm/relay/attrs/memory.h"

namespace raf {
namespace pass {

namespace memory_plan {

using namespace raf::ir;
using namespace raf::value;

template <typename T>
using StdMap = std::unordered_map<Var, T, ObjectPtrHash, ObjectPtrEqual>;
using VSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using StreamVSet = std::unordered_map<int, VSet>;

/*! \brief A tensor group. The intermediate tensors which liveness dummy tensors in a group
 * can be allocated to the same memory buffer.
 */
struct TensorGroup {
  TensorGroup(Var storage, int64_t alignment) : storage(storage), alignment(alignment) {
  }

  /*! \brief The map from dummy tensors created by livensss analysis to its let binding var
   * and required storage size.
   */
  StdMap<std::pair<Var, int64_t>> members;

  std::unordered_map<int, std::vector<Var>> members_live_ended_in_stream;

  /*! \brief The binded variable for the allocated storage. */
  Var storage;

  /*! \brief The buffer size that should be allocated for this group. -1 means the size is
   * dynamic. In this case, this group should only have one tensor.
   */
  int64_t size = 0;

  /*! \brief The alignment of this group. */
  int64_t alignment;
};

/*! \brief A list of tensor groups with manipulation utilities. */
class TensorGroups {
 public:
  /*! \brief A list of storage allocation groups. */
  std::vector<TensorGroup> groups;
  std::vector<std::unordered_set<int>> used_streams;

  TensorGroups(liveness_analysis::LivenessAnalyzer* analyzer) : analyzer_(analyzer) {
    auto output_tensor_vars = analyzer_->GetOutputTensorVars();
    for (const auto& it : output_tensor_vars) {
      for (const auto& v: it.second) {
        dummy_out_vars_.insert(v);
      }
    }
  }

  void CalcUsedStreams() {
    for (const auto& group : groups) {
      std::unordered_set<int> streams;
      for (const auto& it: group.members) {
        auto vset = analyzer_->GetLiveLines(it.second.first);
        for (const auto& live_line: analyzer_->GetLiveLines(it.second.first)) {
          streams.insert(live_line.first);
        }
      }
      used_streams.push_back(streams);
    }
    CHECK(used_streams.size() == groups.size());
  }

  /*! \brief Get the dummy output tensor in liveness analyzer. */
  Var GetTensorVar(const Var& let_var) {
    auto target_vars = analyzer_->GetTensorVars(let_var);
    CHECK_EQ(target_vars.size(), 1U) << "Failed to get var for " << let_var;
    return target_vars[0];
  }

  bool HasTensorVar(const Var& let_var) {
    auto target_vars = analyzer_->GetTensorVars(let_var);
    return target_vars.size() > 0;
  }

  /*! \brief Check whether the tensor is a final output. */
  bool IsOutputTensor(const Var& tensor_var) {
    auto target_var = GetTensorVar(tensor_var);
    return dummy_out_vars_.find(target_var) != dummy_out_vars_.end();
  }

  /*! \brief Find the group ID that the target var belongs to, or -1 if not found. */
  int FindGroupIdByMember(const Var& let_var) {
    const Var target_var = GetTensorVar(let_var);
    for (size_t j = 0; j < groups.size(); ++j) {
      if (groups[j].members.find(target_var) != groups[j].members.end()) {
        return j;
      }
    }
    return -1;
  }

  /*! \brief Find the group ID that the target storage var belongs to, or -1 if not found. */
  int FindGroupIdByStorageVar(const Var& storage_var) {
    for (size_t j = 0; j < groups.size(); ++j) {
      if (groups[j].storage == storage_var) {
        return j;
      }
    }
    return -1;
  }

  /*! \brief Find valid tensor groups of the given tensor and its alignment. */
  std::vector<int> FindValidGroups(const Var& let_var, const int64_t alignment) {
    // Live in vars.
    auto live_in_vars = analyzer_->GetLiveVars(let_var);
    VSet flattened_live_in_vars;
    for (auto it: live_in_vars) {
      flattened_live_in_vars.insert(it.second.begin(), it.second.end());
    }

    // Find valid tensor groups.
    std::vector<int> candidates;
    for (size_t i = 0; i < groups.size(); ++i) {
      bool valid = true;
      for (const auto& member : groups[i].members) {
        if (groups[i].size == -1 || flattened_live_in_vars.count(member.first) > 0 ||
            groups[i].alignment != alignment) {
          // This group is invalid for this tensor to join if:
          // 1. Its size is dynamic,
          // 2. One of its members appears at the live-in set, or
          // 3. Its alignment is not the same as the current tensor.
          valid = false;
          break;
        }
      }
      if (valid) {
        candidates.push_back(i);
      }
    }
    return candidates;
  }

  /*! \brief Join the given tensor group. */
  void JoinGroup(size_t group_id, const Var& let_var, int64_t size = 0) {
    const Var target_var = GetTensorVar(let_var);
    groups[group_id].members[target_var] = std::make_pair(let_var, size);
    groups[group_id].size = (groups[group_id].size > size) ? groups[group_id].size : size;
  }

  /*! \brief Create a new group and return its ID. */
  int CreateGroup(const Var& storage_var, int64_t alignment) {
    groups.emplace_back(TensorGroup(storage_var, alignment));
    return groups.size() - 1;
  }

  /*! \brief Remove the tensor from the group, update the storage size, and return the size of
   * the removed tensor.
   */
  int64_t RemoveFromGroup(size_t group_id, const Var& let_var) {
    const Var target_var = GetTensorVar(let_var);
    CHECK_GT(groups[group_id].members.count(target_var), 0U);
    auto storage_nbytes = groups[group_id].members[target_var].second;
    if (groups[group_id].size == storage_nbytes) {
      // The storage size of this group may be reduced due to the removal of this tensor.
      int64_t max_size = 0;
      for (auto kv : groups[group_id].members) {
        if (kv.first == target_var) {
          continue;
        }
        max_size = (kv.second.second > max_size) ? kv.second.second : max_size;
      }
      groups[group_id].size = max_size;
    }
    groups[group_id].members.erase(target_var);
    return storage_nbytes;
  }

  /*! \brief Calculate the total memory footprint in MBs. */
  float GetTotalMemoryMBs() {
    float total = 0.0;
    for (size_t j = 0; j < groups.size(); ++j) {
      total += groups[j].size;
    }
    return total / 1024.0 / 1024.0;
  }

  /*!\ brief Log tensor groups with details for debugging. */
  std::string DebugDumpGroups() {
    std::stringstream ss1;
    for (const auto& group : groups) {
      std::stringstream ss2;
      for (const auto member : group.members) {
        ss2 << member.second.first->name_hint() << "(" << member.second.second << "), ";
      }
      ss1 << "Storage " << group.storage->name_hint() << ", size " << group.size
          << ", members: " << ss2.str() << std::endl;
    }
    return ss1.str();
  }

 private:
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief Dummy vars that output tensors map to. */
  VSet dummy_out_vars_;
};

class TensorTracker {
  public:
    TensorTracker(const Expr& body, const Array<Var>& params): ell_(ExplicitLetList::make(body)) {
      for (const auto& param : params) {
        var_alloc_map_[param] = param;
      }
      for (int i=0; i < ell_->vars.size(); ++i) {
        auto var = ell_->vars[i];
        auto expr = ell_->exprs[i];
        if (auto call = expr.as<CallNode>()) {
          if (call->op.as<OpNode>()->name == "raf.op.vm.set_shape") {
            auto tensor_var = Downcast<Var>(call->args[0]);
            if (var_alloc_map_.count(tensor_var)) {
              var_alloc_map_[var] = var_alloc_map_[tensor_var];
            } else {
              CHECK(alloc_tensor_vars_.count(tensor_var));
              var_alloc_map_[var] = tensor_var;
            }
          } else if (call->op.as<OpNode>()->name == "raf.op.vm.alloc_tensor") {
            alloc_tensor_vars_.insert(var);
          }
        } else if (auto tuple = expr.as<TupleNode>()) {
          for (int tuple_idx = 0; tuple_idx < tuple->fields.size(); ++tuple_idx) {
            if (tuple->fields[tuple_idx].as<ConstantNode>()) {
              tuple_alloc_map_[var].push_back(Var());
            } else {
              auto field_var = Downcast<Var>(tuple->fields[tuple_idx]);
              if (var_alloc_map_.count(field_var)) {
                tuple_alloc_map_[var].push_back(var_alloc_map_[field_var]);
              } else if (tuple_alloc_map_.count(field_var)) {
                tuple_alloc_map_[var].push_back(field_var);
              } else {
                CHECK(alloc_tensor_vars_.count(field_var)) << "Cannot find " << field_var;
                tuple_alloc_map_[var].push_back(field_var);
              }
            }
          }
        } else if (auto tgi = expr.as<TupleGetItemNode>()) {
          Var tuple_var = Downcast<Var>(tgi->tuple);
          if (var_alloc_map_.count(tuple_var)) {
            var_alloc_map_[var] = var_alloc_map_[tuple_var];
          } else {
            CHECK(tuple_alloc_map_.count(tuple_var));
            int index = tgi->index;
            CHECK(index < tuple_alloc_map_[tuple_var].size());
            auto tensor_var = tuple_alloc_map_[tuple_var][index];
            if (tensor_var->checked_type().as<TensorTypeNode>()) {
              var_alloc_map_[var] = tensor_var;
            } else {
              tuple_alloc_map_[var] = tuple_alloc_map_[tensor_var];
            }
          }
        } else if (auto expr_var = expr.as<VarNode>()) {
          // direct assignment
          auto tensor_var = Downcast<Var>(expr);
          if (var_alloc_map_.count(tensor_var)) {
            var_alloc_map_[var] = var_alloc_map_.at(tensor_var);
          } else {
            CHECK(alloc_tensor_vars_.count(tensor_var));
            var_alloc_map_[var] = tensor_var;
          }
        }
      }
      // create reverse map
      for (int i=0; i < ell_->vars.size(); ++i) {
        auto var = ell_->vars[i];
        auto expr = ell_->exprs[i];
        if (auto call = expr.as<CallNode>()) {
          if (call->op.as<OpNode>()->name == "raf.op.vm.set_shape") {
            auto tensor_var = var_alloc_map_.at(var);
            tensor_var_to_reshape_vars_[tensor_var].push_back(var);
          }
        }
      }
    }

    Array<Var> GetReshapeVarsFromTensorVar(Var var) const {
      if (tensor_var_to_reshape_vars_.count(var)) {
        return tensor_var_to_reshape_vars_.at(var);
      } else {
        return {};
      }
    }

private:
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_alloc_map_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> alloc_tensor_vars_;
  std::unordered_map<Var, Array<Var>, ObjectPtrHash, ObjectPtrEqual> tuple_alloc_map_;
  std::unordered_map<Var, Array<Var>, ObjectPtrHash, ObjectPtrEqual> tensor_var_to_reshape_vars_;
};

/*! \brief A mutator to perform the following tasks:
 * 1. Run tensor grouper to group the tensors generated by alloc_tensor according to
 *    the liveness analysis. All tensors in a tensor group will use the same storage.
 * 2. Mutate alloc_tensor to use storage of the group it belongs to.
 * 3. Mutate alloc_storage to allocate sufficient memory, which is required by
 *    the largest tensor in the group.
 * 4. Remove alloc_storages that do not be used by any group.
 * 5. Insert free(%x) to free tensor/storage %x at the end of its life-cycle.
 */
class MemoryPlanner : public ExprMutator {
 public:
  MemoryPlanner(const Function& func, liveness_analysis::LivenessAnalyzer* analyzer)
      : func_(func), analyzer_(analyzer), tensor_groups_(Group()), tensor_tracker_(func->body, func->params) {
    scopes_.emplace_back(new LetList);
    tensor_groups_.CalcUsedStreams();
  }

  Expr Run() {
    static const Op& add_event_op = Op::Get("raf.op.add_event");
    // LOG(INFO) << "Tensor groups:";
    // LOG(INFO) << tensor_groups_.DebugDumpGroups();

    // LOG(INFO) << "Live In sets: ";
    // LOG(INFO) << analyzer_->DebugDumpLiveIn();

    auto device = Device::Current(/*allow_default=*/false);
    CHECK_NE(device.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
    device_id_ = device.device_id();

    // scan through the body to find max currently used event id
    auto ell = ExplicitLetList::make(func_->body);
    for (auto expr: ell->exprs) {
      if (auto call = expr.as<CallNode>()) {
        if (auto op = call->op.as<OpNode>()) {
          if (GetRef<Op>(op) == add_event_op) {
            auto event_id = call->args[0].as<ConstantNode>()->value.as<IntValueObj>()->value;
            next_event_id_ = std::max(next_event_id_, event_id + 1);
          }
        }
      }
    }

    // List storage vars that will be used by one or more tensors.
    for (const auto& group : tensor_groups_.groups) {
      if (group.members.size() > 0) {
        used_storages_.insert(group.storage);
      }
    }

    // insert all function parameters into live_tensors_
    for(auto param: func_->params) {
      if (param->checked_type().as<TensorTypeNode>()) {
        auto live_lines = analyzer_->GetLiveLines(param);
        for (auto it: live_lines) {
          live_tensors_[it.first].insert(param);
        }
      }
    }

    // Mutate the function.
    return this->Mutate(func_);
  }

  Expr VisitExpr_(const LetNode* node) {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      curr_let_ = node->var;
      // LOG(INFO) << "Processing " << curr_let_ << " ...";

      // Free allocated tensors that will not be used anymore.
      auto live_in_vars = analyzer_->GetLiveVars(curr_let_);
      // std::stringstream ss;
      // ss << "Live in vars: \n";
      // for (auto it: live_in_vars) {
      //   ss << "Stream: " << it.first << "\nVars: ";
      //   for (auto var: it.second) {
      //     ss << var << ", ";
      //   }
      //   ss << "\n";
      // }
      // LOG(INFO) << ss.str();
      std::unordered_set<int> streams;
      for (auto& stream_it: live_in_vars) {
        streams.insert(stream_it.first);
      }
      for (int stream: streams) {
        auto& live_tensors_stream_ = live_tensors_[stream];
        auto& live_in_vars_stream = live_in_vars[stream];
        auto it = live_tensors_stream_.begin();
        while (it != live_tensors_stream_.end()) {
          const Var target_var = tensor_groups_.GetTensorVar(*it);
          if (live_in_vars_stream.find(target_var) == live_in_vars_stream.end()) {
            auto group_id = tensor_groups_.FindGroupIdByMember(*it);
            // If a tensor does not belong to any group, meaning that we its storage is not
            // allocated by us (i.e., parameters), so simply remove it from the live set.
            if (group_id != -1) {
              // Note that we do not need to insert vm.free for tensors as only the storage
              // should hold the memory pointer.
              CHECK(group_id < tensor_groups_.groups.size());
              CHECK(group_id < tensor_groups_.used_streams.size());
              auto& group = tensor_groups_.groups[group_id];
              auto& used_streams = tensor_groups_.used_streams[group_id];
              if (used_streams.size() == 1) {
                // user on a single stream, follow default behavior
                // LOG(INFO) << "Single Stream: Remove " << *it << " from group " << group_id;
                tensor_groups_.RemoveFromGroup(group_id, *it);
                if (group.members.size() == 0) {
                  // LOG(INFO) << "Free storage " << group.storage;
                  scope->Push(MakeFreeMemory(group.storage));
                }
              } else {
                // LOG(INFO) << "Multi Stream: Mark " << *it << " as ended in stream " << stream;
                group.members_live_ended_in_stream[stream].push_back(*it);
                // If all members have ended their lifetime in this stream,
                // we can register the free.
                if (group.members_live_ended_in_stream[stream].size() == group.members.size()) {
                  // LOG(INFO) << "All usage of storage " << group.storage << " in stream " << stream << " has ended";
                  // scope->Push(MakeFreeMemory(group.storage));
                  Expr add_event_expr;
                  int64_t event_id;
                  std::tie(add_event_expr, event_id) = RegisterEvent(stream);
                  scope->Push(add_event_expr);
                  var_to_mem_free_events_[group.storage].push_back(event_id);
                }
                // if all members have ended their lifetime in all streams,
                // we can free the storage
                bool all_streams_done = true;
                for (auto stream: used_streams) {
                  if (group.members_live_ended_in_stream[stream].size() != group.members.size()) {
                    all_streams_done = false;
                    break;
                  }
                }
                if (all_streams_done) {
                  // free the memory 
                  // LOG(INFO) << "All usage of storage " << group.storage << " has ended in all streams";
                  auto exprs = MakeFreeMemory(group.storage, var_to_mem_free_events_[group.storage], current_stream_);
                  for (auto expr: exprs) {
                    scope->Push(expr);
                  }
                }
              }
            } else {
              // check if this is an extra parameter
              auto name = std::string((*it)->name_hint());
              if (name.rfind("extra", 0) == 0) {
                scope->Push(MakeForceFreeMemory((*it)));
              }
            }
            it = live_tensors_stream_.erase(it);
          } else {
            it++;
          }
        }
      }

      auto value = VisitExpr(node->value);

      // Determine whether the alloc_storage is never used and should be dismissed.
      bool dismiss_storage = false;
      if (const auto& node = value.as<CallNode>()) {
        const auto* op_node = node->op.as<OpNode>();
        if (op_node && GetRef<Op>(op_node) == alloc_storage_op &&
            used_storages_.find(curr_let_) == used_storages_.end()) {
          dismiss_storage = true;
        }
      }
      if (!dismiss_storage) {
        expr_map_.emplace(node->var, value);
        scope->Push(node->var, value);
      }

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto new_body = VisitExpr(body);
    auto ret = scopes_.back()->Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) final {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("raf.op.vm.alloc_tensor");
    static const Op& reshape_tensor_op = Op::Get("raf.op.vm.set_shape");
    static const Op& set_stream_op = Op::Get("raf.op.set_stream");
    const auto* op_node = node->op.as<OpNode>();

    auto call = GetRef<Call>(node);
    if (op_node && GetRef<Op>(op_node) == alloc_storage_op) {
      // Mutate the size of alloc_storage indicated by the tensor group.

      auto group_id = tensor_groups_.FindGroupIdByStorageVar(curr_let_);
      if (group_id != -1 && tensor_groups_.groups[group_id].size > 0) {
        Array<Expr> new_args;
        for (auto& arg : call->args) {
          new_args.push_back(VisitExpr(arg));
        }
        new_args.Set(0, MakeConstant(ScalarValue::make(tensor_groups_.groups[group_id].size)));
        return Call(alloc_storage_op, new_args);
      }
    } else if (op_node && GetRef<Op>(op_node) == alloc_tensor_op) {
      // Reassign alloc_tensor to the alloc_storage indicated by the tensor group.
      auto live_lines = analyzer_->GetLiveLines(curr_let_);
      for (auto it: live_lines) {
        live_tensors_[it.first].insert(curr_let_);
      }

      Array<Expr> new_args;
      for (auto& arg : call->args) {
        new_args.push_back(VisitExpr(arg));
      }

      // Find the tensor group of this alloc_tensor and check whether this tensor needs
      // to be re-assigned to another storage.
      auto group_id = tensor_groups_.FindGroupIdByMember(curr_let_);
      CHECK_NE(group_id, -1) << "Internal error: output tensor of " << curr_let_->name_hint()
                             << " does not belong to any tensor group";
      auto storage_var = Downcast<Var>(call->args[0]);
      if (tensor_groups_.groups[group_id].storage != storage_var) {
        DLOG(INFO) << "Assign " << curr_let_->name_hint() << " to "
                   << tensor_groups_.groups[group_id].storage->name_hint() << " from "
                   << storage_var->name_hint();
        new_args.Set(0, tensor_groups_.groups[group_id].storage);
      }

      // Override the own memory flag argument.
      bool is_output_tensor = tensor_groups_.IsOutputTensor(curr_let_);
      Array<Var> reshape_vars = tensor_tracker_.GetReshapeVarsFromTensorVar(curr_let_);
      if (reshape_vars.size() > 0) {
        for (auto var: reshape_vars) {
          if (tensor_groups_.IsOutputTensor(var)) {
            is_output_tensor = true;
            break;
          }
        }
      }
      auto own = MakeConstant(ScalarValue::make(is_output_tensor));
      if (call->args.size() == 4) {
        new_args.push_back(own);
      } else {
        new_args.Set(4, own);
      }

      return Call(alloc_tensor_op, new_args);
    } else if (op_node && GetRef<Op>(op_node) == reshape_tensor_op) {
      // Other ops that will also create a new tensor/view. We do not need to mutate them,
      // but have to trace their life-cycle to know the right place of inserting free storage.
      auto live_lines = analyzer_->GetLiveLines(curr_let_);
      for (auto it: live_lines) {
        live_tensors_[it.first].insert(curr_let_);
      }
    } else if (op_node && GetRef<Op>(op_node) == set_stream_op) {
      current_stream_ = node->args[1].as<ConstantNode>()->value.as<IntValueObj>()->value;
    }
    return ExprMutator::VisitExpr_(node);
  }

 private:
  class TensorGrouper;

  TensorGroups Group();

  inline std::pair<Expr, int64_t> RegisterEvent(int64_t stream_id) {
    static const Op& add_event_op = Op::Get("raf.op.add_event");
    int64_t event_id = next_event_id_;
    next_event_id_++;
    return std::make_pair(Call(add_event_op, {MakeConstant(ScalarValue::make(event_id)), MakeConstant(ScalarValue::make(stream_id))}), event_id);
  }

  inline Expr MakeFreeMemory(const Var& memory_var) {
    static const Op& op = Op::Get("raf.op.vm.free");
    return Call(op, {memory_var});
  }

  inline std::vector<Expr> MakeFreeMemory(const Var& memory_var, std::vector<int64_t> event_ids, int64_t current_stream) {
    static const Op& set_stream_op = Op::Get("raf.op.set_stream");
    static const Op& wait_event_op = Op::Get("raf.op.wait_event");
    static const Op& free_op = Op::Get("raf.op.vm.free");
    std::vector<Expr> ret;
    int64_t target_stream = stream_pool::StreamTagEnum::AsyncMemFree();
    Expr set_stream = Call(set_stream_op, {MakeConstant(ScalarValue::make(static_cast<int64_t>(device_id_))),
                                           MakeConstant(ScalarValue::make(target_stream))});
    ret.push_back(set_stream);
    // wait for all events
    for (auto event_id: event_ids) {
      Expr wait_event = Call(wait_event_op, {MakeConstant(ScalarValue::make(event_id)), MakeConstant(ScalarValue::make(target_stream))});
      ret.push_back(wait_event);
    }
    // free memory
    Expr free_memory = Call(free_op, {memory_var});
    ret.push_back(free_memory);
    // set stream back
    Expr set_stream_back = Call(set_stream_op, {MakeConstant(ScalarValue::make(static_cast<int64_t>(device_id_))),
                                                 MakeConstant(ScalarValue::make(current_stream))});
    ret.push_back(set_stream_back);
    return ret;
  }

  inline Expr MakeForceFreeMemory(const Var& memory_var) {
    static const Op& op = Op::Get("raf.op.vm.force_free");
    return Call(op, {memory_var});
  }

 private:
  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The functio to be optimized. */
  const Function& func_;
  /*! \brief device id */
  int device_id_;
  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief A map from let varr to its expression. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief A list of storage allocation groups. */
  TensorGroups tensor_groups_;
  /*! \brief current stream id */
  int current_stream_ = -1;
  /*! \brief next event id */
  int64_t next_event_id_ = -1;
  /*! \brief Used storage vars. */
  VSet used_storages_;
  /*! \brief Current live tensor vars. */
  StreamVSet live_tensors_;
  /*! \brief maps a var to mem free event ids */
  StdMap<std::vector<int64_t>> var_to_mem_free_events_;
  TensorTracker tensor_tracker_;
};

/*! \brief A visitor to group tensors generated by alloc_tensor according to
 * the liveness analysis. A tensor can join an existing group (i.e., can reuse the storage
 * of the group) if the group:
 * 1) has no other tensors in the live-in set of the current tensor,
 * 2) has the same the alignment, and
 * 3) has the closest storage size as the current tensor.
 */
class MemoryPlanner::TensorGrouper : public ExprVisitor {
 public:
  TensorGrouper(const Expr& body, liveness_analysis::LivenessAnalyzer* analyzer)
      : analyzer_(analyzer), tensor_groups_(analyzer), ell_(ExplicitLetList::make(body)) {
    CHECK(analyzer_->IsSuccess());
  }

  TensorGroups Run() {
    const auto& vars = ell_->vars;
    const auto& exprs = ell_->exprs;
    CHECK_EQ(vars.size(), exprs.size());
    size_t n = exprs.size();

    for (size_t i = 0; i < n; ++i) {
      expr_map_[vars[i]] = exprs[i];
    }

    for (int i = 0; i < n; ++i) {
      curr_let_ = vars[i];
      ExprVisitor::VisitExpr(exprs[i]);
    }

    return tensor_groups_;
  }

  void VisitExpr_(const CallNode* node) override {
    static const Op& alloc_storage_op = Op::Get("raf.op.vm.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("raf.op.vm.alloc_tensor");
    static const Op& reshape_tensor_op = Op::Get("raf.op.vm.set_shape");
    const auto* op_node = node->op.as<OpNode>();

    if (GetRef<Op>(op_node) == alloc_tensor_op) {
      for (auto& arg : node->args) {
        VisitExpr(arg);
      }

      // Get the storage size in bytes and alignment of this tensor.
      auto storage_var = Downcast<Var>(node->args[0]);
      auto storage_node = expr_map_[storage_var].as<CallNode>();
      CHECK(storage_node && Downcast<Op>(storage_node->op) == alloc_storage_op)
          << "Expected alloc_storage as the first arg of alloc_tensor, but got "
          << Downcast<Op>(storage_node->op)->name;

      // Alignment.
      CHECK(storage_node->args[1].as<ConstantNode>())
          << "The alignment of alloc_storage is not a constant";
      auto align_val = storage_node->args[1].as<ConstantNode>()->value;
      CHECK(align_val->IsInstance<IntValueObj>());
      int64_t alignment = align_val.as<IntValueObj>()->value;

      // Storage size.
      int64_t storage_nbytes = -1;
      if (storage_node->args[0].as<ConstantNode>()) {
        auto size_val = storage_node->args[0].as<ConstantNode>()->value;
        CHECK(size_val->IsInstance<IntValueObj>());
        storage_nbytes = size_val.as<IntValueObj>()->value;
      }

      // Create a new tensor group.
      auto cand_group_id = tensor_groups_.CreateGroup(storage_var, alignment);
      DLOG(INFO) << "Create a new group " << cand_group_id << " for " << storage_var->name_hint();
      tensor_groups_.JoinGroup(cand_group_id, curr_let_, storage_nbytes);
    } else if (GetRef<Op>(op_node) == reshape_tensor_op) {
      // set_shape creates a new tensor view so it has to be considered as a new tensor too.
      for (auto& arg : node->args) {
        VisitExpr(arg);
      }
      auto tensor_var = Downcast<Var>(node->args[0]);
      auto group_id = tensor_groups_.FindGroupIdByMember(tensor_var);

      // If its original tensor is not in a tensor group, it means the tensor is a parameter,
      // which is not allocated by us so we do not need to track it.
      if (group_id != -1) {
        // Put this tensor to the samee group as its original tensor.
        DLOG(INFO) << curr_let_->name_hint() << " joins group " << group_id;
        tensor_groups_.JoinGroup(group_id, curr_let_);
      }
    } else {
      ExprVisitor::VisitExpr_(node);
    }
  }

  /*! \brief The current processing let var. */
  Var curr_let_;
  /*! \brief The let list. */
  std::unique_ptr<ExplicitLetList> ell_{nullptr};
  /*! \brief A map from let varr to its expression. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief The liveness analyzer, including liveness analysis results. */
  liveness_analysis::LivenessAnalyzer* analyzer_;
  /*! \brief A list of storage allocation groups. */
  TensorGroups tensor_groups_;
};

TensorGroups MemoryPlanner::Group() {
  return TensorGrouper(func_, analyzer_).Run();
}

}  // namespace memory_plan

TVM_REGISTER_PASS_CONFIG_OPTION("raf.memory_plan.dump_liveness_stat", Bool);

Pass MemoryPlan() {
  PassContext pass_ctx = PassContext::Current();
  Bool dump_stat = pass_ctx->GetConfig("raf.memory_plan.dump_liveness_stat", Bool(false)).value();
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    f = EliminateRudundantShare(f.operator->());
    auto analyzer = liveness_analysis::LivenessAnalyzer(f);
    auto live_in = analyzer.Run();
    if (!analyzer.IsSuccess()) {
      LOG(WARNING) << "Memory planning is disabled because liveness analysis was failed";
      return f;
    }
    // if (dump_stat) {
    //   liveness_analysis::DumpLivenessStat(live_in);
    // }
    return Downcast<ir::Function>(memory_plan::MemoryPlanner(f, &analyzer).Run());
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 3, "MemoryPlan", {});
  PassInfo pass_info(3, "MemoryPlan", {});
  return RAFSequential({InferType(), func_pass}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.MemoryPlan").set_body_typed(MemoryPlan);

}  // namespace pass
}  // namespace raf
