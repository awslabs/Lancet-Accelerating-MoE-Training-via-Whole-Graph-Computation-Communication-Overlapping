/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2021 by Contributors
 * \file data_parallel_schedule.cc
 * \brief Schedules ops during data parallel training.
 */
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <tvm/node/serialization.h>
#include <relay/transforms/pass_utils.h>
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/device.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/dist_context.h"
#include "raf/pass.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/analysis.h"
#include "raf/serialization.h"
#include "../common.h"
#include "../let_list.h"
#include "../stream_schedule.h"
#include "./scheduler_utils.h"
#include "raf/stream_pool.h"
#include "./scheduler_common.h"
#include "./extended_dfg.h"
#include "./lancet_optimization.h"
#include "./profile_utils.h"
#include "./lerp.h"

#define WARMUP_ITER 20
#define WARMUP_THRESHOLD_RATIO 0.5
#define TRUNCATE_PERCENT 6.25

namespace raf {
namespace pass {
namespace data_parallel_schedule {

using namespace raf::ir;
using namespace raf::analysis;
using op::IsCollectiveOp;
using namespace raf::pass::scheduler_common;
using namespace raf::pass::extended_dfg;
using LinkNode = tvm::support::LinkNode<Node*>;
using LinkedList = tvm::relay::LinkedList<Node*>;
using stream_schedule::StreamSchedulerBase;
using namespace raf::profiler;
using namespace pass::lancet_optimization;
using namespace pass::profile_utils;
using TimeType = double;
using pass::scheduler_utils::fixOrder;
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::BytesCompactType;

std::string DebugDumpDependencyGraphWithAttrs(DependencyGraph* dg, NodeMap<TimeType>& exec_time_map, NodeMap<std::string>& node_name_map, NodeMap<CommComponents>& comm_size_map,
                                              NodeMap<NodeType>& node_type_map) {
  std::ostringstream debug_str;
  NodeExprMap node_expr;
  std::unordered_map<Node*, int> node_idx;
  for (auto& it : dg->expr_node) {
    node_expr[it.second] = it.first;
  }
  for(auto node : dg->post_dfs_order) {
    node_idx[node] = (int)node_idx.size();
  }

  std::vector<std::vector<int>> edges;
  for (auto node : dg->post_dfs_order) {
    int parent_idx = node_idx[node];
    for (auto child_iter = node->children.head; child_iter; child_iter = child_iter->next) {
      int child_idx = node_idx[child_iter->value];
      edges.push_back({child_idx, parent_idx});
    }
  }
  // # nodes, # edges, # cost models
  // debug_str << node_idx.size() << " " << edges.size() << " " << cost_model.size() << std::endl;
  // pcomp, pmem, pcomm 
  // debug_str << "0 0 " << comm_throughput << std::endl;
  // omatmul, oelemwise, oothers, ocomm
  // debug_str << "0 0 0 " << comm_overhead << std::endl;
  // for(auto it: cost_model) {
  //   // directly print the op name. 
  //   // op keys also starts with op name so it's easy to match cost model
  //   // with nodes in simulator
  //   debug_str << it.first << " " << it.second->overhead
  //                         << " " << it.second->throughput << std::endl;
  // }
  // nodes
  for (auto node : dg->post_dfs_order) {
    // node_id >> node_name >> node_type >>
    // exec_time >> flops(0) >> memory(0) >> scale_factor(1) >> comm_size >>
    // visibility;
    // node_idx, node str
    Expr parent = node_expr[node];
    debug_str << node_idx[node] << " ";
    bool is_visible = false;
    if (node_name_map.count(node)) {
      is_visible = true;
      debug_str << node_name_map.at(node);
    } else {
      if(parent.defined()) {
        debug_str << parent->GetTypeKey();
      } else {
        debug_str << "Dummy";
      }
    }
    CHECK(node_type_map.count(node));
    bool is_comm_node = node_type_map.at(node) == NodeType::kCommNode;
    debug_str << ( is_comm_node ? " Comm " : " Comp ");
    CHECK(exec_time_map.count(node));
    debug_str << exec_time_map.at(node) << " 0 0 1 ";
    if(is_comm_node) {
      CHECK(comm_size_map.count(node));
      CHECK_EQ(comm_size_map[node].size(), 1) << "Only unfused dependency graph is supported.";
      size_t comm_size;
      for(auto it: comm_size_map[node]) {
        comm_size = it.second;
      }
      debug_str << comm_size << " 1" << std::endl;
    } else {
      debug_str << "0 "<< static_cast<int>(is_visible) << std::endl;
    }
  }
  for (auto& edge : edges) {
    // node_idx, node str
    debug_str << edge[0] << " " << edge[1] << " 0" << std::endl;
  }
  return debug_str.str();
}

class FixLoadedOpType : public ExprMutator {
  Expr VisitExpr_(const OpNode* op_node) final {
    auto op = GetRef<Op>(op_node);
    if(op::IsDialectOp(op)) {
      auto base_op = op::GetBaseOp(op);
      op->op_type = base_op->op_type;
    }
    return op;
  }
};

class TVMZerosToCudaZeros : public ExprMutator {
  Op cuda_zeros_op = op::GetDialectOp("raf.op.cuda.zeros");
  Op tvm_zeros_op = op::GetDialectOp("raf.op.tvm.zeros");
  Expr VisitExpr_(const OpNode* op_node) final {
    auto op = GetRef<Op>(op_node);
    if(op == tvm_zeros_op) {
      return cuda_zeros_op;
    }
    return op;
  }
};

// this class transforms all primitive functions back to GNF
class PrimFunctionRestorer : public ExprMutator {
public:
  Expr Run(Expr e) {
    if(auto func_node = e.as<FunctionNode>()) {
      // input is main function, mutate body
      auto new_body = VisitExpr(func_node->body);
      return WithFields(GetRef<Function>(func_node), func_node->params, new_body);
    }
    return VisitExpr(e);
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    // Convert the function into GNF
    auto mod = IRModule::FromExpr(GetRef<Function>(func_node));
    mod = ToGraphNormalForm()(mod);
    auto new_func = Downcast<Function>(mod->Lookup("main"));
    return new_func;
  }
};

// this class fuses communication ops based on a predefined bucket size
class SizeBasedCommFusor : public ExprMutator {
public:
  Expr VisitExpr_(const CallNode* call) override {
    if (call->op.as<OpNode>() && Downcast<Op>(call->op) == allreduce_op_) {
      // check if it needs to be fused
      if (orig_expr_to_fused_tgi_.count(GetRef<Call>(call))) {
        // return the fused call
        return orig_expr_to_fused_tgi_.at(GetRef<Call>(call));
      }
    }
    return ExprMutator::VisitExpr_(call);
  }

  Expr Run(Expr e) {
    size_t group_size = 0;
    if(const char* group_size_str = getenv("LANCET_COMM_PREFUSE_BUCKET_SIZE")) {
      group_size = std::stol(std::string(group_size_str));
    }
    if (group_size == 0) {
      return e;
    }
    //create the data flow graph
    Arena arena;
    DependencyGraph dfg = CreateDependencyGraph(&arena, e, /*prune_atomic_nodes=*/true);
    // map each node in the dependency graph to the expression it represents
    NodeExprMap node_expr;
    // ready queue for all ops that directly depends on a communication op
    std::queue<Node*> comm_successor_ready_queue;
    // ready queue for all other ops
    std::queue<Node*> ready_queue;
    // counter that keeps track of the number of each op's current unscheduled predecessors
    // the dependency graph in tvm is a data flow graph with edge direction reversed, so we
    // use out-degree here instead of in-degree.
    std::unordered_map<Node*, int> out_degree;
    // keeps track of whether an op directly depends on a communication op
    std::unordered_set<Node*> comm_successor_nodes;

    for (auto& it : dfg.expr_node) {
      node_expr[it.second] = it.first;
    }
    std::vector<Node*>& nodes = dfg.post_dfs_order;

    // calculate out-degree for each node and populate comm_successor_nodes map
    for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); node_it++) {
      out_degree[(*node_it)] = 0;
      if (auto call_node = node_expr[*node_it].as<CallNode>()) {
        if (IsCollectiveOp(call_node->op)) {
          // record direct successor nodes of communication op
          for (auto parent = (*node_it)->parents.head; parent; parent = parent->next) {
            comm_successor_nodes.insert(parent->value);
          }
        }
      }

      for (auto child = (*node_it)->children.head; child; child = child->next) {
        out_degree[(*node_it)]++;
      }
    }
    // push nodes with zero predecessors into the queue
    for (auto& node : nodes) {
      if (out_degree[node] == 0) {
        ready_queue.push(node);
      }
    }

    std::vector<Expr> comm_expr_order;
    // in each step, we pop an op out of the queue, add it to the ANF and
    // push all its ready successors into the corresponding ready queue
    auto process_queue_element = [&](std::queue<Node*>& q) {
      while (!q.empty()) {
        Node* node = q.front();
        Expr expr = node_expr.at(node);
        if (auto call_node = expr.as<CallNode>()) {
          if (call_node->op == allreduce_op_) {
            // we only handle allreduce currently
            comm_expr_order.push_back(expr);
          }
        }
        for (auto parent = node->parents.head; parent; parent = parent->next) {
          out_degree[parent->value]--;
          if (out_degree[parent->value] == 0) {
            if (comm_successor_nodes.count(parent->value)) {
              comm_successor_ready_queue.push(parent->value);
            } else {
              ready_queue.push(parent->value);
            }
          }
        }
        q.pop();
      }
    };

    while (!ready_queue.empty() || !comm_successor_ready_queue.empty()) {
      process_queue_element(ready_queue);
      process_queue_element(comm_successor_ready_queue);
    }

    // decide which comm operators to fuse
    int processed_ops = 0;
    size_t current_accum_size = 0;
    std::vector<Expr> current_comms_to_fuse;
    for (auto expr: comm_expr_order) {
        CHECK(expr->checked_type().as<TensorTypeNode>()) << "Encountered fused comm op " << expr << " before running size based fusion.";
        current_comms_to_fuse.push_back(expr);
        size_t size = BytesCompactTensor(expr->checked_type().as<TensorTypeNode>());
        current_accum_size += size;
        if (current_accum_size >= group_size || processed_ops == comm_expr_order.size() - 1) {
            if(current_comms_to_fuse.size() > 1) {
                LOG(INFO) << "Prefusion: Fusing " << current_comms_to_fuse.size() << " comm nodes.";
                auto fused_expr = FuseCollectives(current_comms_to_fuse);
                for(int i=0; i< current_comms_to_fuse.size(); i++) {
                    auto expr_to_fuse = current_comms_to_fuse[i];
                    orig_expr_to_fused_tgi_[expr_to_fuse] = GetFusedCollectiveTGI(fused_expr, i);
                }
            }
            current_comms_to_fuse.clear();
            current_accum_size = 0;
        }
        processed_ops ++;
    }
    return pass::InferType(pass::DeadCodeElimination(this->Mutate(e)));
  }

protected:
  Expr FuseCollectives(const std::vector<Expr>& exprs) {
    CHECK(exprs.size() > 1) << "Cannot fuse less than 2 collectives.";
    Array<Expr> args;
    for(auto expr: exprs) {
        auto call_node = expr.as<CallNode>();
        CHECK(call_node);
        auto input_tuple = call_node->args[0];
        CHECK(input_tuple.as<TupleNode>());
        CHECK(input_tuple.as<TupleNode>()->fields.size() == 1);
        args.push_back(Downcast<Tuple>(input_tuple)->fields[0]);
    }
    Expr fused_args = Tuple(args);
    Array<Expr> fused_call_args = {fused_args};
    for (size_t i = 1; i < exprs[0].as<CallNode>()->args.size(); i++) {
        fused_call_args.push_back(exprs[0].as<CallNode>()->args[i]);
    }
    return Call(allreduce_op_, fused_call_args);
  }

  Expr GetFusedCollectiveTGI(const Expr& fused_expr, int index) {
    return TupleGetItem(fused_expr, index);
  }

private:
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> orig_expr_to_fused_tgi_;
  Op allreduce_op_ = op::GetDialectOp("raf.op.nccl._allreduce");
};

// base class for all profile-based schedulers
class ProfiledCUDAScheduler : public StreamSchedulerBase {
public:
  explicit ProfiledCUDAScheduler(bool is_simulation=false) : is_simulation_(is_simulation) {
    if(const char* load_optimized_module_path = getenv("LOAD_OPTIMIZED_MODULE_FROM")) {
      // don't need to load profile
      return;
    }
    if (is_simulation) {
      // do nothing, must manually load profile from file
      return;
    }
    std::unordered_map<OpKey, std::vector<uint64_t>> profile_time_map;
    // get profiler
    auto profile_stats = Profiler::Get()->GetProfileStats();
    if(profile_stats.empty()) {
      LOG(WARNING) << "Profile stats are empty.";
      has_profile_ = false;
    } else {
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
        if(start_time_set && end_time_set) {
          OpKey key = GenerateOpKeyFromProfileStats(stat);
          CHECK_LE(start_time, end_time);
          profile_time_map[key].push_back(end_time - start_time);
          has_profile_ = true;
        }
       }
    }
    bool warning_issued = false;
    for(auto& it: profile_time_map) {
      // throw away first WARMUP_ITER iterations of data if WARMUP_ITER is smaller than total iter*WARMUP_THRESHOLD_RATIO
      // throw away top TRUNCATE_PERCENT% and bottom TRUNCATE_PERCENT% points & calculate avg
      std::sort(it.second.begin(), it.second.end());
      int num_data = it.second.size();
      int init_offset = 0;
      if(num_data >= WARMUP_ITER / WARMUP_THRESHOLD_RATIO) {
        init_offset = WARMUP_ITER;
        num_data -= WARMUP_ITER;
      } else {
        if (!warning_issued) {
          LOG(WARNING) << "Number of profile data is small. Result can be unstable.";
          warning_issued = true;
        }
      }
      // LOG(INFO) << "Using " << num_data << " profiling data points for " << it.first;
      int num_disgrad_each_end = (int)(num_data * (TRUNCATE_PERCENT / 100.0));
      double avg = 0;
      for(int i = init_offset + num_disgrad_each_end; i < init_offset + num_data - num_disgrad_each_end; i++) {
        avg += it.second[i];
      }
      profile_time_map_[it.first] = static_cast<uint64_t>(avg / (num_data - 2 * num_disgrad_each_end));
    }
    SynchronizeProfile_();
  }

  bool HasProfile() { return has_profile_; }

  virtual Expr Schedule(Expr e) {
    CHECK(has_profile_) << "No profile data is found. Did you forget to load profile from file?";
    return e;
  }

  Expr LoadProfiledModuleFromFile(std::string fn_prefix) {
    auto mod_main = LoadExprFromFile(fn_prefix + ".mod");
    // load profile
    std::string profile_dump_path(fn_prefix);
    profile_dump_path += ".profile";
    LOG(INFO) << "Loading profile from " << profile_dump_path;
    std::ifstream profile_dump_file;
    profile_dump_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    profile_dump_file.open(profile_dump_path, std::ios::binary | std::ios::ate);
    std::streamsize profile_size = profile_dump_file.tellg();
    profile_dump_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(profile_size);
    CHECK(profile_dump_file.read(buffer.data(), profile_size)) << "Failed to read profile data";
    profile_time_map_ = DeserializeProfile_(buffer.data(), profile_size);
    has_profile_ = true;
    LOG(INFO) << "Successfully loaded profile.";
    return mod_main;
  }

  static Expr LoadExprFromFile(std::string fn_path) {
    // load module expr
    std::string module_dump_path(fn_path);
    LOG(INFO) << "Loading IR from " << module_dump_path;
    std::ifstream module_dump_file;
    module_dump_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    module_dump_file.open(module_dump_path);
    std::stringstream module_file_buffer;
    module_file_buffer << module_dump_file.rdbuf();
    auto expr = Downcast<Expr>(ir::serialization::LoadJSON(module_file_buffer.str()));
    auto type_infered_expr = pass::InferType(expr);
    auto mod = IRModule::FromExpr(type_infered_expr);
    auto mod_main = TVMZerosToCudaZeros().Mutate(mod->Lookup("main"));
    auto optype_fixed_mod_main = FixLoadedOpType().Mutate(mod_main);
    LOG(INFO) << "Successfully loaded module.";
    return optype_fixed_mod_main;
  }

protected:
  void CheckAndDumpProfiledExpr(Expr e) {
    if(const char* profile_dump_file = getenv("DEBUG_DUMP_PROFILE_PREFIX")) {
      SerializeProfiledModule(e, profile_dump_file);
    }
  }

  uint64_t GetExecTime(Call& op) {
    if (!has_profile_) return 0;
    OpKey key = GenerateOpKeyFromCallOp(op, profile_time_map_, is_simulation_);
    if (profile_time_map_.count(key)) {
      return profile_time_map_[key];
    } else {
      return 0;
    }
  }

  void SerializeProfiledModule(Expr e, std::string fn_prefix) {
    static auto connector = distributed::connector::ConnectorManager::Get()->GetConnector("mpi");
    if (connector->rank == 0) {
      // serialize and dump module
      auto module_json = ir::serialization::SaveJSON(e);
      std::string module_dump_path(fn_prefix);
      module_dump_path += ".mod";
      std::ofstream module_dump_file;
      module_dump_file.exceptions(std::ofstream::failbit);
      LOG(INFO) << "Dumping module to " << module_dump_path << ".";
      try {
        module_dump_file.open(module_dump_path);
        module_dump_file << module_json;
        module_dump_file.close();
      } catch (std::ofstream::failure& e) {
          LOG(FATAL) << "Failed to write to dump file " << module_dump_path << ".";
      }
      // serialize and dump profile
      auto size_and_buffer = SerializeProfile_();
      size_t bytes_size;
      std::shared_ptr<char> profile_buffer;
      bytes_size = size_and_buffer.first;
      profile_buffer = size_and_buffer.second;
      std::string profile_dump_path(fn_prefix);
      profile_dump_path += ".profile";
      std::ofstream profile_dump_file;
      profile_dump_file.exceptions(std::ofstream::failbit);
      LOG(INFO) << "Dumping profile to " << profile_dump_path << ".";
      try {
        profile_dump_file.open(profile_dump_path, std::ofstream::binary);
        profile_dump_file.write(profile_buffer.get(), bytes_size);
        profile_dump_file.close();
      } catch (std::ofstream::failure& e) {
          LOG(FATAL) << "Failed to write to dump file " << profile_dump_path << ".";
      }
      connector->Barrier();
    } else {
      connector->Barrier();
    }
  }

  void SynchronizeProfile_() {
    // synchronizes the profile data between devices so their generated schedule is identical
    auto connector = distributed::connector::ConnectorManager::Get()->GetConnector("mpi");
    // send profiled keys to root
    size_t bytes_size;
    std::shared_ptr<char> profile_buffer;
    auto size_and_buffer = SerializeProfile_();
    bytes_size = size_and_buffer.first;
    profile_buffer = size_and_buffer.second;
    
    if (connector->rank == 0) {
      size_t all_ranks_bytes_sizes[connector->size];
      connector->Gather(&bytes_size, sizeof(size_t), all_ranks_bytes_sizes, sizeof(size_t), 0);
      // check all ranks have the same byte size
      for(int i=1; i<connector->size; i++) {
        CHECK_EQ(bytes_size, all_ranks_bytes_sizes[i]);
      }
      // now send data
      auto serialized_profiles = MakeBuffer_(bytes_size * connector->size);
      connector->Gather(profile_buffer.get(), bytes_size, serialized_profiles.get(), bytes_size, 0);
      std::vector<std::unordered_map<OpKey, uint64_t>> all_ranks_profiles;
      char* current_buf_ptr = serialized_profiles.get();
      char* buf_end = current_buf_ptr + bytes_size * connector->size;
      for(size_t i=0; i<connector->size; i++) {
        all_ranks_profiles.emplace_back(DeserializeProfile_(current_buf_ptr, bytes_size));
        current_buf_ptr += bytes_size;
      }
      CHECK_EQ(current_buf_ptr, buf_end);
      // calulate average
      for(auto& it: profile_time_map_) {
        uint64_t sum_rank_times = 0;
        for(auto& rank_map: all_ranks_profiles) {
          CHECK(rank_map.count(it.first)) << "Keys are different across ranks!";
          sum_rank_times += rank_map.at(it.first);
        }
        profile_time_map_[it.first] = sum_rank_times / all_ranks_profiles.size();
      }
      // broadcast the avged map
      size_t avged_bytes_size;
      std::shared_ptr<char> avged_profile_buffer;
      auto size_and_buffer_avged = SerializeProfile_();
      avged_bytes_size = size_and_buffer_avged.first;
      avged_profile_buffer = size_and_buffer_avged.second;
      connector->Broadcast(avged_profile_buffer.get(), avged_bytes_size, 0);
    } else {
      // byte sizes
      connector->Gather(&bytes_size, sizeof(size_t), nullptr, 0, 0);
      // profile data
      connector->Gather(profile_buffer.get(), bytes_size, nullptr, 0, 0);
      // receive avged profile data
      auto serialized_avged_profile = MakeBuffer_(bytes_size);
      connector->Broadcast(serialized_avged_profile.get(), bytes_size, 0);
      // deserialize buffer
      profile_time_map_ = DeserializeProfile_(serialized_avged_profile.get(), bytes_size);
    }
    // barrier here to make the processes more uniform
    connector->Barrier();
  }

  std::pair<size_t, std::shared_ptr<char>> SerializeProfile_() {
    std::vector<OpKey> keys;
    std::vector<uint64_t> values;
    uint64_t total_key_length = 0;
    for(auto& it: profile_time_map_) {
      keys.push_back(it.first);
      total_key_length += it.first.size();
    }
    std::sort(keys.begin(), keys.end());
    for(auto& key: keys) {
      values.push_back(profile_time_map_.at(key));
    }
    size_t bytes_length = /*length*/ sizeof(uint64_t) + /*keys*/ total_key_length + keys.size() + /*times*/ sizeof(uint64_t) * keys.size();

    auto profile_buffer = MakeBuffer_(bytes_length);

    auto serialize_uint64_t = [](char* buffer, uint64_t n) -> size_t {
      std::memcpy(buffer, &n, sizeof(uint64_t));
      return sizeof(uint64_t);
    };

    auto serialize_string = [](char* buffer, std::string& s) -> size_t {
      std::memcpy(buffer, s.c_str(), s.size() + 1);
      return s.size() + 1;
    };

    char* current_buf_ptr = profile_buffer.get();
    char* end_of_buffer = current_buf_ptr + bytes_length;
    current_buf_ptr += serialize_uint64_t(current_buf_ptr, static_cast<uint64_t>(keys.size()));
    for(auto key: keys) {
      current_buf_ptr += serialize_string(current_buf_ptr, key);
      CHECK(current_buf_ptr < end_of_buffer);
    }
    for(size_t i=0; i<values.size(); i++) {
      current_buf_ptr += serialize_uint64_t(current_buf_ptr, values[i]);
      if(i != values.size()-1) {
        CHECK(current_buf_ptr < end_of_buffer);
      }
    }
    CHECK_EQ(current_buf_ptr, end_of_buffer);
    return std::make_pair(bytes_length, profile_buffer);
  }

  std::unordered_map<OpKey, uint64_t> DeserializeProfile_(char* buffer, size_t buffer_size) {
    char* buffer_end = buffer + buffer_size;
    auto deserialize_uint64_t = [buffer_end](char* buf) -> uint64_t {
      CHECK_LE(buf+sizeof(uint64_t), buffer_end) << "Access past end of buffer: at "
                                                << static_cast<void*>(buf+sizeof(uint64_t))
                                                << " while end is: " << static_cast<void*>(buffer_end);
      uint64_t n;
      std::memcpy(&n, buf, sizeof(uint64_t));
      return n;
    };
    auto deserialize_string = [buffer_end](char* buf) -> std::string {
      std::string s;
      size_t offset = 0;
      while(buf[offset] != '\0') {
        s.push_back(buf[offset]);
        offset++;
        CHECK(buf+offset < buffer_end) << "Access past end of buffer: at "
                                      << static_cast<void*>(buf+offset)
                                      << " while end is: " << static_cast<void*>(buffer_end)
                                      <<". Current offset: " << offset << ".";
      }
      return s;
    };
    std::vector<OpKey> keys;
    std::vector<uint64_t> values;
    char* current_buf_ptr = buffer;
    uint64_t key_size = deserialize_uint64_t(buffer);
    current_buf_ptr += sizeof(uint64_t);
    for(size_t i=0; i<key_size; i++) {
      std::string key = deserialize_string(current_buf_ptr);
      current_buf_ptr += key.size() + 1;
      keys.emplace_back(std::move(key));
    }
    for(size_t i=0; i<key_size; i++) {
      values.emplace_back(deserialize_uint64_t(current_buf_ptr));
      current_buf_ptr += sizeof(uint64_t);
    }
    CHECK_EQ(current_buf_ptr, buffer_end);
    std::unordered_map<OpKey, uint64_t> result;
    for(int i=0; i<key_size; i++) {
      result[keys[i]] = values[i];
    }
    return result;
  }

  std::shared_ptr<char> MakeBuffer_(size_t size) {
    return std::shared_ptr<char>(new char[size], std::default_delete<char[]>());
  }

  bool has_profile_ = false;
  bool is_simulation_ = false;
  std::unordered_map<OpKey, uint64_t> profile_time_map_;
};


// a simple non-profile based scheduler
class FIFOScheduler : public StreamSchedulerBase {
 public:
  /*! This scheduler schedules the execution order of ops so communication ops can better overlap
   * with computation ops. It works on BBNF/GNF and outputs the scheduled expression in ANF.
   *
   * Why do we need this pass:
   * Since we do not have a dynamic execution engine that only launches an op when all its inputs
   * becomes ready, explicit synchronization ops are introduced to preserve correct dependencies
   * between ops launched to different CUDA streams. However, a bad op launch order can prevent the
   * overlap between ops on different streams. Take the following data flow graph as an example,
   * where all allreduce ops are executed on a separate communication stream (stream b) and all
   * computation ops (atan and mul) are executed on a single compute stream (stream a):
   *
   *           -> allreduce --> atan3 -
   *         /                          \
   *   atan0 --->   atan1   --> atan2 ---> mul
   *
   * When transforming it into ANF, the following two op orders are both possible:
   *   Order 1: atan0 -> allreduce -> atan3 -> atan1 -> atan2 -> mul
   *     Execution timeline:
   *       Stream a:   atan0              -> atan3 -> atan1 -> atan2 -> mul
   *       Stream b:         -> allreduce
   *
   *   Order 2: atan0 -> allreduce -> atan1 -> atan2 -> atan3 -> mul
   *     Execution timeline:
   *       Stream a:   atan0 -> atan1 -> atan2 -> atan3 -> mul
   *       Stream b:         -> allreduce
   *
   * In order 1, since atan3 depends on the output of allreduce, it must wait for allreduce to
   * finish. However, the scheduled op order demands that atan1 and atan2 are launched after atan3,
   * thus they are also blocked. Order 2 schedules atan1 and atan2 before atan3 to avoid this
   * problem.
   *
   * Current ToANormalForm travels the data flow graph in post DFS order, in which chains of ops
   * (e.g. allreduce and the consumer of its output) will be put next to each other, resulting in a
   * immediate synchonization after the communication op. This can lead to unnecessary blocking
   * if no enough computation ops have been launched on the computation stream to overlap with
   * the communication op. To alleviate this problem, this pass implements a simple FIFO scheduling
   * strategy:
   *   1. Maintain a counter for each op that keeps track of the number of its unscheduled
   *      predecessors. Add all ops with no predecessors into a ready queue.
   *   2. While the queue is not empty, pop an op out of the queue and mark it as scheduled.
   *      Decrease the counter for all its successors. If any successor has no unscheduled
   *      predecessors (i.e. becomes ready), push it into the exection queue.
   *   3. The order of popping ops out of the ready queue is out final op launch order.
   *
   * This effectively performs a topological sort on the ops and issues ops in a "BFS" order.
   * In addition, this pass also tries to delay any node that depends on a communication node
   * as late as possible by using a separate queue for them. Consider the following graph:
   *
   *           -> allreduce --> atan3 -----------------------
   *         /                                               \
   *   atan0 --->   atan1   --> atan2 --> atan4 --> atan5 --> mul
   *
   * Even if FIFO scheduling is used, atan3 will be scheduled at a relatively early position,
   * which can still block computation if the allreduce is long, e.g.:
   *
   * ANF order: atan0 -> atan1 -> allreduce -> atan2 -> atan3 -> atan4 -> atan5 -> mul
   * Execution timeline:
   *   Stream a:   atan0 -> atan1 -> atan2             -> atan3 -> atan4 -> atan5 -> mul
   *   Stream b:         -> l o n g _ a l l r e d u c e
   *
   * By using a separate queue for ops that directly depends on a communication,
   * those ops are delayed until no other op is available, leaving more room for overlap.
   */
  Expr Schedule(Expr e) {
    // create the data flow graph
    Arena arena;
    DependencyGraph dfg = CreateDependencyGraph(&arena, e, /*prune_atomic_nodes=*/true);
    // map each node in the dependency graph to the expression it represents
    NodeExprMap node_expr;
    // ready queue for all ops that directly depends on a communication op
    std::queue<Node*> comm_successor_ready_queue;
    // ready queue for all other ops
    std::queue<Node*> ready_queue;
    // counter that keeps track of the number of each op's current unscheduled predecessors
    // the dependency graph in tvm is a data flow graph with edge direction reversed, so we
    // use out-degree here instead of in-degree.
    std::unordered_map<Node*, int> out_degree;
    // keeps track of whether an op directly depends on a communication op
    std::unordered_set<Node*> comm_successor_nodes;

    for (auto& it : dfg.expr_node) {
      node_expr[it.second] = it.first;
    }
    std::vector<Node*>& nodes = dfg.post_dfs_order;

    // calculate out-degree for each node and populate comm_successor_nodes map
    for (auto node_it = nodes.rbegin(); node_it != nodes.rend(); node_it++) {
      out_degree[(*node_it)] = 0;
      if (auto call_node = node_expr[*node_it].as<CallNode>()) {
        if (IsCollectiveOp(call_node->op)) {
          // record direct successor nodes of communication op
          for (auto parent = (*node_it)->parents.head; parent; parent = parent->next) {
            comm_successor_nodes.insert(parent->value);
          }
        }
      }

      for (auto child = (*node_it)->children.head; child; child = child->next) {
        out_degree[(*node_it)]++;
      }
    }
    // push nodes with zero predecessors into the queue
    for (auto& node : nodes) {
      if (out_degree[node] == 0) {
        ready_queue.push(node);
      }
    }

    Expr ret;
    // in each step, we pop an op out of the queue, add it to the ANF and
    // push all its ready successors into the corresponding ready queue
    auto process_queue_element = [&](std::queue<Node*>& q) {
      while (!q.empty()) {
        Node* node = q.front();
        ret = VisitExpr(node_expr.at(node));
        for (auto parent = node->parents.head; parent; parent = parent->next) {
          out_degree[parent->value]--;
          if (out_degree[parent->value] == 0) {
            if (comm_successor_nodes.count(parent->value)) {
              comm_successor_ready_queue.push(parent->value);
            } else {
              ready_queue.push(parent->value);
            }
          }
        }
        q.pop();
      }
    };

    while (!ready_queue.empty() || !comm_successor_ready_queue.empty()) {
      process_queue_element(ready_queue);
      process_queue_element(comm_successor_ready_queue);
    }

    return let_list_.Get(ret);
  }
};

// main implementation of Lancet, performing both scheduling and model partitioning
class LancetScheduler : public ProfiledCUDAScheduler {
 public:
  LancetScheduler(bool is_simulation=false) : ProfiledCUDAScheduler(is_simulation) {}

  Map<Expr, Array<Array<FloatImm>>> GetProfileData() {
    CHECK(has_comm_data_) << "Must run Schedule once to get processed profile data.";
    Map<Expr, Array<Array<FloatImm>>> result;
    for(auto it: comm_times) {
      Expr expr = it.first;
      auto comm_times_vector = it.second;
      CHECK(comm_sizes.count(it.first));
      auto comm_sizes_vector = comm_sizes.at(it.first);
      CHECK_EQ(comm_times_vector.size(), comm_sizes_vector.size());
      Array<Array<FloatImm>> data_pairs;
      for(int i=0; i<comm_times_vector.size(); i++) {
        data_pairs.push_back({{DataType::Float(32), comm_sizes_vector[i]}, {DataType::Float(32), comm_times_vector[i]}});
      }
      result.Set(it.first, data_pairs);
    }
    return result;
  }

  Function Schedule(Function func, ScheduleHeuristics sched_heuristic,
                    TimelineOptAlgo timeline_opt_algo, int dp_group_size,
                    bool disable_load_module = false,
                    bool disable_fusion = false, bool disable_partition = false) {
    auto connector = distributed::connector::ConnectorManager::Get()->GetConnector("mpi");
    const char* load_optimized_module_path = getenv("LOAD_OPTIMIZED_MODULE_FROM");
    if(load_optimized_module_path != nullptr && !disable_load_module) {
      // skip optimization and directly load a expr
      if(connector->rank == 0) {
        // load module expr
        std::string module_dump_path(load_optimized_module_path);
        LOG(INFO) << "Loading optimized module from " << module_dump_path;
        auto mod_main = LoadExprFromFile(module_dump_path);
        mod_main = PrimFunctionRestorer().Run(mod_main);
        mod_main = pass::DelayAllToAllvInFunction(Downcast<Function>(mod_main));
        LOG(INFO) << "Successfully loaded module, skipping optimization.";

        auto serialized_expr = ir::serialization::SaveJSON(mod_main);
        // bcast string size first
        int json_size = serialized_expr.size();
        connector->Broadcast(&json_size, sizeof(int), 0);
        // bcast the actual expr
        connector->Broadcast(const_cast<char*>(serialized_expr.c_str()), json_size + 1, 0);
        // barrier
        connector->Barrier();
        return Downcast<Function>(mod_main);
      } else {
        // received json size first
        int json_size;
        connector->Broadcast(&json_size, sizeof(int), 0);
        // create buffer
        char* serialized_expr_buffer = new char[json_size + 1];
        connector->Broadcast(serialized_expr_buffer, json_size + 1, 0);

        // deserialize the expr
        std::string serilized_expr(serialized_expr_buffer);
        Function result_expr = Downcast<Function>(ir::serialization::LoadJSON(serilized_expr));
        auto optype_fixed_expr = Downcast<Function>(FixLoadedOpType().Mutate(result_expr));
        // barrier
        connector->Barrier();
        delete[] serialized_expr_buffer;
        return optype_fixed_expr;
      }
    }
    CheckAndDumpProfiledExpr(func);
    Expr e = ProfiledCUDAScheduler::Schedule(func->body);
    e = SizeBasedCommFusor().Run(e);
    Arena arena;
    dfg_ = CreateDependencyGraph(&arena, e, true);
    orig_expr_node_ = dfg_.expr_node;

    NodeMap<NodeType> node_type_map;
    NodeMap<TimeType> exec_time_map;
    NodeMap<std::string> node_name_map;
    NodeSet comm_nodes_without_profile;

    // assign a id to each node for schedule serialization and synchronizaton
    // we use post_dfs_order here, the mapping of expr to id is identical on
    // each machine
    std::unordered_map<const Node*, int> node2serid;
    std::unordered_map<int, const Node*> serid2node;

    for(int serid=0; serid < dfg_.post_dfs_order.size(); serid++) {
      const Node* n = dfg_.post_dfs_order[serid];
      node2serid[n] = serid;
      serid2node[serid] = n;
    }

    // data for fitting allreduce cost model
    Nodes comm_nodes;
    NodeMap<CommComponents> comm_size_map;

    int tuple_counter = 0;
    int tgi_counter = 0;
    std::unordered_map<std::string, int> unique_name_counter;

    // we try to get expert dim size from alltoall ops here
    int n_experts = -1;

    // populate node_type_map, exec_time_map, node_name_map for each node
    for (auto& it : dfg_.expr_node) {
      node_expr_[it.second] = it.first;
      if(auto call_node = it.first.as<CallNode>()) {
        if(IsCollectiveOp(call_node->op)) {
          node_type_map[it.second] = NodeType::kCommNode;
          if(IsAllToAll(it.first)) {
            // get shape
            auto shape = call_node->checked_type_.as<TensorTypeNode>()->shape;
            int inferred_n_experts = -1;
            if (shape.size() == 3) {
              // [E, C, M]
              inferred_n_experts = shape[0].as<IntImmNode>()->value;
            } else {
              CHECK(shape.size() == 4);
              // [G, LE, C, M]
              inferred_n_experts = shape[1].as<IntImmNode>()->value * shape[0].as<IntImmNode>()->value;
            }
            if (n_experts == -1) {
              n_experts = inferred_n_experts;
            } else {
              CHECK_EQ(n_experts, inferred_n_experts);
            }
          }
        } else {
          node_type_map[it.second] = NodeType::kCompNode;
        }
        OpKey key = GenerateOpKeyFromCallOp(Downcast<Call>(it.first), profile_time_map_, is_simulation_);
        if(unique_name_counter.count(key)) {
          node_name_map[it.second] = key + "_" + std::to_string(unique_name_counter.at(key));
          unique_name_counter[key] ++;
        } else {
          node_name_map[it.second] = key;
          unique_name_counter[key] = 1;
        }
        if (profile_time_map_.count(key)) {
          exec_time_map[it.second] = profile_time_map_.at(key);
          if(node_type_map[it.second] == NodeType::kCommNode) {
            comm_times[call_node->op].push_back(exec_time_map[it.second]);
            size_t output_size = 0;
            if(auto tuple_type = call_node->checked_type().as<TupleTypeNode>()) {
              for(auto& ty: tuple_type->fields) {
                size_t field_size = 1;
                if(auto tensor_ty = ty.as<TensorTypeNode>()) {
                  for(auto& x: tensor_ty->shape) {
                    field_size *= Downcast<IntImm>(x)->value;
                  }
                  field_size *= tensor_ty->dtype.bits();
                  output_size += field_size;
                }
              }
            } else {
              auto tensor_type = call_node->checked_type().as<TensorTypeNode>();
              CHECK(tensor_type);
              output_size = 1;
              for(auto& x: tensor_type->shape) {
                output_size *= Downcast<IntImm>(x)->value;
              }
              output_size *= tensor_type->dtype.bits();
            }
            comm_nodes.push_back(it.second);
            CHECK(call_node->op->IsInstance<OpNode>());
            comm_sizes[call_node->op].push_back(output_size);
            DLOG(INFO) << "Profiled comm op: comm name: " << key << ", comm size: " << output_size << ", comm time: " << exec_time_map[it.second];
            comm_size_map[it.second] = {{OpToCommType(Downcast<Op>(call_node->op)), output_size}};
          }
        } else {
          if (IsCollectiveOp(call_node->op)) {
            // if we don't have profile data for this collective op, we fill it later using cost model
            comm_nodes_without_profile.insert(it.second);
          } else {
            if (!IsReshapeOp(call_node->op)) {
              LOG(WARNING) << "Cannot find Op " << key << " in profile.";
            }
            exec_time_map[it.second] = 0;
          }
        }
      } else {
        if(it.first->IsInstance<TupleNode>()) {
          node_name_map[it.second] = "Tuple_" + std::to_string(tuple_counter);
          tuple_counter++;
        } else if (it.first->IsInstance<TupleGetItemNode>()) {
          node_name_map[it.second] = "TupleGetItem_" + std::to_string(tgi_counter);
          tgi_counter++;
        }
        node_type_map[it.second] = NodeType::kCompNode;
        exec_time_map[it.second] = 0;
      }
    }
    // fill the rest node with default values
    for(auto node: dfg_.post_dfs_order) {
      if(!node_type_map.count(node)) {
        node_type_map[node] = NodeType::kCompNode;
      }
      if(!exec_time_map.count(node) && !comm_nodes_without_profile.count(node)) {
        exec_time_map[node] = 0;
      }
    }
    // fit cost model for each comm op
    ExprSet ops_to_fit;
    for(auto it: comm_times) {
      ops_to_fit.insert(it.first);
    }
    for(auto node: comm_nodes_without_profile) {
      CHECK(node_expr_.count(node));
      auto expr = node_expr_.at(node);
      CHECK(expr->IsInstance<CallNode>());
      CHECK(Downcast<Call>(expr)->op->IsInstance<OpNode>());
      ops_to_fit.insert(Downcast<Call>(expr)->op);
    }
    std::unordered_map<CommType, lerp::LinearInterpolator<SimulateTimeType>> comm_cost_model_map;
    for(auto it: ops_to_fit) {
      LOG(INFO) << "Fitting cost model for op " << it;
      CHECK(it->IsInstance<OpNode>());
      auto comm_type = OpToCommType(Downcast<Op>(it));
      if(comm_type == CommType::kAllToAll) {
        if(const char* all2all_supp_profile_str = getenv("ALL2ALL_SUPPLEMENT_PROFILE")) {
          std::string all2all_supp_profile_path(all2all_supp_profile_str);
          LoadSupplementProfile(Downcast<Op>(it), all2all_supp_profile_path);
        }
      }
      if(comm_type == CommType::kAllReduce) {
        if(const char* allreduce_supp_profile_str = getenv("ALLREDUCE_SUPPLEMENT_PROFILE")) {
          std::string allreduce_supp_profile_path(allreduce_supp_profile_str);
          LoadSupplementProfile(Downcast<Op>(it), allreduce_supp_profile_path);
        }
      }
      if(comm_type == CommType::kReduceScatter) {
        if(const char* reducescatter_supp_profile_str = getenv("REDUCESCATTER_SUPPLEMENT_PROFILE")) {
          std::string reducescatter_supp_profile_path(reducescatter_supp_profile_str);
          LoadSupplementProfile(Downcast<Op>(it), reducescatter_supp_profile_path);
        }
      }
      if(comm_type == CommType::kAllGather) {
        if(const char* allgather_supp_profile_str = getenv("ALLGATHER_SUPPLEMENT_PROFILE")) {
          std::string allgather_supp_profile_path(allgather_supp_profile_str);
          LoadSupplementProfile(Downcast<Op>(it), allgather_supp_profile_path);
        }
      }
      CHECK(comm_sizes.count(it)) << "Cannot find comm size for op " << it << ". Please make sure you have profile for this op.";
      CHECK(comm_times.count(it)) << "Cannot find comm size for op " << it << ". Please make sure you have profile for this op.";
      for (auto it: comm_times) {
        Expr op = it.first;
        CommType key = OpToCommType(Downcast<Op>(op));
        auto& comm_sizes_for_key = comm_sizes.at(op);
        comm_cost_model_map[key] = lerp::LinearInterpolator<double>(comm_sizes_for_key,
                                                                    it.second);
      }
    }
    has_comm_data_ = true;

    // construct cost model fn
    CommCostModel comm_cost_model = [&](const CommComponents& components) {
      SimulateTimeType total_time = 0;
      // CommType comm_type = CommType::kAllReduce;
      // int64_t comm_size = 0;
      for (auto it: components) {
        if (comm_cost_model_map.count(it.first)) {
          total_time += comm_cost_model_map[it.first](it.second);
          // comm_type = it.first;
          // comm_size = it.second;
        } else {
          LOG(FATAL) << "Cannot find cost model for " << it.first;
        }
      }
      // LOG(INFO) << "Comm: " << comm_type << "size: " << comm_size << ", cost: " << total_time;
      return total_time;
    };

    // now fill the rest comm nodes with cost model
    for(auto node: comm_nodes_without_profile) {
      CHECK(node_expr_.count(node));
      auto expr = node_expr_.at(node);
      CHECK(expr->IsInstance<CallNode>());
      auto comm_type = OpToCommType(Downcast<Op>(Downcast<Call>(expr)->op));
      auto size = BytesCompactType(expr->checked_type()) * 8;
      exec_time_map[node] = comm_cost_model({{comm_type, size}});
      comm_size_map[node] = {{comm_type, size}};
    }

    if(const char* dfg_dump_file = getenv("DEBUG_DUMP_DFG_PREFIX")) {
      std::string dump_path(dfg_dump_file);
      dump_path += "_" + std::to_string(getpid()) + ".txt";
      std::ofstream dump_file;
      dump_file.exceptions(std::ofstream::failbit);
      LOG(INFO) << "Dumping debug DFG to " << dump_path << ".";
      try {
        dump_file.open(dump_path);
        dump_file << DebugDumpDependencyGraphWithAttrs(&dfg_, exec_time_map, node_name_map, comm_size_map, node_type_map);
        dump_file.close();
      } catch (std::ofstream::failure& e) {
          LOG(FATAL) << "Failed to write to dump file " << dump_path << ".";
      }
    }

    // construct extended dfg
    ExtendedDFG extended_dfg(&dfg_, exec_time_map, node_type_map, node_name_map, comm_size_map);


    Nodes node_order;
    ScheduledDFG result(extended_dfg);
    Function result_expr;
    // only rank 0 runs schedule optimization
    if(connector->rank == 0) {
      // get optimization parameters
      int max_iterations = 1000000;
      if (const char* max_iterations_str = getenv("MAX_ITERATIONS")) {
        max_iterations = std::stod(std::string(max_iterations_str));
      }
      if (const char* disable_fusion_str = getenv("DISABLE_FUSION")) {
        if (strcmp(disable_fusion_str, "0") != 0) {
            LOG(INFO) << "DISABLE_FUSION env set true.";
            disable_fusion = true;
        }
      }
      if (const char* disable_partition_str = getenv("DISABLE_PARTITION")) {
        if (strcmp(disable_partition_str, "0") != 0) {
            LOG(INFO) << "DISABLE_PARTITION env set true.";
            disable_partition = true;
        }
      }
      if (disable_fusion) {
        LOG(INFO) << "Fusion disabled.";
      }
      if (disable_partition) {
        LOG(INFO) << "Partition disabled.";
      }
      // run the optimizations in Lancet
      auto makespan_and_sched_dfg = RunOptimization(func, extended_dfg, sched_heuristic, timeline_opt_algo, comm_cost_model, dp_group_size, n_experts, max_iterations, disable_fusion, disable_partition);
      result = makespan_and_sched_dfg.second;
      CHECK(result.scheduled_expr.defined()) << "After running the optimization, scheduled_expr is not defined.";
      node_order = result.total_order;

      LOG(INFO) << "Scheduler predicted end_to_end time: " << makespan_and_sched_dfg.first / 1000.0 << " ms." << std::endl;

      // reconstruct the main function
      result_expr = WithFields(func, func->params, result.scheduled_expr);

      // post processing: delay alltoallvs for better overlapping with computation
      result_expr = pass::DelayAllToAllvInFunction(result_expr);

      if (const char* optimized_expr_prefix = getenv("DUMP_OPTIMIZED_EXPR_PREFIX")) {
        DumpExprToJSON(std::string(optimized_expr_prefix), result_expr);
      }

      auto serialized_expr = ir::serialization::SaveJSON(result_expr);
      // bcast string size first
      int json_size = serialized_expr.size();
      connector->Broadcast(&json_size, sizeof(int), 0);
      // bcast the actual expr
      connector->Broadcast(const_cast<char*>(serialized_expr.c_str()), json_size + 1, 0);
      // barrier
      connector->Barrier();
    } else {
      // received json size first
      int json_size;
      connector->Broadcast(&json_size, sizeof(int), 0);
      // create buffer
      char* serialized_expr_buffer = new char[json_size + 1];
      connector->Broadcast(serialized_expr_buffer, json_size + 1, 0);

      // deserialize the expr
      std::string serilized_expr(serialized_expr_buffer);
      result_expr = Downcast<Function>(ir::serialization::LoadJSON(serilized_expr));

      // barrier
      connector->Barrier();
      delete[] serialized_expr_buffer;
    }
    if (const char* exit_after_dump = getenv("RAF_EXIT_AFTER_OPT")) {
      auto connector = distributed::connector::ConnectorManager::Get()->GetConnector("mpi");
      connector->Finalize();
      exit(0);
    }


    result_expr = Downcast<Function>(PrimFunctionRestorer().Run(result_expr));
    result_expr = fixOrder(result_expr);

    Function type_infered_func = Downcast<Function>(pass::InferType(result_expr));
    return type_infered_func;
  }
protected:
  void LoadSupplementProfile(Op comm_op, const std::string& profile_path) {
    // get range of comm sizes used in the original profile
    CommType comm_type = OpToCommType(comm_op);
    int64_t min_size = -1;
    int64_t max_size = -1;
    // use min_size / 4 ~ max_size * 4 as the range
    if (comm_sizes.count(comm_op)) {
      for(auto comm_size: comm_sizes.at(comm_op)) {
        if (min_size == -1 || comm_size < min_size) {
          min_size = comm_size;
        }
        if (max_size == -1 || comm_size > max_size) {
          max_size = comm_size;
        }
      }
    }
    if (min_size < max_size) {
      min_size = min_size / 4;
      max_size = max_size * 4;
    } else {
      // orig profile only contain one size. use all supplentary profile data
      min_size = 0;
      max_size = std::numeric_limits<int64_t>::max();
    }
    LOG(INFO) << "Loading " << comm_type << " supplement profile from " << profile_path;
    std::ifstream profile_file;
    profile_file.exceptions(std::ifstream::badbit);
    profile_file.open(profile_path, std::ios::ate);
    std::streamsize supp_profile_size = profile_file.tellg();
    profile_file.seekg(0, std::ios::beg);

    std::string line;
    LOG(INFO) << "Using data from size range: (" << min_size / 1000000 << " MB, " << max_size / 1000000 << " MB)";
    while(std::getline(profile_file, line)) {
      if(line.empty()) {
        continue;
      }
      if(line.back() == '\n') {
        line.pop_back();
      }
      if(line.empty()) {
        LOG(INFO) << "empty line, continue";
        continue;
      }
      auto size_str = line.substr(0, line.find(","));
      auto time_str = line.substr(line.find(",")+1, std::string::npos);
      double size = std::stod(size_str) * 32;
      double time = std::stod(time_str);
      if (size < max_size && size > min_size) {
        DLOG(INFO) << "Using " << comm_type << " supp size: " << size << ", time: " << time;
        comm_times[comm_op].push_back(time);
        comm_sizes[comm_op].push_back(size);
      }
    }
  }

private:
  std::unordered_set<Node*> ready_set_;
  std::unordered_map<Node*, int> out_degree_;
  std::unordered_set<Node*> is_update_node_;
  DependencyGraph dfg_;
  ExprMap<Node*> orig_expr_node_;
  NodeExprMap node_expr_;
  NodeMap<Nodes> fusion_groups_;
  VarMap<std::vector<Var>> fusion_groups_as_var_;
  ExprMap<std::vector<double>> comm_times;
  ExprMap<std::vector<double>> comm_sizes;
  bool has_comm_data_ = false;
};

Expr FIFOScheduleTransform(const Expr& e) {
  // auto result = FIFOScheduler().Schedule(e);
  // if (const char* optimized_expr_prefix = getenv("DUMP_OPTIMIZED_EXPR_PREFIX")) {
  //   DumpExprToJSON(std::string(optimized_expr_prefix), result);
  // }
  // return result;
  return FIFOScheduler().Schedule(e);
}


std::function<Function(const Function&)>
CreateLancetSchedulerTransform(ScheduleHeuristics sched_heuristic, TimelineOptAlgo timeline_opt_algo, int dp_group_size, bool disable_load_module = false,
                              bool disable_fusion = false, bool disable_partition = false) {
  return [=](const Function& e) -> Function {
    return LancetScheduler().Schedule(e, sched_heuristic, timeline_opt_algo, dp_group_size, disable_load_module, disable_fusion, disable_partition);
  };
}

std::function<Function(const Function&)>
CreateFIFOProfiledScheduleTransform(int dp_group_size, TimelineOptAlgo timeline_opt_algo, bool disable_load_module) {
  // FIFOProfiledSchedule is just lancet schedule with scheduling and partition disabled
  return CreateLancetSchedulerTransform(ScheduleHeuristics::kFIFO, timeline_opt_algo, dp_group_size, disable_load_module, true, true);
}

std::function<Function(const Function&)>
CreateLancetSchedulerFIFOTransform(int dp_group_size, TimelineOptAlgo timeline_opt_algo, bool disable_load_module) {
  return CreateLancetSchedulerTransform(ScheduleHeuristics::kFIFO, timeline_opt_algo, dp_group_size, disable_load_module);
}

}  // namespace data_parallel_schedule

TVM_REGISTER_PASS_CONFIG_OPTION("raf.dp_schedule.use_profile", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.dp_schedule.enable_lancet", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.dp_schedule.disable_load_module", Bool);

using scheduler_common::TimelineOptAlgo;

static std::unordered_map<std::string, TimelineOptAlgo> topt_map = {
  {
    {"dp", TimelineOptAlgo::kDP},
    {"heuristic", TimelineOptAlgo::kHeuristic},
    {"range", TimelineOptAlgo::kRangeBased},
  }
};

Pass DataParallelSchedule(int dp_group_size) {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> dp_schedule_pass_func = [=](Function f, IRModule m,
                                                                                         PassContext pc) {
    Bool use_profile = pc->GetConfig("raf.dp_schedule.use_profile", Bool(false)).value();
    Bool enable_lancet = pc->GetConfig("raf.dp_schedule.enable_lancet", Bool(true)).value();
    String timeline_opt_algo = pc->GetConfig("raf.dp_schedule.timeline_opt_algo", String("dp")).value();
    Bool disable_load_module = pc->GetConfig("raf.dp_schedule.disable_load_module", Bool(false)).value();
    if (!topt_map.count(timeline_opt_algo)) {
      LOG(FATAL) << "Unknown timeline optimization algorithm " << timeline_opt_algo << ".";
    }
    TimelineOptAlgo timeline_opt_algo_enum = topt_map.at(timeline_opt_algo);
    if (static_cast<bool>(use_profile)) {
      if(enable_lancet) {
        LOG(INFO) << "Using Lancet optimizations.";
        return data_parallel_schedule::CreateLancetSchedulerFIFOTransform(dp_group_size, timeline_opt_algo_enum, static_cast<bool>(disable_load_module))(f);
      } else {
        LOG(INFO) << "Using FIFO Profiled Scheduler.";
        return data_parallel_schedule::CreateFIFOProfiledScheduleTransform(dp_group_size, timeline_opt_algo_enum, static_cast<bool>(disable_load_module))(f);
      }
    } else {
      return Downcast<Function>(
        tvm::relay::TransformF(data_parallel_schedule::FIFOScheduleTransform, f));
    }
  };

  static auto* dev2str = tvm::runtime::Registry::Get("raf._core.core_utils.dev2str");
  Device device = Device::Current(/*allow_default=*/false);
  CHECK_NE(device.device_type(), DevType::kUnknown()) << "Encountered unknown device type.";
  tvm::String device_str = (*dev2str)(device);

  Pass func_pass = CreateRAFFunctionPass(dp_schedule_pass_func, 0, "DataParallelSchedule", {});
  PassInfo pass_info(2, "DataParallelSchedule", {});
  return RAFSequential({InferType(), func_pass, AssignDevice(device_str)}, pass_info);
}

class LancetScheduleSimulatorNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  /*! \brief the joint scheduler */
  data_parallel_schedule::LancetScheduler scheduler = data_parallel_schedule::LancetScheduler(true);

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "raf.distributed.LancetScheduleSimulator";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(LancetScheduleSimulatorNode, Object);
};

class LancetScheduleSimulator : public ObjectRef {
 public:
  static LancetScheduleSimulator make();
  Expr LoadProfile(std::string fn_prefix);
  Expr LoadIR(std::string fn_prefix);
  Expr Schedule(Expr e, std::string schedule_heuristic, std::string timeline_opt_algo, int dp_group_size);
  Map<Expr, Array<Array<FloatImm>>> GetProfileData();
  LancetScheduleSimulatorNode* operator->() {
    CHECK(get() != nullptr);
    return static_cast<LancetScheduleSimulatorNode*>(get_mutable());
  }
  TVM_DEFINE_OBJECT_REF_METHODS(LancetScheduleSimulator, ObjectRef, LancetScheduleSimulatorNode);
};

TVM_REGISTER_NODE_TYPE(LancetScheduleSimulatorNode);

LancetScheduleSimulator LancetScheduleSimulator::make() {
  ObjectPtr<LancetScheduleSimulatorNode> n = make_object<LancetScheduleSimulatorNode>();
  return LancetScheduleSimulator(n);
}

Expr LancetScheduleSimulator::LoadProfile(std::string fn_prefix) {
  LancetScheduleSimulatorNode* n = operator->();
  return n->scheduler.LoadProfiledModuleFromFile(fn_prefix);
}

Expr LancetScheduleSimulator::LoadIR(std::string fn_path) {
  LancetScheduleSimulatorNode* n = operator->();
  return n->scheduler.LoadExprFromFile(fn_path);
}

Expr LancetScheduleSimulator::Schedule(Expr e, std::string schedule_heuristic, std::string timeline_opt_algo, int dp_group_size) {
  static std::unordered_map<std::string, lancet_optimization::ScheduleHeuristics> heuristics_map = {
    {
      {"FIFO", lancet_optimization::ScheduleHeuristics::kFIFO},
      {"dW", lancet_optimization::ScheduleHeuristics::kDW},
    }
  };
  CHECK(e->IsInstance<FunctionNode>()) << "Input expr must be the main function.";
  Function func = Downcast<Function>(e);
  CHECK(heuristics_map.count(schedule_heuristic)) << "Unknown schedule heuristic: " << schedule_heuristic << ".";
  CHECK(topt_map.count(timeline_opt_algo)) << "Unknown timeline optimization algorithm: " << timeline_opt_algo << ".";
  LancetScheduleSimulatorNode* n = operator->();
  return n->scheduler.Schedule(func, heuristics_map.at(schedule_heuristic), topt_map.at(timeline_opt_algo), dp_group_size);
}

Map<Expr, Array<Array<FloatImm>>> LancetScheduleSimulator::GetProfileData() {
  LancetScheduleSimulatorNode* n = operator->();
  return n->scheduler.GetProfileData();
}

class LancetScheduleSimulatorInternal {
public:
  static Expr LoadProfile(LancetScheduleSimulator simulator, String fn_prefix) {
    return simulator.LoadProfile(fn_prefix);
  }
  static Expr LoadIR(LancetScheduleSimulator simulator, String fn_path) {
    return simulator.LoadIR(fn_path);
  }
  static Expr RunSchedule(LancetScheduleSimulator simulator, Expr e, String schedule_heuristic, String timeline_opt_algo, int dp_group_size) {
    return simulator.Schedule(e, schedule_heuristic, timeline_opt_algo, dp_group_size);
  }
  static Map<Expr, Array<Array<FloatImm>>> GetProfileData(LancetScheduleSimulator simulator) {
    return simulator.GetProfileData();
  }
};

Expr PartitionExpr(Function e, int dp_group_size, int n_experts) {
    using Arena = tvm::support::Arena;
    using namespace raf::pass::scheduler_common;
    using namespace raf::pass::extended_dfg;
    using op::IsCollectiveOp;
    using namespace pass::profile_utils;
    // partition interface for python
    // construct DFG
    Expr body = e->body;
    Arena arena;
    auto dfg_ = CreateDependencyGraph(&arena, body, true);
    auto orig_expr_node_ = dfg_.expr_node;

    NodeMap<NodeType> node_type_map;
    NodeMap<SimulateTimeType> exec_time_map;
    NodeMap<std::string> node_name_map;
    NodeMap<CommComponents> comm_size_map;

    // fill in the maps
    for (auto& it : dfg_.expr_node) {
      auto expr = it.first;
      auto node = it.second;
      if(auto call_node = expr.as<CallNode>()) {
        if(IsCollectiveOp(call_node->op)) {
          node_type_map[it.second] = NodeType::kCommNode;
        } else {
          node_type_map[it.second] = NodeType::kCompNode;
        }
        OpKeyProfiledTimeMap dummy_map;
        auto key = GenerateOpKeyFromCallOp(Downcast<Call>(expr), dummy_map, true);
        node_name_map[it.second] = key;
      } else {
        node_type_map[it.second] = NodeType::kCompNode;
      }
      exec_time_map[it.second] = 0;
      comm_size_map[it.second] = {{CommType::kAllReduce, 0}};
    }
    ExtendedDFG extended_dfg(&dfg_, exec_time_map, node_type_map, node_name_map, comm_size_map);
    auto result = lancet_optimization::RunPartitionOnEntireGraph(extended_dfg, dp_group_size, n_experts);
    if (result.first) {
      auto result_expr = WithFields(e, e->params, result.second.scheduled_expr);
      return result_expr;
    } else {
      return Expr();
    }
}

RAF_REGISTER_GLOBAL("raf.pass_.DataParallelSchedule").set_body_typed(DataParallelSchedule);
RAF_REGISTER_GLOBAL("raf.distributed.CommCostModelParams").set_body_typed(lancet_optimization::CommCostModelParams::make);
RAF_REGISTER_GLOBAL("raf.distributed.LancetScheduleSimulator").set_body_typed(LancetScheduleSimulator::make);
RAF_REGISTER_GLOBAL("raf.distributed.LancetScheduleSimulatorLoadProfile").set_body_typed(LancetScheduleSimulatorInternal::LoadProfile);
RAF_REGISTER_GLOBAL("raf.distributed.LancetScheduleSimulatorLoadIR").set_body_typed(LancetScheduleSimulatorInternal::LoadIR);
RAF_REGISTER_GLOBAL("raf.distributed.LancetScheduleSimulatorRunSchedule").set_body_typed(LancetScheduleSimulatorInternal::RunSchedule);
RAF_REGISTER_GLOBAL("raf.distributed.LancetScheduleSimulatorGetProfileData").set_body_typed(LancetScheduleSimulatorInternal::GetProfileData);
RAF_REGISTER_GLOBAL("raf.distributed.DynamicEvalCurrentSchedule_").set_body_typed(lancet_optimization::DynamicEvalCurrentSchedule_);
RAF_REGISTER_GLOBAL("raf.distributed.DynamicScheduleParams").set_body_typed(lancet_optimization::DynamicScheduleParams::make);
RAF_REGISTER_GLOBAL("raf.distributed.PartitionExpr").set_body_typed(PartitionExpr);

}  // namespace pass
}  // namespace raf

