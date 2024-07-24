/*!
 * Copyright (c) 2022 by Contributors
 * \file partition_exprs.h
 * \brief Partition given exprs.
 */
#pragma once
#include <stdexcept>
#include "./extended_op_profiler.h"
#include "./partition_common.h"
#include "./solve_partition_axes.h"

namespace raf {
namespace pass {
namespace partition_exprs {

using namespace partition_common;
using namespace solve_partition_axes;
using extended_op_profiler::ExtendedOpProfiler;


class InvalidPartitionException : public std::exception {
public:
    InvalidPartitionException(const char* message) : m_message(message) {}

    virtual const char* what() const noexcept override {
        return m_message;
    }

private:
    const char* m_message;
};

struct PartitionInstr {
  int n_part;
};

enum class ScheduleType {
  kPrelude, kPipeline, kEpilogue, kArbitrary
};

std::ostream& operator << (std::ostream &os, const ScheduleType &st);

struct ScheduleLocation {
  ScheduleType type;
  int partition_index;
};

std::ostream& operator << (std::ostream &os, const ScheduleLocation &location);

ScheduleLocation ScheduleAtPrelude();
ScheduleLocation ScheduleAtPipelineIndex(int index);
ScheduleLocation ScheduleAtEpilogue();
ScheduleLocation ArbitrarySchedule();

std::string PrintVarAndExpr(const VarAndExpr& vae);

std::string PrintVarAndExprs(const VarAndExprs& vaes);

std::string PrintInstr(const PartitionInstr& instr);

struct PartitionedANFBlock {
  // a list of exprs that should be prepended to the pipeline.
  // this includes any input splitting / transforming exprs
  VarAndExprs prelude;
  // a list of exprs that should be appended after the pipeline
  // this includes any output merging exprs
  VarAndExprs epilogue;
  // partitioned exprs. each VarAndExprs corresponds to a partition
  // for exprs that require multiple partitioned instr to form a partition, we assume that the last expr
  // in its corresponding VarAndExprs corresponds to the original output
  std::vector<VarAndExprs> partitions;
  // index of the expr which serves as the output of the partitioned ANF block
  // in partitions, for each partition
  std::vector<int> partition_output_indices;

  void PushPrelude(Var var, Expr expr);
  void PushEpilogue(Var var, Expr expr);

  void PushExprsToPartition(const Var& var, const Expr& expr, Index partition_idx, bool is_output = true);
  void PushExprsToPartition(const VarAndExpr& vae, Index partition_idx, bool is_output = true);
  void PushExprsToPartition(const VarAndExprs& vaes, Index partition_idx, bool is_output = true);

  void MergePrelude(const PartitionedANFBlock& other);
  void Merge(const PartitionedANFBlock& other);
  void MergeEpilogue(const PartitionedANFBlock& other);

  VarAndExprs GetPartitionOutputs() const;
  VarAndExpr GetEpilogueOutputs() const;

  static PartitionedANFBlock MakeIdentity(Var var, Expr expr, int n_partitions);
};

std::string PrintPartitionedANFBlock(const PartitionedANFBlock& pblock);

class PartitionEnv {
 public:
  explicit PartitionEnv(ScheduledDFG& sched_dfg,
                        ExtendedOpProfiler& op_profiler,
                        const NodeMap<MoENodeLabel>& moe_label_map,
                        const ExprMap<InOutAxes>& expr_inout_axes,
                        const ExprMap<Exprs>& expr_func_exprs,
                        const ExprMap<IndexMap<Index>>& expr_arg_var,
                        const ExprMap<ExprMap<ExprMap<Index>>>& expr_arg_consumer_index,
                        int dp_group_size, int n_part);

  ExprMap<PartitionedANFBlock> GetPartitionedExprs() const;

  VarAndExprs GetPartitionedOutputs(Expr expr, Indices& axes);

  bool HasPartitionedExprs(Expr expr, Indices& axes) const;
  PartitionedANFBlock GetPartitionedExprs(Expr expr, Indices& axes) const;
  void SetPartitionedExprs(Expr expr, Indices& axes, PartitionedANFBlock pblock);

  bool HasArgVar(Expr expr, Index index) const;
  Index GetArgVar(Expr expr, Index index) const;
  void SetArgVar(Expr expr, Index arg, Index var);

  bool HasUpdatedArg(Expr expr, Index index) const;
  Expr GetUpdatedArg(Expr expr, Index index) const;
  void SetUpdatedArg(Expr expr, Index index, Expr updated_arg);

  ExprMap<Index> GetConsumerIndex(Expr expr, Expr arg) const;

  bool HasInOutAxes(Expr expr) const;
  Indices GetInAxes(Expr expr, Index index) const;
  Indices GetArgInAxes(Expr expr, Index index) const;
  Indices GetOutAxes(Expr expr) const;

  Expr GetExprFromVar(Var var) const;
  Var GetVarFromExpr(Expr expr);

  PartitionInstr GetPartitionInstr(Expr expr) const;
  Exprs GetFuncExprsFromCaller(Expr expr) const;
  Exprs GetAllFuncExprs() const;

  ExprMap<ScheduleLocation> GetScheduleLocations() const;
  void SetScheduleLocation(Expr expr, ScheduleLocation loc);
  void RemoveScheduleLocation(Expr expr);

  MoENodeLabel GetMoENodeLabel(Expr expr) const;

  void RemoveExpr(Expr expr);

  void AddSynchronization(Expr input_expr, Expr expr, Indices axes);

  void AccumulateProfileElapsedTime(const std::chrono::milliseconds::rep& time);
  std::chrono::milliseconds::rep GetProfileElapsedTime() const;

  int GetDPGroupSize() const;

  // reference to the scheduled dfg in PartitionEnvBuilder
  ScheduledDFG& sched_dfg;
  ExtendedOpProfiler& op_profiler;

 private:
  Expr GetAndCheckExpr_(Expr expr) const;
  void SetAxesExprs_(int n_part);
  void SetFuncExprs_();
  void SetInstr_(const ExtendedDFG& dfg, int dp_group_size, int n_part);

  // expr -> calculated input and output partition axes
  ExprMap<InOutAxes> expr_inout_axes_;

  // partitioned expr -> their schedule locations
  // this map should be filled for every operator that needs to be scheduled
  // i.e. partitioned operators + all overhead operators
  ExprMap<ScheduleLocation> partitioned_expr_schedule_loc_;

  // expr -> expr in the function called
  ExprMap<Exprs> expr_func_exprs_;
  // expr -> input arg index -> var index in in_indices
  ExprMap<IndexMap<Index>> expr_arg_var_;
  // original expr -> arg -> consumer -> input arg index 
  ExprMap<ExprMap<ExprMap<Index>>> expr_arg_consumer_index_;
  // expr -> arg index -> updated non tensor / tuple args
  // When calling FPartition, set the updated arguments.
  // Would be used when partitioning the function.
  ExprMap<IndexMap<Expr>> expr_index_updated_arg_;

  // original expr -> partition axes -> partitioned vars and exprs
  ExprMap<HashIndexMap<PartitionedANFBlock>> expr_axes_exprs_;
  // expr -> instr indicating how it should be partitioned
  ExprMap<PartitionInstr> expr_instr_;
  // expr -> var for func exprs
  ExprMap<Var> func_expr_var_;
  // var -> expr for func exprs
  ExprMap<Expr> func_var_expr_;

  ExprMap<Vars> expr_dispatch_mask_;
  const NodeMap<MoENodeLabel>& moe_label_map_;

  std::chrono::milliseconds::rep profile_total_ = 0;

  int dp_group_size_;

  bool record_schedule_location_ = true;
};

class PartitionEnvBuilder : public ExprVisitor {
 public:
  PartitionEnvBuilder(ScheduledDFG& sched_dfg, ExtendedOpProfiler& op_profiler, const NodeMap<MoENodeLabel>& moe_label_map,
                      const CPSolution& solution, int dp_group_size, int n_part, int n_experts);

  PartitionEnv Build(const Exprs& exprs);
  void VisitExpr_(const LetNode* let) override;
  void VisitExpr_(const TupleNode* tuple) override;
  void VisitExpr_(const TupleGetItemNode* tgi) override;
  void VisitExpr_(const CallNode* call) override;
  
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const FunctionNode* func) override;
  
  void VisitExpr(const Expr& expr);

 private:
  void Setup_();
  void Finalize_(const Exprs& exprs);

  // the sched_dfg will be mutated alongside
  ScheduledDFG& sched_dfg_;

  void PartitionFunction_();

  // Current call stack.
  Exprs stack_;
  PartitionEnv env_;
  int n_experts_;
};

struct PartitionResult {
  PartitionResult(ScheduledDFG&& sched_dfg, ExprMap<ScheduleLocation>&& expr_schedule_locations);
  ScheduledDFG sched_dfg;
  ExprMap<ScheduleLocation> expr_schedule_locations;
};

std::ostream& operator << (std::ostream &os, const PartitionResult &result);

bool HasFPartition(std::string op);

PartitionResult PartitionNodes(const ScheduledDFG& sched_dfg, const Nodes& nodes, const CPSolution& solution, ExtendedOpProfiler& op_profiler, int dp_group_size, int n_part, int n_experts, const NodeMap<MoENodeLabel>& moe_label_map = {});
PartitionResult PartitionNodes(const ScheduledDFG& sched_dfg, const NodeSet& nodes, const CPSolution& solution, ExtendedOpProfiler& op_profiler, int dp_group_size, int n_part, int n_experts, const NodeMap<MoENodeLabel>& moe_label_map = {});

class PhasedScheduleGenerator: public ScheduleGenerator<CompareNode, StableStaticPriorityNodeQueue> {
public:
    PhasedScheduleGenerator(const ExtendedDFG& G, CompareNode& comp_comparator, CompareNode& comm_comparator, const NodeMap<ScheduleLocation>& node_phase_map, NodeSet node_mask = {})
    : ScheduleGenerator<CompareNode, StableStaticPriorityNodeQueue>(G, comp_comparator, comm_comparator, node_mask), node_phase_map_(node_phase_map) {
        for (auto& it: node_phase_map_) {
          int node_phase = ScheduleLocationToPhaseId(it.second);
          max_phase_ = std::max(max_phase_, node_phase);
          if (dfg_.getNodeType(it.first) == NodeType::kCompNode) {
              if (!per_phase_comp_priority_queue_.count(node_phase)) {
                  per_phase_comp_priority_queue_.insert({node_phase, StableStaticPriorityNodeQueue(comp_comparator)});
              }
          } else {
              if (!per_phase_comm_priority_queue_.count(node_phase)) {
                  per_phase_comm_priority_queue_.insert({node_phase, StableStaticPriorityNodeQueue(comm_comparator)});
              }
          }
        }
        for (auto i: {-2, -1, 0}) {
          if (!per_phase_comp_priority_queue_.count(i)) {
              per_phase_comp_priority_queue_.insert({i, StableStaticPriorityNodeQueue(comp_comparator)});
          }
          if (!per_phase_comm_priority_queue_.count(i)) {
              per_phase_comm_priority_queue_.insert({i, StableStaticPriorityNodeQueue(comm_comparator)});
          }
        }
        _resetState();
    }

    // override the Run method
    SimulateTimeType Run(NodeMap<NodeDuration>& node_durations) {
        _fetchNodeForExecution(node_durations);

        while (!pending_events_.empty()) {
            auto current_time_and_task = pending_events_.top();
            SimulateTimeType current_time = current_time_and_task.first;
            const Node* current_task = current_time_and_task.second;
            node_durations[current_task].end = current_time;
            pending_events_.pop();

            _updateStateAtTaskFinish(current_task, current_time, node_durations);

            _fetchNodeForExecution(node_durations);
        }

        NodeSet executed_nodes;
        Nodes unexecuted_nodes;
        for(auto& it: node_executed_) {
            if (!it.second) {
                unexecuted_nodes.push_back(it.first);
            } else {
                executed_nodes.insert(it.first);
            }
        }
        if(!unexecuted_nodes.empty()) {
            LOG(INFO) << "Found unexecuted nodes in DFG: ";
            for (auto node: unexecuted_nodes) {
                LOG(INFO) << "\t -> " << dfg_.getNodeNameOrDefault(node) << " unexecuted.";
                LOG(INFO) << "\t\tBlockers: ";
                for (auto child: dfg_.getChildren(node)) {
                    if(_shouldConsiderNode(child) && !executed_nodes.count(child)) {
                        LOG(INFO) << "\t\t -> " << dfg_.getNodeNameOrDefault(child);
                    }
                }
            }
            LOG(INFO) << "Executed nodes: ";
            for (auto node: node_launch_order_) {
                if (executed_nodes.count(node)) {
                    LOG(INFO) << "\t -> " << dfg_.getNodeNameOrDefault(node);
                }
            }
            LOG(FATAL) << "Incomplete execution of the dependency graph!" << std::endl;
        }
        has_result_ = true;
        return device_time_;
    }

    void _updateStateAtTaskFinish(const Node* current_task,
                                  SimulateTimeType current_time, NodeMap<NodeDuration>& node_durations) {
        node_executed_[current_task] = true;
        if (dfg_.getNodeType(current_task) == NodeType::kCompNode) {
            CHECK(comp_device_busy_) << "Expected comp_device_busy_ == true, but got false";
            comp_device_busy_ = false;
        } else {
            CHECK(comm_device_busy_) << "Expected comm_device_busy_ == true, but got false";
            comm_device_busy_ = false;
        }
        for(auto parent: dfg_.getParents(current_task)) {
            if(!_shouldConsiderNode(parent)) {
                continue;
            }
            CHECK(pred_counter_.count(parent));
            pred_counter_[parent]--;
            if(pred_counter_[parent] == 0) {
                node_durations[parent].blocked_by = current_task;
                node_durations[parent].ready_time = current_time;
                CHECK(node_phase_map_.count(parent)) << "Node " << dfg_.getNodeNameOrDefault(parent) << " not found in node_phase_map.";
                int phase = ScheduleLocationToPhaseId(node_phase_map_.at(parent));
                if (dfg_.getNodeType(parent) == NodeType::kCompNode) {
                  CHECK(per_phase_comp_priority_queue_.count(phase));
                  per_phase_comp_priority_queue_.at(phase).Push(parent);
                } else {
                  CHECK(per_phase_comp_priority_queue_.count(phase));
                  per_phase_comm_priority_queue_.at(phase).Push(parent);
                }
            }
        }
        device_time_ = current_time;
    }

    void _fetchNodeForExecution(NodeMap<NodeDuration>& node_durations) {
        int n_phase_inspected = 0;
        while (!comp_device_busy_) {
          CHECK(per_phase_comp_priority_queue_.count(current_comp_phase_)) << "Phase " << current_comp_phase_ << " not found in per_phase_comp_priority_queue_.";
          if (!per_phase_comp_priority_queue_.at(current_comp_phase_).Empty()) {
              auto comp_task = per_phase_comp_priority_queue_.at(current_comp_phase_).Top();
              pending_events_.push(
                  std::make_pair(device_time_ + dfg_.getNodeExecTime(comp_task), comp_task));
              node_durations[comp_task].start = device_time_;
              if (device_time_ > node_durations[comp_task].ready_time) {
                  CHECK(!comp_launch_order_.empty());
                  node_durations[comp_task].blocked_by = comp_launch_order_[comp_launch_order_.size()-1];
              }
              comp_device_busy_ = true;
              per_phase_comp_priority_queue_.at(current_comp_phase_).Pop();
              node_launch_order_.push_back(comp_task);
              comp_launch_order_.push_back(comp_task);
              break;
          }
          current_comp_phase_++;
          current_comp_phase_ = current_comp_phase_ % (max_phase_ + 1);
          n_phase_inspected++;
          if (n_phase_inspected > max_phase_ + 1) {
              break;
          }
        }
        // also check arbitrary phase and epilogue phase
        for (auto phase: {-2, -1}) {
          if (!comp_device_busy_) {
              CHECK(per_phase_comp_priority_queue_.count(phase)) << "Phase " << phase << " not found in per_phase_comp_priority_queue_.";
              if (!per_phase_comp_priority_queue_.at(phase).Empty()) {
                  auto comp_task = per_phase_comp_priority_queue_.at(phase).Top();
                  pending_events_.push(
                      std::make_pair(device_time_ + dfg_.getNodeExecTime(comp_task), comp_task));
                  node_durations[comp_task].start = device_time_;
                  if (device_time_ > node_durations[comp_task].ready_time) {
                      CHECK(!comp_launch_order_.empty());
                      node_durations[comp_task].blocked_by = comp_launch_order_[comp_launch_order_.size()-1];
                  }
                  comp_device_busy_ = true;
                  per_phase_comp_priority_queue_.at(phase).Pop();
                  node_launch_order_.push_back(comp_task);
                  comp_launch_order_.push_back(comp_task);
              }
          }
        }
        n_phase_inspected = 0;
        while(!comm_device_busy_) {
            CHECK(per_phase_comm_priority_queue_.count(current_comm_phase_)) << "Phase " << current_comm_phase_ << " not found in per_phase_comm_priority_queue_.";
            if (!per_phase_comm_priority_queue_.at(current_comm_phase_).Empty()) {
                auto comm_task = per_phase_comm_priority_queue_.at(current_comm_phase_).Top();
                pending_events_.push(
                    std::make_pair(device_time_ + dfg_.getNodeExecTime(comm_task), comm_task));
                node_durations[comm_task].start = device_time_;
                if (device_time_ > node_durations[comm_task].ready_time) {
                    CHECK(!comm_launch_order_.empty());
                    node_durations[comm_task].blocked_by = comm_launch_order_[comm_launch_order_.size()-1];
                }
                comm_device_busy_ = true;
                per_phase_comm_priority_queue_.at(current_comm_phase_).Pop();
                node_launch_order_.push_back(comm_task);
                comm_launch_order_.push_back(comm_task);
                per_phase_comm_priority_queue_.at(current_comm_phase_).SetPrevEle(comm_task);
                break;
            }
            current_comm_phase_++;
            current_comm_phase_ = current_comm_phase_ % (max_phase_ + 1);
            n_phase_inspected++;
            if (n_phase_inspected > max_phase_ + 1) {
                break;
            }
        }
        // also check arbitrary phase and epilogue phase
        for (auto phase: {-2, -1}) {
          if (!comm_device_busy_) {
              CHECK(per_phase_comm_priority_queue_.count(phase)) << "Phase " << phase << " not found in per_phase_comm_priority_queue_.";
              if (!per_phase_comm_priority_queue_.at(phase).Empty()) {
                  auto comm_task = per_phase_comm_priority_queue_.at(phase).Top();
                  pending_events_.push(
                      std::make_pair(device_time_ + dfg_.getNodeExecTime(comm_task), comm_task));
                  node_durations[comm_task].start = device_time_;
                  if (device_time_ > node_durations[comm_task].ready_time) {
                      CHECK(!comm_launch_order_.empty());
                      node_durations[comm_task].blocked_by = comm_launch_order_[comm_launch_order_.size()-1];
                  }
                  comm_device_busy_ = true;
                  per_phase_comm_priority_queue_.at(phase).Pop();
                  node_launch_order_.push_back(comm_task);
                  comm_launch_order_.push_back(comm_task);
                  per_phase_comm_priority_queue_.at(phase).SetPrevEle(comm_task);
              }
          }
        }
    }

protected:
    void _resetState() {
        // initialize pred_counter_ and ready queues
        // calculate indegree for each node
        for(auto node: dfg_.nodes()) {
            if(!_shouldConsiderNode(node)) {
                continue;
            }
            int indegree = 0;
            for(auto child: dfg_.getChildren(node)) {
                if(_shouldConsiderNode(child)) {
                    indegree ++;
                }
            }
            pred_counter_[node] = indegree;
            if (indegree == 0) {
                int phase = ScheduleLocationToPhaseId(node_phase_map_.at(node));
                if (dfg_.getNodeType(node) == NodeType::kCompNode) {
                  CHECK(per_phase_comp_priority_queue_.count(phase));
                  per_phase_comp_priority_queue_.at(phase).Push(node);
                  comp_queue_total_time += dfg_.getNodeExecTime(node);
                } else {
                  CHECK(per_phase_comp_priority_queue_.count(phase));
                  per_phase_comm_priority_queue_.at(phase).Push(node);
                  comm_queue_total_time += dfg_.getNodeExecTime(node);
                }
            }
            node_executed_[node] = false;
        }
        device_time_ = 0;
        comp_device_busy_ = false;
        comm_device_busy_ = false;
        has_result_ = false;
        node_launch_order_.clear();
        comm_launch_order_.clear();
        comp_launch_order_.clear();
    }

    int ScheduleLocationToPhaseId(const ScheduleLocation& loc) {
        if (loc.type == ScheduleType::kPrelude) {
            return 0;
        } else if (loc.type == ScheduleType::kPipeline) {
            return loc.partition_index + 1;
        } else if (loc.type == ScheduleType::kEpilogue) {
            return -1;
        } else if (loc.type == ScheduleType::kArbitrary) {
            return -2;
        }
        LOG(FATAL) << "Unknown schedule type: " << loc.type;
        return -1;
    }

    const NodeMap<ScheduleLocation>& node_phase_map_;
    int max_phase_ = 0;
    int current_comp_phase_ = 0;
    int current_comm_phase_ = 0;
    std::unordered_map<int, StableStaticPriorityNodeQueue> per_phase_comp_priority_queue_;
    std::unordered_map<int, StableStaticPriorityNodeQueue> per_phase_comm_priority_queue_;
};

} // namespace partition_exprs
} // namespace pass
} // namespace raf