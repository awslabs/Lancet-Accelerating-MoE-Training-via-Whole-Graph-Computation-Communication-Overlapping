/*!
 * Copyright (c) 2022 by Contributors
 * \file schedule_generator.h
 * \brief Schedule generator.
 */
#pragma once
#include <functional>
#include "./scheduler_common.h"
#include "./priority_queue.h"

namespace raf {
namespace pass {
namespace schedule_generator {

using namespace raf::pass::scheduler_common;
using namespace raf::pass::priority_queue;

using QueueEntry = std::pair<SimulateTimeType, const Node*>;

template <class TCompareNode, class TPriorityQueue>
class ScheduleGenerator {
   public:
    class CompareTime {
    public:
        bool operator()(QueueEntry lhs, QueueEntry rhs) {
            return lhs.first > rhs.first;
        }
    };

    ScheduleGenerator(const ExtendedDFG& G, TCompareNode& comp_comparator, TCompareNode& comm_comparator, NodeSet node_mask = {})
        : dfg_(G),
          node_mask_(node_mask),
          comp_ready_queue_(comp_comparator),
          comm_ready_queue_(comm_comparator),
          comp_delayed_ready_queue_(comp_comparator),
          comm_delayed_ready_queue_(comm_comparator) {
        _resetState();
    };

    // runs the generator, fill in the node start times and modifies the input
    // graph G. returns the makespan.
    SimulateTimeType Run(NodeMap<NodeDuration>& node_durations, NodeSet nodes_to_delay={}) {
        _fetchNodeForExecution(node_durations);

        while (!pending_events_.empty()) {
            auto current_time_and_task = pending_events_.top();
            SimulateTimeType current_time = current_time_and_task.first;
            const Node* current_task = current_time_and_task.second;
            node_durations[current_task].end = current_time;
            pending_events_.pop();

            _updateStateAtTaskFinish(current_task, current_time, node_durations, nodes_to_delay);

            _fetchNodeForExecution(node_durations);
        }

        // another round that considers the delayed nodes
        while (!comp_delayed_ready_queue_.Empty()) {
            auto comp_task = comp_delayed_ready_queue_.Top();
            comp_ready_queue_.Push(comp_task);
            comp_delayed_ready_queue_.Pop();
        }
        while (!comm_delayed_ready_queue_.Empty()) {
            auto comm_task = comm_delayed_ready_queue_.Top();
            comm_ready_queue_.Push(comm_task);
            comm_delayed_ready_queue_.Pop();
        }
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

    std::tuple<Nodes, Nodes, Nodes> GetNodeOrder() {
        if(has_result_) {
            return std::make_tuple(node_launch_order_, comp_launch_order_, comm_launch_order_);
        } else {
            return {};
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
                if (dfg_.getNodeType(node) == NodeType::kCompNode) {
                    comp_ready_queue_.Push(node);
                    comp_queue_total_time += dfg_.getNodeExecTime(node);
                } else {
                    comm_ready_queue_.Push(node);
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

    void _updateStateAtTaskFinish(const Node* current_task,
                                  SimulateTimeType current_time, NodeMap<NodeDuration>& node_durations,
                                  const NodeSet& nodes_to_delay = {}) {
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
                if (dfg_.getNodeType(parent) == NodeType::kCompNode) {
                    if (nodes_to_delay.count(parent)) {
                        comp_delayed_ready_queue_.Push(parent);
                    } else {
                        comp_ready_queue_.Push(parent);
                        comp_queue_total_time += dfg_.getNodeExecTime(parent);
                    }
                } else {
                    if (nodes_to_delay.count(parent)) {
                        comm_delayed_ready_queue_.Push(parent);
                    } else {
                        comm_ready_queue_.Push(parent);
                        comm_queue_total_time += dfg_.getNodeExecTime(parent);
                    }
                }
            }
        }
        device_time_ = current_time;
        double comp_hint = comm_queue_total_time;
        double comm_hint = comp_queue_total_time;
        comp_ready_queue_.SetDynamicHint(/*comp_hint=*/comm_queue_total_time,
                                         /*comm_hint=*/comp_queue_total_time);
    }

    void _fetchNodeForExecution(NodeMap<NodeDuration>& node_durations) {
        if (!comp_device_busy_ && !comp_ready_queue_.Empty()) {
            auto comp_task = comp_ready_queue_.Top();
            pending_events_.push(
                std::make_pair(device_time_ + dfg_.getNodeExecTime(comp_task), comp_task));
            node_durations[comp_task].start = device_time_;
            if (device_time_ > node_durations[comp_task].ready_time) {
                CHECK(!comp_launch_order_.empty());
                node_durations[comp_task].blocked_by = comp_launch_order_[comp_launch_order_.size()-1];
            }
            comp_device_busy_ = true;
            comp_ready_queue_.Pop();
            node_launch_order_.push_back(comp_task);
            comp_launch_order_.push_back(comp_task);
        }
        if (!comm_device_busy_ && !comm_ready_queue_.Empty()) {
            auto comm_task = comm_ready_queue_.Top();
            pending_events_.push(
                std::make_pair(device_time_ + dfg_.getNodeExecTime(comm_task), comm_task));
            node_durations[comm_task].start = device_time_;
            if (device_time_ > node_durations[comm_task].ready_time) {
                CHECK(!comm_launch_order_.empty());
                node_durations[comm_task].blocked_by = comm_launch_order_[comm_launch_order_.size()-1];
            }
            comm_device_busy_ = true;
            comm_ready_queue_.Pop();
            node_launch_order_.push_back(comm_task);
            comm_launch_order_.push_back(comm_task);
            comm_ready_queue_.SetPrevEle(comm_task);
        }
    }

    bool _shouldConsiderNode(const Node* node) {
        if(node_mask_.empty()) {
            return true;
        }
        return node_mask_.count(node);
    }

    const ExtendedDFG& dfg_;
    NodeMap<int> pred_counter_;

    NodeSet node_mask_;

    TPriorityQueue comp_ready_queue_;
    TPriorityQueue comm_ready_queue_;
    TPriorityQueue comp_delayed_ready_queue_;
    TPriorityQueue comm_delayed_ready_queue_;
    std::priority_queue<QueueEntry,
                        std::deque<QueueEntry>,
                        CompareTime>
        pending_events_;

    SimulateTimeType device_time_ = 0;
    bool comp_device_busy_ = false;
    bool comm_device_busy_ = false;
    Nodes node_launch_order_;
    Nodes comp_launch_order_;
    Nodes comm_launch_order_;

    SimulateTimeType comp_queue_total_time = 0;
    SimulateTimeType comm_queue_total_time = 0;

    std::unordered_map<const Node*, bool> node_executed_;
    bool has_result_ = false;
};

using StaticScheduleGenerator = ScheduleGenerator<CompareNode, StableStaticPriorityNodeQueue>;
using DynamicScheduleGenerator = ScheduleGenerator<CompareNodeDynamic, DynamicPriorityNodeQueue>;

class DynamicScheduleParamsNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  double lambda_comp;
  double lambda_comm;
  double gamma;
  double theta_comp;
  double theta_comm;
  double beta;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("lambda_comp", &lambda_comp);
    v->Visit("lambda_comm", &lambda_comm);
    v->Visit("gamma", &gamma);
    v->Visit("theta_comp", &theta_comp);
    v->Visit("theta_comm", &theta_comm);
    v->Visit("beta", &beta);
  }

  static constexpr const char* _type_key = "raf.distributed.DynamicScheduleParams";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(DynamicScheduleParamsNode, Object);
};

class DynamicScheduleParams : public ObjectRef {
 public:
  DynamicScheduleParams(double lambda_comp, double lambda_comm, double gamma, double theta_comp, double theta_comm, double beta);
  static DynamicScheduleParams make(double lambda_comp, double lambda_comm, double gamma, double theta_comp, double theta_comm, double beta);
  TVM_DEFINE_OBJECT_REF_METHODS(DynamicScheduleParams, ObjectRef, DynamicScheduleParamsNode);
};

void SetScheduleEvalFunc(std::function<double(double, double, double, double, double, double)>* func);

double DynamicEvalCurrentSchedule_(double lamb_comp, double lamb_comm, double gamma, double theta_comp, double theta_comm, double beta);

}  // namespace schedule_generator
}  // namespace pass
}  // namespace raf