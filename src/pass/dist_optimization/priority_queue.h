/*!
 * Copyright (c) 2022 by Contributors
 * \file priority_queue.h
 * \brief Stable priority queue implementation for scheduling.
 */
#pragma once
#include "./scheduler_common.h"
#include "./extended_dfg.h"

namespace raf {
namespace pass {
namespace priority_queue {

using namespace raf::pass::scheduler_common;
using namespace raf::pass::extended_dfg;

template<class T,
         class TCompare>
class StablePriorityQueue final {
    /*
        A wrapper for stl priority queue to make it stable (i.e. for items with the same priority,
        the popping order is the same as pushing order). This wrapper adds a logical timestamp to
        each elements inserted while still using stl priority queue underneath. Note that the
        timestamp is stored using a uint64_t counter, which means that this queue will no longer
        function after UINT64_MAX number of pushes. An error is thrown if such limit is reached.
        TCompare must provide function less(lhs, rhs) and equal(lhs, rhs).
    */
public:
    using TElem = std::pair<T, uint64_t>;
    StablePriorityQueue(TCompare& comparator) :
        comparator_(comparator),
        stl_priority_queue_(WrappedTCompare(comparator)) {}
    StablePriorityQueue(TCompare&& comparator) :
        comparator_(comparator),
        stl_priority_queue_(WrappedTCompare(comparator)) {}

    void SetDynamicHint(double comp_hint, double comm_hint) {}

    void Push(const T& value) {
        if(timestamp_ == UINT64_MAX - 1) {
            LOG(FATAL) << "Timestamp depleted in StablePriorityQueue.";
        }
        auto telem = std::make_pair(value, timestamp_);
        stl_priority_queue_.push(telem);
        timestamp_ ++;
    }

    void Pop() {
        stl_priority_queue_.pop();
    }

    const T& Top() {
        return stl_priority_queue_.top().first;
    }

    bool Empty() {
        return stl_priority_queue_.empty();
    }

    size_t Size() {
        return stl_priority_queue_.size();
    }

    void SetPrevEle(const T& ele) {}
private:
    class WrappedTCompare {
    public:
        WrappedTCompare(TCompare& comparator): orig_comparator_(comparator) {}
        bool operator()(TElem& o1, TElem& o2) {
            if(orig_comparator_.less(o1.first, o2.first)) {
                return true;
            } else if (orig_comparator_.equal(o1.first, o2.first)) {
                CHECK_NE(o1.second, o2.second) << "Encountered identical timestamp.";
                return o1.second > o2.second;
            } else {
                return false;
            }
        }
        TCompare& orig_comparator_;
    };
    std::priority_queue<TElem, std::deque<TElem>, WrappedTCompare> stl_priority_queue_;
    uint64_t timestamp_ = 0;
    TCompare comparator_;
};

class CompareNode {
public:
    CompareNode(const NodeMap<double>& priority) : priority_(priority) {}
    CompareNode(const CompareNode& other): priority_(other.priority_) {}
    CompareNode(CompareNode&& other): priority_(other.priority_) {}
    bool less(const Node* lhs, const Node* rhs) {
        CHECK(priority_.count(lhs) && priority_.count(rhs));
        return priority_.at(lhs) < priority_.at(rhs);
    }
    bool equal(const Node* lhs, const Node* rhs) {
        CHECK(priority_.count(lhs) && priority_.count(rhs));
        return priority_.at(lhs) == priority_.at(rhs);
    }
    void set_dynamic_hint(double comp_hint, double comm_hint) {}
    const NodeMap<double>& priority_;
};

class CompareNodeDynamic {
public:
    CompareNodeDynamic(const ExtendedDFG& dfg, const NodeMap<double>& heft_priority,
                        const NodeMap<double>& unlock_length_priority,
                        double lambda_comp, double lambda_comm, double gamma, double theta_comp, double theta_comm, double beta) :
                        dfg_(dfg), heft_priority_(heft_priority),
                        unlock_length_priority_(unlock_length_priority),
                        lambda_comp_(lambda_comp), lambda_comm_(lambda_comm), gamma_(gamma),
                        theta_comp_(theta_comp), theta_comm_(theta_comm), beta_(beta) {}

    CompareNodeDynamic(const CompareNodeDynamic& other):
        dfg_(other.dfg_),
        heft_priority_(other.heft_priority_),
        unlock_length_priority_(other.unlock_length_priority_),
        lambda_comp_(other.lambda_comp_), lambda_comm_(other.lambda_comm_), gamma_(other.gamma_),
        theta_comp_(other.theta_comp_), theta_comm_(other.theta_comm_), beta_(other.beta_) {}

    CompareNodeDynamic(CompareNodeDynamic&& other):
        dfg_(other.dfg_),
        heft_priority_(other.heft_priority_),
        unlock_length_priority_(other.unlock_length_priority_),
        lambda_comp_(other.lambda_comp_), lambda_comm_(other.lambda_comm_), gamma_(other.gamma_),
        theta_comp_(other.theta_comp_), theta_comm_(other.theta_comm_), beta_(other.beta_) {}

    bool less(const Node* lhs, const Node* rhs) {
        return get_priority(lhs) < get_priority(rhs);
    }

    bool equal(const Node* lhs, const Node* rhs) {
        return get_priority(lhs) == get_priority(rhs);
    }

    double get_priority(const Node* n) {
        double node_hint = 0;
        double cur_theta = 0;
        double cur_lambda = 0;
        double cur_beta = 0;
        if (dfg_.getNodeType(n) == NodeType::kCompNode) {
            node_hint = _comp_hint;
            cur_theta = theta_comp_;
            cur_lambda = lambda_comp_;
        } else {
            node_hint = _comm_hint;
            cur_theta = theta_comm_;
            cur_lambda = lambda_comm_;
            CommType comm_type = IdentifyCommType(dfg_.getCommSize(n));
            if (comm_type == prev_comm_type_) {
                // LOG(INFO) << "comm type == prev comm type.";
                cur_beta = beta_;
            }
        }
        CHECK(heft_priority_.count(n));
        CHECK(unlock_length_priority_.count(n));
        return ((gamma_ * node_hint / 10000.0 + cur_lambda) * (heft_priority_.at(n)) / 10000.0 +
                (1 - cur_lambda - gamma_ * node_hint / 10000.0) * cur_theta / (unlock_length_priority_.at(n) + 1.0)) * (1.0 + cur_beta * 10000.0);
    }

    void set_dynamic_hint(double comp_hint, double comm_hint) {
        _comp_hint = comp_hint;
        _comm_hint = comm_hint;
    }

    void set_prev_comm(const Node* comm_node) {
        prev_comm_type_ = IdentifyCommType(dfg_.getCommSize(comm_node));
    }

    const ExtendedDFG& dfg_;
    const NodeMap<double>& heft_priority_;
    const NodeMap<double>& unlock_length_priority_;
    const double lambda_comp_, lambda_comm_, gamma_, theta_comp_, theta_comm_, beta_;
    CommType prev_comm_type_;
    double _comm_hint = 0;
    double _comp_hint = 0;
};

template<class T, class TCompare, class... Args>
class DynamicPriorityQueue final {
    /*
        A utility class for mimicing a priority queue but the priority of all nodes are dynamically
        changing. We implement it by iterating through the contents and get the maximum. Note that
        this O(n) per Top() solution is already the best we can do, since re-calculating the priority
        for all elements is already O(n). TCompare must provide function less(lhs, rhs) and
        equal(lhs, rhs).
    */
public:
    DynamicPriorityQueue(TCompare& comparator) : comparator_(comparator) {}
    DynamicPriorityQueue(TCompare&& comparator) : comparator_(comparator) {}

    void SetDynamicHint(double comp_hint, double comm_hint) {
        // invalidate prev max
        previous_max_valid = false;
        return comparator_.set_dynamic_hint(comp_hint, comm_hint);
    }

    void Push(const T& value) {
        // directly append to list
        previous_max_valid = false;
        elements_linked_list_.push_back(value);
    }

    void Pop() {
        if(previous_max_valid) {
            elements_linked_list_.erase(previous_max_);
        } else {
            auto max_iter = GetTopIterator();
            elements_linked_list_.erase(max_iter);
        }
        previous_max_valid = false;
    }

    const T& Top() {
        if(previous_max_valid) {
            return (*previous_max_);
        } else {
            auto max_iter = GetTopIterator();
            return (*max_iter);
        }
    }

    bool Empty() {
        return elements_linked_list_.empty();
    }

    size_t Size() {
        return elements_linked_list_.size();
    }

    void SetPrevEle(const T& ele) {
        comparator_.set_prev_comm(ele);
    }

private:
    typename std::list<T>::iterator GetTopIterator() {
        T max_value = elements_linked_list_.front();
        typename std::list<T>::iterator max_iter = elements_linked_list_.begin();
        for(typename std::list<T>::iterator it = elements_linked_list_.begin();
            it != elements_linked_list_.end();
            it++) {
            if(comparator_.less(max_value, (*it))) {
                max_iter = it;
                max_value = (*it);
            }
            // don't update if equal to max value
        }
        previous_max_ = max_iter;
        previous_max_valid = true;
        return max_iter;
    }
    std::list<T> elements_linked_list_;
    TCompare comparator_;
    typename std::list<T>::iterator previous_max_;
    bool previous_max_valid = false;
};

using StableStaticPriorityNodeQueue = StablePriorityQueue<const Node*, CompareNode>;
using DynamicPriorityNodeQueue = DynamicPriorityQueue<const Node*, CompareNodeDynamic>;

}  // namespace priority_queue
}  // namespace pass
}  // namespace raf