/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * Copyright (c) 2022 by Contributors
 * \file extended_dfg.h
 * \brief Extension to DFG.
 */

#include <algorithm>
#include "raf/ir.h"
#include "raf/analysis.h"
#include "../common.h"
#include "../../analysis/dependency_graph.h"
#include "./scheduler_common.h"
#include "extended_dfg.h"

namespace raf {
namespace pass {
namespace extended_dfg {

using namespace raf::pass::scheduler_common;

using NodeRelationMap = std::unordered_map<const Node*, NodeSet>;

void fillParentChildrenMapFromNodes(NodeSet nodes, NodeRelationMap& parents_map, NodeRelationMap& children_map) {
    for(auto node: nodes) {
        parents_map[node] = {};
        children_map[node] = {};
        for(auto child = node->children.head; child != nullptr; child = child->next) {
            if(nodes.count(child->value) && child->value != node) {
                children_map[node].insert(child->value);
            }
        }
        for(auto parent = node->parents.head; parent != nullptr; parent = parent->next) {
            if(nodes.count(parent->value) && parent->value != node) {
                parents_map[node].insert(parent->value);
            }
        }
    }
}

void fillParentChildrenMapFromNodes(const std::vector<Node*> nodes, NodeRelationMap& parents_map, NodeRelationMap& children_map) {
    NodeSet nodes_set;
    for(auto node: nodes) {
        nodes_set.insert(node);
    }
    fillParentChildrenMapFromNodes(nodes_set, parents_map, children_map);
}

void addDisjunctiveEdgesToParentChildrenMap(NodeRelationMap& parents_map, NodeRelationMap& children_map,
                                            const std::vector<Node*>& comp_chain, const std::vector<Node*>& comm_chain,
                                            NodeSet nodes) {
    auto add_dep_for_consecutive_pairs = [&](const std::vector<Node*>& chain) {
        for(int i=1; i< chain.size(); i++) {
            auto prev_node = chain[i-1];
            auto current_node = chain[i];
            if(nodes.empty() || nodes.count(prev_node) && nodes.count(current_node)) {
                children_map[current_node].insert(prev_node);
                parents_map[prev_node].insert(current_node);
            }
        }
    };
    add_dep_for_consecutive_pairs(comp_chain);
    add_dep_for_consecutive_pairs(comm_chain);
}

void extractDFGNodes(const DependencyGraph* dfg, NodeSet& nodes) {
    for(const auto node: dfg->post_dfs_order) {
        nodes.insert(node);
    }
}

void extractDFGNodeExprRelationship(const DependencyGraph* dfg, NodeMap<Expr>& node_expr, ExprMap<const Node*>& expr_node) {
    for(auto it: dfg->expr_node) {
        expr_node[it.first] = it.second;
        node_expr[it.second] = it.first;
    }
}

void removeFunctionNodes(ExtendedDFG& dfg) {
    auto nodes = dfg.nodes();
    for (auto node : nodes) {
        auto expr = dfg.getExprFromNode(node);
        if (expr->IsInstance<FunctionNode>()) {
            dfg.deleteNode(node);
            dfg.deleteFromExprMap(node);
        }
    }
}

ExtendedDFG::ExtendedDFG(const DependencyGraph* dfg, const NodeMap<SimulateTimeType>& exec_time_map,
                const NodeMap<NodeType>& node_type_map, const NodeMap<std::string>& node_name_map, 
                const NodeMap<CommComponents>& comm_size_map) :
                dfg_(dfg), exec_time_map_(exec_time_map), node_type_map_(node_type_map),
                node_name_map_(node_name_map), comm_size_map_(comm_size_map) {
    // we create parent and children maps using dependency in the input dfg
    fillParentChildrenMapFromNodes(dfg_->post_dfs_order, parents_map_, children_map_);
    extractDFGNodes(dfg_, nodes_);
    extractDFGNodeExprRelationship(dfg_, node2expr_, expr2node_);
    removeFunctionNodes(*this);
    createSourceAndSink_();
}

ExtendedDFG::ExtendedDFG(const DependencyGraph* dfg, const NodeMap<SimulateTimeType>& exec_time_map,
            const NodeMap<NodeType>& node_type_map) :
            dfg_(dfg), exec_time_map_(exec_time_map), node_type_map_(node_type_map) {
    // we create parent and children maps using dependency in the input dfg
    fillParentChildrenMapFromNodes(dfg_->post_dfs_order, parents_map_, children_map_);
    extractDFGNodes(dfg_, nodes_);
    extractDFGNodeExprRelationship(dfg_, node2expr_, expr2node_);
    removeFunctionNodes(*this);
    createSourceAndSink_();
}

ExtendedDFG::ExtendedDFG(const DependencyGraph* dfg): dfg_(dfg) {
    fillParentChildrenMapFromNodes(dfg_->post_dfs_order, parents_map_, children_map_);
    extractDFGNodes(dfg_, nodes_);
    extractDFGNodeExprRelationship(dfg_, node2expr_, expr2node_);
    for(auto node: nodes()) {
        exec_time_map_[node] = 0;
        node_type_map_[node] = NodeType::kCompNode;
    }
    removeFunctionNodes(*this);
    createSourceAndSink_();
}

const Node* ExtendedDFG::source() const {
    return source_;
}

const Node* ExtendedDFG::sink() const {
    return sink_;
}

NodeSet ExtendedDFG::nodes() const {
    return nodes_;
}

// getter and setter for the maps
SimulateTimeType ExtendedDFG::getNodeExecTime(const Node* node) const {
    CHECK(node) << "Encountered nullptr in getNodeExecTime.";
    CHECK(exec_time_map_.count(node)) << "Cannot find the node " << getNodeNameOrDefault(node) << " in exec time map.";
    return exec_time_map_.at(node);
}

void ExtendedDFG::setNodeExecTime(const Node* node, const SimulateTimeType exec_time) {
    assertInDFG_(node);
    exec_time_map_[node] = exec_time;
}

NodeMap<SimulateTimeType> ExtendedDFG::getNodeExecTimeAsMap() const {
    return exec_time_map_;
}

NodeType ExtendedDFG::getNodeType(const Node* node) const {
    CHECK(node_type_map_.count(node)) << "Cannot find the node in node type map.";
    return node_type_map_.at(node);
}

// we disallow setting the node type

bool ExtendedDFG::hasNodeName(const Node* node) const {
    if(node_name_map_.count(node)) {
        return true;
    }
    return false;
}

std::string ExtendedDFG::getNodeNameOrDefault(const Node* node) const {
    if(node_name_map_.count(node)) {
        return node_name_map_.at(node);
    }
    return "UnknownNode";
}

void ExtendedDFG::setNodeName(const Node* node, const std::string& name) {
    assertInDFG_(node);
    node_name_map_[node] = name;
}

CommComponents ExtendedDFG::getCommSize(const Node* node) const {
    CHECK_EQ(getNodeType(node), NodeType::kCommNode) << "Encountered non-communication node.";
    CHECK(comm_size_map_.count(node)) << "Cannot find the node in comm size map.";
    return comm_size_map_.at(node);
}

void ExtendedDFG::setCommSize(const Node* node, const CommComponents& size) {
    assertInDFG_(node);
    if(getNodeType(node) == NodeType::kCompNode) {
        node_type_map_[node] = NodeType::kCommNode;
    }
    comm_size_map_[node] = size;
}

// should check whether the returned expr is defined before using
Expr ExtendedDFG::getExprFromNode(const Node* node) const {
    if(node2expr_.count(node)) {
        return node2expr_.at(node);
    }
    return Expr();
}

// same here
const Node* ExtendedDFG::getNodeFromExpr(Expr expr) const {
    // TODO(@ye-tian): Figure out why expr2node's size is 0 and 
    // the count method throws segmentation fault.
    if (expr2node_.size() && expr2node_.count(expr)) {
        return expr2node_.at(expr);
    }
    return nullptr;
}

NodeSet ExtendedDFG::getParents(const Node* node) const {
    CHECK(parents_map_.count(node)) << "Cannot find the node in parents map.";
    return parents_map_.at(node);
}

NodeSet ExtendedDFG::getNonSinkParents(const Node* node) const {
    auto parents = getParents(node);
    if(parents.count(sink_)) {
        parents.erase(sink_);
    }
    return parents;
}

NodeSet ExtendedDFG::getChildren(const Node* node) const {
    CHECK(children_map_.count(node)) << "Cannot find the node in children map.";
    return children_map_.at(node);
}

NodeSet ExtendedDFG::getNonSourceChildren(const Node* node) const {
    auto children = getChildren(node);
    if(children.count(source_)) {
        children.erase(source_);
    }
    return children;
}

void ExtendedDFG::createEdge(const Node* parent, const Node* child) {
    if(parent == child) {
        // don't create self edges
        return;
    }
    checkNodeInRelationMaps_(parent, /*node_hint=*/ "parent");
    checkNodeInRelationMaps_(child, /*node_hint=*/ "child");
    parents_map_[child].insert(parent);
    children_map_[parent].insert(child);
}

void ExtendedDFG::deleteEdge(const Node* parent, const Node* child) {
    checkNodeInRelationMaps_(parent, /*node_hint=*/ "producer");
    checkNodeInRelationMaps_(child, /*node_hint=*/ "consumer");
    auto& child_parents = parents_map_.at(child);
    if(child_parents.count(parent)) {
        child_parents.erase(parent);
    }
    auto& parent_children = children_map_.at(parent);
    if(parent_children.count(child)) {
        parent_children.erase(child);
    }
}

// create a new node without any connecting edges
// the returned ptr will no long be valid after all referencing ExtendedDFG are destroyed
const Node* ExtendedDFG::createCompNode(const std::string& node_name) {
    std::shared_ptr<Node> new_node_shared_ptr(new Node());
    created_nodes_.insert(new_node_shared_ptr);
    const Node* new_node = new_node_shared_ptr.get();
    nodes_.insert(new_node);
    exec_time_map_[new_node] = 0;
    node_type_map_[new_node] = NodeType::kCompNode;
    node_name_map_[new_node] = node_name;
    parents_map_[new_node] = {};
    children_map_[new_node] = {};
    return new_node;
}

const Node* ExtendedDFG::createCompNode(SimulateTimeType exec_time, const std::string& node_name) {
    const Node* new_node = createCompNode(node_name);
    exec_time_map_[new_node] = exec_time;
    return new_node;
}

const Node* ExtendedDFG::createCommNode(const std::string& node_name) {
    const Node* new_node = createCompNode(node_name);
    node_type_map_[new_node] = NodeType::kCommNode;
    comm_size_map_[new_node] = {};
    return new_node;
}

const Node* ExtendedDFG::createCommNode(SimulateTimeType exec_time, CommComponents size, const std::string& node_name) {
    const Node* new_node = createCompNode(exec_time, node_name);
    node_type_map_[new_node] = NodeType::kCommNode;
    comm_size_map_[new_node] = size;
    return new_node;
}

bool ExtendedDFG::hasNode(const Node* node) const {
    return nodes_.count(node);
}

void ExtendedDFG::deleteFromExprMap(const Node* node) {
    if(node2expr_.count(node)) {
        Expr expr2remove = node2expr_.at(node);
        node2expr_.erase(node);
        CHECK_AND_REMOVE(expr2node_, expr2remove);
    }
}

void ExtendedDFG::deleteFromExprMap(Expr expr) {
    if(expr2node_.count(expr)) {
        const Node* node2remove = expr2node_.at(expr);
        expr2node_.erase(expr);
        CHECK_AND_REMOVE(node2expr_, node2remove);
    }
}

void ExtendedDFG::updateNodeExprMap(const Node* node, Expr expr) {
    node2expr_[node] = expr;
    expr2node_[expr] = node;
}

void ExtendedDFG::updateExpr(Expr old_expr, Expr new_expr) {
    if(expr2node_.count(old_expr)) {
        const Node* node = expr2node_.at(old_expr);
        expr2node_.erase(old_expr);
        expr2node_[new_expr] = node;
        node2expr_[node] = new_expr;
    }
}

void ExtendedDFG::updateExprFromMap(const ExprMap<Expr>& expr_map) {
    for(auto it: expr_map) {
        updateExpr(it.first, it.second);
    }
}

// args here should be the actual expr instead of var
const Node* ExtendedDFG::insertNewCompExprAndAddInputEdges(const Expr& expr, const Array<Expr>& args, 
                                                           std::string node_name, SimulateTimeType exec_time) {
    const Node* new_node = createCompNode(exec_time, node_name);
    updateNodeExprMap(new_node, expr);
    for(auto arg: args) {
        if(const Node* arg_node = getNodeFromExpr(arg)) {
            createEdge(new_node, arg_node);
        }
    }
    return new_node;
}

const Node* ExtendedDFG::insertNewCommExprAndAddInputEdges(const Expr& expr, const Array<Expr>& args, std::string node_name,
                                                           SimulateTimeType exec_time, CommComponents size) {
    const Node* new_node = insertNewCompExprAndAddInputEdges(expr, args, node_name, exec_time);
    node_type_map_[new_node] = NodeType::kCommNode;
    comm_size_map_[new_node] = size;
    return new_node;
}

void ExtendedDFG::mergeNodes(const Node* node, const NodeSet& nodes) {
    NodeSet parents = {};
    NodeSet children = {};
    for (auto node_ : nodes) {
        for (auto parent : getParents(node_)) {
            if (!nodes.count(parent)) {
                parents.insert(parent);
            }
        }
        for (auto child : getChildren(node_)) {
            if (!nodes.count(child)) {
                children.insert(child);
            }
        }
    }
    for (auto parent : parents) {
        createEdge(parent, node);
    }
    for (auto child : children) {
        createEdge(node, child);
    }
    for (auto node_ : nodes) {
        deleteNode(node_);
        deleteFromExprMap(node_);
    }
}

const Node* ExtendedDFG::insertNewCompExprAndMergeExprs(const Expr& expr, const Exprs& exprs, 
                                                        std::string node_name, SimulateTimeType exec_time) {
    const Node* new_node = createCompNode(exec_time, node_name);
    updateNodeExprMap(new_node, expr);
    NodeSet nodes = {};
    for (auto expr_ : exprs) {
        nodes.insert(getNodeFromExpr(expr_));
    }
    mergeNodes(new_node, nodes);
    return new_node;
}

const Node* ExtendedDFG::insertNewCommExprAndMergeExprs(const Expr& expr, const Exprs& exprs, std::string node_name,
                                                        SimulateTimeType exec_time, CommComponents size) {
    LOG(FATAL) << "insertNewCommExprAndMergeExprs is not implemented.";
    return nullptr;
}

void ExtendedDFG::deleteNode(const Node* node) {
    CHECK_NE(node, source_) << "Cannot delete root node of DFG.";
    CHECK_NE(node, sink_) << "Cannot delete sink node of DFG.";
    NodeType node_type = getNodeType(node);
    CHECK_AND_REMOVE(node_type_map_, node);
    // if the node is a comm node, also delete its size
    if (node_type == NodeType::kCommNode) {
        CHECK_AND_REMOVE(comm_size_map_, node);
    }
    CHECK_AND_REMOVE(exec_time_map_, node);
    CHECK_AND_REMOVE(node_name_map_, node);
    CHECK_AND_REMOVE(nodes_, node);
    CHECK_AND_REMOVE(parents_map_, node);
    CHECK_AND_REMOVE(children_map_, node);
    // we do not delete the node from expr map here

    // also remove any edges that contains this node
    for(auto current_node: nodes_) {
        auto& parents = parents_map_.at(current_node);
        if(parents.count(node)) {
            parents.erase(node);
        }
        auto& children = children_map_.at(current_node);
        if(children.count(node)) {
            children.erase(node);
        }
    }
}

void ExtendedDFG::recalculateSourceAndSink() {
    for(auto node: nodes_) {
        if(node != source_ && node != sink_) {
            auto parents = getParents(node);
            if(parents.empty()) {
                createEdge(sink_, node);
            } else if (parents.size() >= 2) {
                if(parents.count(sink_)) {
                    deleteEdge(sink_, node);
                }
            }
            auto children = getChildren(node);
            if(children.empty()) {
                createEdge(node, source_);
            } else if (children.size() >= 2) {
                if(children.count(source_)) {
                    deleteEdge(node, source_);
                }
            }
        }
    }
}

void ExtendedDFG::printNodeInfo(std::ostream &os, const Node* node) const {
    os << "Node: " << getNodeNameOrDefault(node) << ", exec_time: " << getNodeExecTime(node) << std::endl;
    os << "\tChildren:" << std::endl;
    for(auto child: getChildren(node)) {
      os << "\t\t-> " << getNodeNameOrDefault(child) << std::endl;
    }
    os << "\tParents:" << std::endl;
    for(auto parent: getParents(node)) {
      os << "\t\t-> " << getNodeNameOrDefault(parent) << std::endl;
    }
}

void ExtendedDFG::checkNodeInRelationMaps_(const Node* node, const std::string& node_hint) const {
    CHECK(parents_map_.count(node) && children_map_.count(node))
        << "Cannot find " + node_hint + " in parent or children maps.";
}

void ExtendedDFG::assertInDFG_(const Node* node) const {
    CHECK(nodes_.count(node)) << "Node is not in the DFG.";
}

void ExtendedDFG::createSourceAndSink_() {
    const Node* source_node = createCompNode("SOURCE");
    const Node* sink_node = createCompNode("SINK");
    for(auto node: nodes_) {
        if(node != source_node && node != sink_node) {
            auto parents = getParents(node);
            if(parents.empty()) {
                createEdge(sink_node, node);
            }
            auto children = getChildren(node);
            if(children.empty()) {
                createEdge(node, source_node);
            }
        }
    }
    source_ = source_node;
    sink_ = sink_node;
}

std::ostream& operator << (std::ostream &os, const ExtendedDFG &dfg) {
  auto topo_order = GetTopologicalOrder(dfg);
  os << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
  for(auto node: topo_order) {
    dfg.printNodeInfo(os, node);
  }
  return os << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
}

std::ostream& operator << (std::ostream &os, const ScheduledDFG& sched_dfg) {
    os << sched_dfg.dfg;
    os << "Comp chain: " << std::endl;
    for(auto node: sched_dfg.comp_chain) {
        os << "\t -> " << sched_dfg.dfg.getNodeNameOrDefault(node) << std::endl;
    }
    os << "------------------------------------------------------" << std::endl;
    os << "Comm chain: " << std::endl;
    for(auto node: sched_dfg.comm_chain) {
        os << "\t -> " << sched_dfg.dfg.getNodeNameOrDefault(node) << std::endl;
    }
    return os << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
}

void FillVarExprMap(const Expr& anf_expr, VarMap<Expr>& var_expr_map, ExprMap<Var>& expr_var_map) {
    std::unique_ptr<ExplicitLetList> ell(ExplicitLetList::make(anf_expr));
    for(int i = 0; i < ell->vars.size(); i++) {
      auto var = ell->vars[i];
      auto expr = ell->exprs[i];
      var_expr_map[var] = expr;
      expr_var_map[expr] = var;
    }
}


ScheduledDFG::ScheduledDFG(const ExtendedDFG& dfg): dfg(dfg) {}
ScheduledDFG::ScheduledDFG(const ExtendedDFG& dfg, const Expr& scheduled_expr, const Nodes& total_order,
    const Nodes& comp_chain, const Nodes& comm_chain, const NodeMap<NodeDuration>& node_durations):
    dfg(dfg), scheduled_expr(scheduled_expr), total_order(total_order),
    comp_chain(comp_chain), comm_chain(comm_chain), node_durations(node_durations) {
    FillVarExprMap(scheduled_expr, var_expr_map, expr_var_map);
    total_order_valid = true;
}

// this function deletes the node from dfg and also removes it from the node chains and the var expr maps
void ScheduledDFG::deleteNode(const Node* node) {
    auto expr = dfg.getExprFromNode(node);
    if(expr.defined()) {
        CHECK(expr_var_map.count(expr)) << "Encountered expr " << expr << " that do not have a corresponding var in sched_dfg.";
        Var var = expr_var_map.at(expr);
        expr_var_map.erase(expr);
        var_expr_map.erase(var);
    }
    dfg.deleteNode(node);
    auto delete_from_nodes = [&](Nodes& nodes) {
        auto it = std::find(nodes.begin(), nodes.end(), node);
        if(it != nodes.end()) {
            nodes.erase(it);
        }
    };
    delete_from_nodes(total_order);
    delete_from_nodes(comp_chain);
    delete_from_nodes(comm_chain);
    CHECK_AND_REMOVE(node_durations, node);
}

void ScheduledDFG::setScheduledExpr(Expr expr) {
    scheduled_expr = expr;
    FillVarExprMap(scheduled_expr, var_expr_map, expr_var_map);
}

Array<Expr> ScheduledDFG::insertNewExprHelper_(Var var, Expr expr, const Array<Expr>& args) {
    var_expr_map[var] = expr;
    expr_var_map[expr] = var;
    Array<Expr> converted_args;
    for(auto arg: args) {
        if(arg->IsInstance<VarNode>() && var_expr_map.count(Downcast<Var>(arg))) {
            converted_args.push_back(var_expr_map.at(Downcast<Var>(arg)));
        } else {
            converted_args.push_back(arg);
        }
    }
    return converted_args;
}

void ScheduledDFG::insertNewCompExprAndAddInputEdges(Var var, Expr expr, const Array<Expr>& args, 
                                                     std::string expr_name, SimulateTimeType exec_time) {
    auto converted_args = insertNewExprHelper_(var, expr, args);
    dfg.insertNewCompExprAndAddInputEdges(expr, converted_args, expr_name, exec_time);
}

void ScheduledDFG::insertNewCommExprAndAddInputEdges(Var var, Expr expr, const Array<Expr>& args,
                                                    std::string expr_name, SimulateTimeType exec_time, CommComponents size) {
    auto converted_args = insertNewExprHelper_(var, expr, args);
    dfg.insertNewCommExprAndAddInputEdges(expr, converted_args, expr_name, exec_time, size);
}

void ScheduledDFG::insertNewCompExprAndMergeExprs(Var var, Expr expr, const Exprs& exprs, std::string expr_name, SimulateTimeType exec_time) {
    var_expr_map[var] = expr;
    expr_var_map[expr] = var;
    for(auto merged_expr: exprs) {
        if (expr_var_map.count(merged_expr)) {
            Var var = expr_var_map.at(merged_expr);
            expr_var_map.erase(merged_expr);
            var_expr_map.erase(var);
        }
    }
    dfg.insertNewCompExprAndMergeExprs(expr, exprs, expr_name, exec_time);
}

void ScheduledDFG::setCriticalPath(const Nodes& critical_path_) {
    critical_path = critical_path_;
    critical_path_valid = true;
}

void ScheduledDFG::setDefaultNodeDuration(const Node* node) {
    auto time = dfg.getNodeExecTime(node);
    node_durations[node].end = time;
}

int GetNumberOfConnectedComponents(const ExtendedDFG& dfg) {
    NodeMap<bool> visited;
    for(auto node: dfg.nodes()) {
        visited[node] = false;
    }
    std::function<void(const Node*)> dfs = [&](const Node* n) {
        NodeSet neighbors = dfg.getParents(n);
        NodeSet children = dfg.getChildren(n);
        neighbors.insert(children.begin(), children.end());

        for(auto neighbor: neighbors) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                dfs(neighbor);
            }
        }
    };

    int n_ccs = 0;
    for(auto node: dfg.nodes()) {
        if (!visited[node]) {
            visited[node] = true;
            n_ccs += 1;
            dfs(node);
        }
    }
    return n_ccs;
}

// Utility functions on extended dfg
Nodes GetReversedPostDFSOrder(const ExtendedDFG& dfg, const NodeSet& nodes) {
  std::unordered_map<const Node*, int> in_degree;
  Nodes reversed_post_dfs_order;
  for (auto node: nodes) {
    for (auto parent: dfg.getParents(node)) {
        if(nodes.count(parent)) {
            in_degree[parent]++;
        }
    }
  }
  Nodes stack;
  for (auto node: nodes) {
    if (in_degree[node] == 0) {
      stack.push_back(node);
    }
  }
  while (!stack.empty()) {
    const Node* node = stack.back();
    stack.pop_back();
    reversed_post_dfs_order.push_back(node);
    for (auto parent: dfg.getParents(node)) {
      if(nodes.count(parent)) {
        if (--in_degree[parent] == 0) {
            stack.push_back(parent);
        }
      }
    }
  }
  return reversed_post_dfs_order;
}

Nodes GetTopologicalOrder(const ExtendedDFG& dfg, const NodeSet& nodes, bool reversed) {
    auto rev_post_dfs_order = GetReversedPostDFSOrder(dfg, nodes);
    if(reversed) {
        std::reverse(rev_post_dfs_order.begin(), rev_post_dfs_order.end());
        return rev_post_dfs_order;
    }
    return rev_post_dfs_order;
}

Nodes GetTopologicalOrder(const ExtendedDFG& dfg, const Nodes& nodes, bool reversed) {
    NodeSet node_set(nodes.begin(), nodes.end());
    return GetTopologicalOrder(dfg, node_set, reversed);
}

Nodes GetTopologicalOrder(const ExtendedDFG& dfg, bool reversed) {
    return GetTopologicalOrder(dfg, dfg.nodes(), reversed);
}

Nodes addResourceEdgesAndGetTopologicalOrder(const ScheduledDFG& G, bool reversed) {
    // make a local copy
    ExtendedDFG tmp_dfg = G.dfg;
    for (size_t i=1; i< G.comp_chain.size(); i++) {
        tmp_dfg.createEdge(G.comp_chain[i], G.comp_chain[i-1]);
    }
    for (size_t i=1; i< G.comm_chain.size(); i++) {
        tmp_dfg.createEdge(G.comm_chain[i], G.comm_chain[i-1]);
    }
    auto topo_order = GetTopologicalOrder(tmp_dfg, /*reversed=*/reversed);
    if(topo_order.size() != tmp_dfg.nodes().size()) {
        LOG(FATAL) << "Resource edges created cycles in the dfg, topo_order.size(): " << topo_order.size() << ", dfg.nodes().size(): " << tmp_dfg.nodes().size();
    }
    return topo_order;
}

void PruneRedundantEdges(ExtendedDFG& dfg) {
  NodeMap<NodeSet> indirect_children_map;
  std::vector<std::pair<const Node*, const Node*>> edges2remove;

  auto topo_order = GetTopologicalOrder(dfg);

  for (const Node* node : topo_order) {
    auto& indirect_children = indirect_children_map[node];
    for (auto child: dfg.getChildren(node)) {
      if (indirect_children.count(child)) {
        // There is a path from child to node that does not go through edge (child, node) directly.
        edges2remove.emplace_back(child, node);
      }
    }
    for (auto child: dfg.getChildren(node)) {
      indirect_children.insert(child);
    }
    for (auto parent: dfg.getParents(node)) {
      auto& parent_indirect_children = indirect_children_map[parent];
      for (const Node* indirect_child : indirect_children) {
        parent_indirect_children.insert(indirect_child);
      }
    }
  }

  for (auto& pr : edges2remove) {
    const Node *parent, *child;
    std::tie(child, parent) = pr;
    dfg.deleteEdge(parent, child);
  }
}

} // namespace extended_dfg
} // namespace pass
} // namespace raf
