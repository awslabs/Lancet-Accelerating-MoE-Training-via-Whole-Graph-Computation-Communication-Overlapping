/*!
 * Copyright (c) 2022 by Contributors
 * \file extended_dfg.h
 * \brief Extension to DFG.
 */
#pragma once
#include <algorithm>
#include <iostream>
#include "raf/ir.h"
#include "raf/analysis.h"
#include "../../analysis/dependency_graph.h"
#include "./scheduler_common.h"

#define CHECK_AND_REMOVE(set_or_map, key)        \
  do {                                           \
    if(set_or_map.count(key)) {                  \
        set_or_map.erase(key);                   \
    }                                            \
  } while (0);

#define CREATE_SUBMAP_NODESETS(src_map, dst_map, nodes)     \
  do {                                                      \
    for(auto it: src_map) {                                 \
        if(nodes.count(it.first)) {                         \
            dst_map[it.first] = {};                         \
            for(const auto node_in_set: it.second) {        \
                if(nodes.count(node_in_set)) {              \
                    dst_map[it.first].insert(node_in_set);  \
                }                                           \
            }                                               \
        }                                                   \
    }                                                       \
  } while (0);

#define CREATE_SUBMAP_ATTRS(src_map, dst_map, nodes)        \
  do {                                                      \
    for(auto it: src_map) {                                 \
        if(nodes.count(it.first)) {                         \
            dst_map[it.first] = it.second;                  \
        }                                                   \
    }                                                       \
  } while (0);

namespace raf {
namespace pass {
namespace extended_dfg {

using namespace raf::pass::scheduler_common;

using NodeRelationMap = std::unordered_map<const Node*, NodeSet>;

// Mutates parents_map and children_map
void fillParentChildrenMapFromNodes(NodeSet nodes, NodeRelationMap& parents_map, NodeRelationMap& children_map);

// Mutates parents_map and children_map
void fillParentChildrenMapFromNodes(const std::vector<Node*> nodes, NodeRelationMap& parents_map, NodeRelationMap& children_map);

// Mutates parents_map and children_map
void addDisjunctiveEdgesToParentChildrenMap(NodeRelationMap& parents_map, NodeRelationMap& children_map,
                                            const std::vector<Node*>& comp_chain, const std::vector<Node*>& comm_chain,
                                            NodeSet nodes = {});

void extractDFGNodes(const DependencyGraph* dfg, NodeSet& nodes);

void extractDFGNodeExprRelationship(const DependencyGraph* dfg, NodeMap<Expr>& node_expr, ExprMap<const Node*>& expr_node);

class ExtendedDFG {
    /* 
        This is a utility class which serves as an extension to the Relay Dependency Graph
        It supports associating attributes to a graph node.

        The original DependencyGraph and Expr is constant in ExtendedDFG. To actually apply
        modification to the Expr, use the two functions: Fuse and Partition. The original 
        DependencyGraph is not modified though (since we do not have the associated arena, and 
        we won't directly access it anyway).
    */
public:
    // constructors
    ExtendedDFG(const DependencyGraph* dfg, const NodeMap<SimulateTimeType>& exec_time_map,
                const NodeMap<NodeType>& node_type_map, const NodeMap<std::string>& node_name_map, 
                const NodeMap<CommComponents>& comm_size_map);

    // constructor without node_name_map and comm_size_map. This should only be used in static schedule without optimization
    ExtendedDFG(const DependencyGraph* dfg, const NodeMap<SimulateTimeType>& exec_time_map,
                const NodeMap<NodeType>& node_type_map);
    
    // constructor without any attribute maps. This simply wraps the original dfg.
    ExtendedDFG(const DependencyGraph* dfg);

    // ExtendedDFG can be copied using default copy and move constructors
    // Same for copy assignment operator

    const Node* source() const;
    const Node* sink() const;
    NodeSet nodes() const;

    // getter and setter for the maps
    SimulateTimeType getNodeExecTime(const Node* node) const;
    void setNodeExecTime(const Node* node, const SimulateTimeType exec_time);

    NodeMap<SimulateTimeType> getNodeExecTimeAsMap() const;
    NodeType getNodeType(const Node* node) const;

    // we disallow setting the node type

    bool hasNodeName(const Node* node) const;
    std::string getNodeNameOrDefault(const Node* node) const;
    void setNodeName(const Node* node, const std::string& name);

    CommComponents getCommSize(const Node* node) const;
    void setCommSize(const Node* node, const CommComponents& size);

    // should check whether the returned expr is defined before using
    Expr getExprFromNode(const Node* node) const;
    // same here
    const Node* getNodeFromExpr(Expr expr) const;

    NodeSet getParents(const Node* node) const;
    NodeSet getNonSinkParents(const Node* node) const;
    NodeSet getChildren(const Node* node) const;
    NodeSet getNonSourceChildren(const Node* node) const;

    void createEdge(const Node* parent, const Node* child);
    void deleteEdge(const Node* parent, const Node* child);

    // create a new node without any connecting edges
    // the returned ptr will no long be valid after all referencing ExtendedDFG are destroyed
    const Node* createCompNode(const std::string& node_name);
    const Node* createCompNode(SimulateTimeType exec_time, const std::string& node_name);
    const Node* createCommNode(const std::string& node_name);
    const Node* createCommNode(SimulateTimeType exec_time, CommComponents size, const std::string& node_name);

    bool hasNode(const Node* node) const;
    void deleteFromExprMap(const Node* node);
    void deleteFromExprMap(Expr expr);
    void updateNodeExprMap(const Node* node, Expr expr);
    void updateExpr(Expr old_expr, Expr new_expr);
    void updateExprFromMap(const ExprMap<Expr>& expr_map);
    const Node* insertNewCompExprAndAddInputEdges(const Expr& expr, const Array<Expr>& args, 
                                                  std::string node_name, SimulateTimeType exec_time);
    const Node* insertNewCommExprAndAddInputEdges(const Expr& expr, const Array<Expr>& args, std::string node_name,
                                                  SimulateTimeType exec_time, CommComponents size);
    const Node* insertNewCompExprAndMergeExprs(const Expr& expr, const Exprs& exprs, std::string node_name, SimulateTimeType exec_time);
    const Node* insertNewCommExprAndMergeExprs(const Expr& expr, const Exprs& exprs, std::string node_name, SimulateTimeType exec_time, CommComponents size);
    void mergeNodes(const Node* node, const NodeSet& nodes);
    void deleteNode(const Node* node);

    void recalculateSourceAndSink();

    void printNodeInfo(std::ostream &os, const Node* node) const;

protected:
    void checkNodeInRelationMaps_(const Node* node, const std::string& node_hint = "node") const;
    void assertInDFG_(const Node* node) const;
    void createSourceAndSink_();

    // The const reference to the dfg constructed from the original IR. This DFG will not be modified
    // throughout the optimization process.
    const DependencyGraph* dfg_;

    // node maps associated with the Extended DFG, which will be modified during optimization
    NodeMap<SimulateTimeType> exec_time_map_;
    NodeMap<NodeType> node_type_map_;
    NodeMap<std::string> node_name_map_;
    NodeMap<CommComponents> comm_size_map_;

    // the definition of parents and children follows DependencyGraph
    NodeRelationMap parents_map_;
    NodeRelationMap children_map_;

    // the expr node maps
    NodeMap<Expr> node2expr_;
    ExprMap<const Node*> expr2node_;

    // the set of all nodes
    NodeSet nodes_;
    // the set of nodes that is created by us outside of arena
    // Note: the raw ptrs will be invalid after all referencing ExtendedDFG are destroyed
    std::unordered_set<std::shared_ptr<Node>> created_nodes_;

    const Node* source_;
    const Node* sink_;
};

std::ostream& operator << (std::ostream &os, const ExtendedDFG &dfg);

void removeFunctionNodes(ExtendedDFG& dfg);

struct ScheduledDFG {
    ExtendedDFG dfg;
    Expr scheduled_expr;
    Nodes critical_path;
    Nodes total_order;
    Nodes comp_chain;
    Nodes comm_chain;
    NodeMap<NodeDuration> node_durations;
    VarMap<Expr> var_expr_map;
    ExprMap<Var> expr_var_map;

    bool dfg_has_resource_edge = false;
    bool critical_path_valid = false;
    bool total_order_valid = false;

    explicit ScheduledDFG(const ExtendedDFG& dfg);
    ScheduledDFG(const ExtendedDFG& dfg, const Expr& scheduled_expr, const Nodes& total_order,
                 const Nodes& comp_chain, const Nodes& comm_chain, const NodeMap<NodeDuration>& node_durations);
    // this function deletes the node from dfg and also removes it from the node chains
    void deleteNode(const Node* node);
    void setScheduledExpr(Expr expr);
    // here args can be vars in the ANF
    void insertNewCompExprAndAddInputEdges(Var var, Expr expr, const Array<Expr>& args, 
                                           std::string expr_name, SimulateTimeType exec_time);
    void insertNewCommExprAndAddInputEdges(Var var, Expr expr, const Array<Expr>& args,
                                           std::string expr_name, SimulateTimeType exec_time, CommComponents size);
    Array<Expr> insertNewExprHelper_(Var var, Expr expr, const Array<Expr>& args);
    void insertNewCompExprAndMergeExprs(Var var, Expr expr, const Exprs& exprs, std::string expr_name, SimulateTimeType exec_time);
    void setCriticalPath(const Nodes& critical_path_);
    void setDefaultNodeDuration(const Node* node);
};

std::ostream& operator << (std::ostream &os, const ScheduledDFG& sched_dfg);

int GetNumberOfConnectedComponents(const ExtendedDFG& dfg);

// Utility functions on extended dfg
Nodes GetReversedPostDFSOrder(const ExtendedDFG& dfg, const NodeSet& nodes);

Nodes GetTopologicalOrder(const ExtendedDFG& dfg, const NodeSet& nodes, bool reversed = false);

Nodes GetTopologicalOrder(const ExtendedDFG& dfg, const Nodes& nodes, bool reversed = false);

Nodes GetTopologicalOrder(const ExtendedDFG& dfg, bool reversed = false);

Nodes addResourceEdgesAndGetTopologicalOrder(const ScheduledDFG& G, bool reversed = false);

void PruneRedundantEdges(ExtendedDFG& dfg);

} // namespace extended_dfg
} // namespace pass
} // namespace raf