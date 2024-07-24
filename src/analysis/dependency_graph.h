/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/analysis/dependency_graph.h
 * \brief Utilities to manipulate and analyze dependency graph
 */
#pragma once

namespace raf {
namespace analysis {
namespace dependency_graph {

using Arena = tvm::support::Arena;
using Node = DependencyGraph::Node;
using NodeList = tvm::relay::LinkedList<Node*>;
using NodeExprMap = std::unordered_map<const Node*, Expr>;

/*!
 * \brief Remove the edge between parent and child. If there are duplicated edges between them,
 * the duplicated edges will also be removed.
 * \param parent The parent node of the edge.
 * \param child The child node of the edge.
 */
void RemoveGraphEdge(Node* parent, Node* child);

/*!
 * \brief Remove a set of nodes fron the dfg. All incident edges of the removed nodes are 
 * also removed.
 * \param dg The dependency graph.
 * \param node The set of nodes to remove.
 */
void RemoveGraphNode(DependencyGraph* dg, std::unordered_set<DependencyGraph::Node*> node);

/*!
 * \brief Prune the redundant edges in the dependency graph. One edge (u, v) is redundant if and
 * only if there exists a path from u to v that does not go through the edge (u, v) directly. We
 * call the edge "redundant" because the dependency relation has been indicated by the path.
 *
 * The time complexity is O(NE), where N is the number of nodes and E is the number of edges in
 * the dependency graph, which may be slow for large complete graph. But it is efficient enough for
 * almost all neural networks.
 *
 * \param dg The dependency graph.
 */
void DependencyGraphPruneRedundantEdges(DependencyGraph* dg);

/*!
 * Add an edge between parent and child
 * \param arena The arena to allocate memory.
 * \param parent The parent node of the edge.
 * \param child The child node of the edge.
 */
void AddGraphEdge(Arena* arena, Node* parent, Node* child);

/*!
 * Add an edge between parent and child
 * \param arena The arena to allocate memory.
 * \param parent The parent node of the edge.
 * \param child The child node of the edge.
 */
bool AddGraphEdge(DependencyGraph::Node* parent, DependencyGraph::Node* child, Arena* arena);

/*!
 * Get a topological order of the graph derived from given nodes.
 * \param nodes The nodes to get the topological order from.
 * \return A topological order of nodes.
 */
std::vector<Node*> GetTopologicalOrder(const std::vector<Node*>& nodes);

void GetTopologicalOrder(DependencyGraph* dfg, std::vector<Node*>& nodes);

void GetTopologicalOrder(DependencyGraph* dfg, std::vector<const Node*>& nodes);

/*!
 * \brief Get a topological order on the graph that reverses all the edges in the dependency graph.
 * Thus, the first node in the order is the node corresponding to the expression used to create the
 * dependency graph.
 * \param dg The dependency graph.
 * \return The post dfs order on the reversed graph.
 */
std::vector<DependencyGraph::Node*> GetReversedTopologicalOrder(DependencyGraph* dg);

/*!
 * \brief Get the number of nodes in the linked list.
 * \return The number of nodes.
 */
size_t GetListSize(const tvm::relay::LinkedList<Node*>& node_list);

/*!
 * Create a new node.
 * \param arena The arena to allocate memory.
 * \return The created node.
 */
Node* CreateNewNode(Arena* arena);

}  // namespace dependency_graph
}  // namespace analysis
}  // namespace raf
