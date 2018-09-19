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
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <mxnet/op_attr_types.h>
#include <unordered_set>
#include <stack>
#include <queue>
#include <string>

#include "./default_subgraph_op.h"
#include "../nn/convolution-inl.h"
#include "../nn/pooling-inl.h"
#include "./common.h"

namespace mxnet {
namespace op {
namespace sg {

class SgRemoveSparseSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    sStart,
    sSuccess
  };

 private:
  int multiple = 2; //can be changed for more general solution
  SelectStatus status;
  std::vector<nnvm::Node*> matched_list; //include matched convs
  std::vector<nnvm::Node*> matched_conv_ptr; //contains original convs
  std::vector<nnvm::NodeEntry*> need_pooling_list;

 public:
  SgRemoveSparseSelector(){
  }
  virtual bool Select(const nnvm::Node& n) { return false; }

  virtual bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node){
    return false;
  }

  bool Select(const SimpleNode &simple_node) {
    nnvm::Node n = *(simple_node.node);
    //expand match condition for more general solution (when kernel = (1,1) and stride = (any number))
    bool match = ((!n.is_variable()) && (n.op()->name == "Convolution") &&
                  (n.attrs.dict["stride"] == "(2, 2)") &&
                  (n.attrs.dict["kernel"] == "(1, 1)"));
    if (match) {
      status = sStart;
      matched_conv_ptr.clear();
      matched_conv_ptr.push_back(simple_node.node);
      return true;
    }
    return false;
  }

  bool BackwardCheck(const sg::SimpleNode &new_node) {
    //return true if the output nodes are not all conv with kernal (1,1) and stride (2,2).
    for (auto it = new_node.outputs.begin(); it != new_node.outputs.end(); ++it) {
      nnvm::Node n = *(it->first);
      bool hit = ((!n.is_variable()) && (n.op()->name == "Convolution") &&
                  (n.attrs.dict["stride"] == "(2, 2)") &&
                  (n.attrs.dict["kernel"] == "(1, 1)"));
      if (!hit){
        return true;
      }
    }

    //if all outputs are conv with kernal (1,1) and stride (2,2), save them
    for (auto it = new_node.outputs.begin(); it != new_node.outputs.end(); ++it) {
      matched_conv_ptr.push_back(it->first);
    }
    return false;
  }

bool SelectInput( nnvm::Node &n, sg::SimpleNode &new_node) {
    //check number of outputs of this new node, if it has more than one output
    //nodes, continue if the outputs nodes are all conv with kernal (1,1) stride (2,2).
    //This covers the case for resnetv1-50. If we want a more general solution,
    //please expand this backward check.
    if ((new_node.outputs.size()>1) && BackwardCheck(new_node)){
      auto& inputs = n.inputs;
      for (size_t i = 0; i < inputs.size(); i++){
        auto& e = inputs[i];
        if (n.inputs[i].node.get() == new_node.node){
          need_pooling_list.push_back(&e);
        }
      }
      return false;
    }

    //if it's empty op, discard
    if (new_node.node->op() == nullptr){
      return false;
    }

    //check op name of the new input.
    std::string op_name = new_node.node->op()->name;
    //if encounter conv, stop searching, return false
    if ((op_name == "Convolution") && (!(new_node.node->attrs.dict["kernel"] == "(1, 1)"))){
      matched_list.push_back(new_node.node);
      status = sSuccess;
      return false;
    }
    //if we encounter other nodes (or special conv) continue, can be expanded for more general solution
    if ((op_name == "BatchNorm") || (op_name == "Activation") ||
       (op_name == "Pooling")||(op_name == "Pooling") || (op_name =="elemwise_add") ||
       ((op_name == "Convolution") && (new_node.node->attrs.dict["stride"] == "(1, 1)") &&
        (new_node.node->attrs.dict["kernel"] == "(1, 1)"))){
      return true;
    }else{
      return false;
    }
  }

  bool SelectOutput(const nnvm::Node &n,
                            const nnvm::Node &new_node) override {
    return false;
  }

  virtual std::vector<nnvm::Node *> Filter(
      nnvm::Graph *g, const std::vector<nnvm::Node *> &candidates,
      std::vector<nnvm::NodeEntry*>& entries,
      const std::vector<SimpleNodePtr>& simple_nodes) {

    //get indexed graph
    const auto& indexed_graph = g->indexed_graph();

    if (status == sStart) {
      return std::vector<nnvm::Node*>(0);
    }else{
      LOG(INFO) << "remove_sparse happen";
      // change and mark original convs
      for (auto it = matched_conv_ptr.begin(); it !=matched_conv_ptr.end(); ++it) {
        //find simple node
        const auto nid = indexed_graph.node_id((*it));
        SimpleNode* simple_node_ptr = simple_nodes[nid].get();

        simple_node_ptr->label = 1;
        (*it)->attrs.dict["stride"] = "(1, 1)";
        (*it)->op()->attr_parser(&((*it)->attrs));
      }

      //change matched convs
      for (auto sn = matched_list.begin(); sn !=matched_list.end(); ++sn) {
        (*sn)->attrs.dict["stride"] = "(2, 2)";
        (*sn)->op()->attr_parser(&((*sn)->attrs));
      }

      //push back node_attributes that need to be changed
      for (auto ne = need_pooling_list.begin(); ne !=need_pooling_list.end(); ++ne) {
        entries.push_back((*ne));
      }

      //return original conv to unmark other nodes
      std::vector<nnvm::Node*> filtered;
      filtered.push_back(candidates[0]);
      return filtered;
    }
  }
}; //end of class SgRemoveSparseSelector

class SgRemoveSparseProperty : public SubgraphProperty {
 public:
  SgRemoveSparseProperty() {}

  virtual SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgRemoveSparseSelector>();
    return selector;
  }

  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    return nullptr;
  }

  virtual void ConnectSubgraphOutput(
      const nnvm::NodePtr n,
      std::vector<nnvm::NodeEntry *> &output_entries) const override {
  }

 private:

};

} // namespace sg
} // namespace op
} // namespace mxnet

namespace mxnet {

namespace op {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::Graph;

namespace sg {
/*!
 * \brief Given a MXNet computational graph, create an undirected graph from it.
 * \param g the MXNet computational graph
 * \param simple_nodes the nodes of undirected graph in top sorted order
 */
void CreateSimpleGraphRS(const Graph& g,
                       std::vector<SimpleNodePtr>* simple_nodes) {
  const auto& indexed_graph = g.indexed_graph();
  simple_nodes->reserve(indexed_graph.num_nodes());
  DFSVisit(g.outputs, [&](const NodePtr& node) {
    SimpleNodePtr sn = SimpleNode::Create();
    sn->node = node.get();
    for (size_t i = 0; i < sn->node->inputs.size(); ++i) {
      const auto& e = sn->node->inputs[i];
      const auto input_nid = indexed_graph.node_id(e.node.get());
      CHECK_LT(input_nid, simple_nodes->size());
      auto& input_node_outputs = (*simple_nodes)[input_nid]->outputs;
      auto it = input_node_outputs.find(sn->node);
      if (it == input_node_outputs.end()) {
        input_node_outputs.emplace(sn->node, std::vector<size_t>{i});
      } else {
        it->second.push_back(i);
      }
    }
    simple_nodes->emplace_back(std::move(sn));
  });
}

/*!
 * \brief Reset labels of the subgraph nodes to the original state
 * and clear the vector of subgraph nodes.
 */
void ResetNodeLabelsRS(const nnvm::Graph& g,
                     const std::vector<SimpleNodePtr>& simple_nodes,
                     std::vector<nnvm::Node*>* subgraph_nodes) {
  for (auto n : *subgraph_nodes) {
    const auto nid = g.indexed_graph().node_id(n);
    simple_nodes[nid]->label = -1;
  }
  subgraph_nodes->clear();
}

/*!
 * \brief This function traverses the nodes in a computation graph from a starting
 * node following the input edges and output edges, and marks all nodes that
 * can be accessed from the starting node. Before the function returns,
 * it will conduct checking whether there is a loop between the potential subgraph
 * and the outside nodes. If so, add the node that should break the loop
 * in excluded_nodes and return false. Otherwise, return true.
 * \param g the whole graph
 * \subgraph_selector determines whether the visited node should be choosen or not
 * \label the label of the current subgraph
 * \snid node id of the seed simple node
 * \simple_nodes all simple nodes in the top sorted order
 * \subgraph_nodes all the nodes belonging to the same subgraph of seed node
 * \excluded_nodes set of nodes that should be excluded from the current subgraph
 */
bool LabelSubgraphRS(const Graph& g,
                   SubgraphSelectorPtr subgraph_selector,
                   const int label,
                   const size_t snid,  // simple node id, this is a seed
                   const std::vector<SimpleNodePtr>& simple_nodes,
                   std::vector<nnvm::Node*>* subgraph_nodes,
                   std::unordered_set<const nnvm::Node*>* excluded_nodes = nullptr) {
  const auto& indexed_graph = g.indexed_graph();
  std::queue<SimpleNode*> node_queue;
  if (!excluded_nodes || !excluded_nodes->count(simple_nodes[snid]->node)) {
    CHECK_EQ(simple_nodes[snid]->label, -1);
    simple_nodes[snid]->label = label;
    node_queue.push(simple_nodes[snid].get());
  }
  // key: nodes that serve as input/output nodes to the subgraph
  // value: pair of vectors of nodes in the subgraph. The first vector contains the
  // output nodes of the key in the subgraph, and the second vector contains the
  // input nodes of the key in the subgraph.
  // If a non-subgraph node has inputs from the subgraph and the other non-subgraph node
  // has outputs to the subgraph, and the first non-subgraph node is an ancestor
  // of the second non-subgraph node, there exits a cycle.
  // When breaking the cycle, we want to start from removing the node with the largest node id
  // in the subgraph.
  std::unordered_map<const nnvm::Node*,
    std::pair<std::vector<const nnvm::Node*>,
              std::vector<const nnvm::Node*>>> non_subgraph_node_map;
  while (!node_queue.empty()) {
    SimpleNode* cur_node = node_queue.front();
    node_queue.pop();
    subgraph_nodes->push_back(cur_node->node);
    // get qualified adjacent input nodes
    for (auto& e : cur_node->node->inputs) {
      const auto nid = indexed_graph.node_id(e.node.get());
      SimpleNode* simple_e_ptr = simple_nodes[nid].get();
      const bool select_input =
          (!excluded_nodes || !excluded_nodes->count(e.node.get())) &&
          (dynamic_cast<SgRemoveSparseSelector*>(subgraph_selector.get()))->SelectInput(*(cur_node->node), *simple_e_ptr);
      if (select_input) {
        // e.node is a subgraph node
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1) {
          simple_nodes[nid]->label = label;
          node_queue.push(simple_nodes[nid].get());
        }
      } else {
        // e.node is an input node of the subgraph
        non_subgraph_node_map[e.node.get()].first.push_back(cur_node->node);
      }
    }
    // get qualified output nodes
    for (auto it = cur_node->outputs.begin(); it != cur_node->outputs.end(); ++it) {
      const bool select_output = (!excluded_nodes || !excluded_nodes->count(it->first))
          && subgraph_selector->SelectOutput(*cur_node->node, *it->first);
      if (select_output) {
        // it->first is a subgraph node
        const auto nid = indexed_graph.node_id(it->first);
        CHECK_LT(nid, simple_nodes.size());
        // this node has not been visited yet
        if (simple_nodes[nid]->label == -1) {
          simple_nodes[nid]->label = label;
          node_queue.push(simple_nodes[nid].get());
        }
      } else {
        // it->first is an output node of the subgraph
        non_subgraph_node_map[it->first].second.push_back(cur_node->node);
      }
    }
  }
  // prepare to check if there is a cycle
  auto node_cmp = [&] (const nnvm::Node* node1, const nnvm::Node* node2) {
    return indexed_graph.node_id(node1) < indexed_graph.node_id(node2);
  };
  std::vector<const nnvm::Node*> non_subgraph_nodes;
  non_subgraph_nodes.reserve(non_subgraph_node_map.size());
  for (auto& kv : non_subgraph_node_map) {
    auto& output_nodes = kv.second.first;
    std::sort(output_nodes.begin(), output_nodes.end(), node_cmp);
    auto& input_nodes = kv.second.second;
    std::sort(input_nodes.begin(), input_nodes.end(), node_cmp);
    non_subgraph_nodes.push_back(kv.first);
  }
  // check whether there is a cycle between the subgraph and its input/output nodes
  auto is_ancestor = [&](const nnvm::Node* ancestor, const nnvm::Node* descendant,
                         const std::vector<nnvm::Node*>& snodes) {
    if (ancestor == descendant) return true;
    std::stack<const nnvm::Node*> s;
    s.push(descendant);
    size_t count = 0;
    while (!s.empty()) {
      CHECK_LT(count, indexed_graph.num_nodes()) << "Finding ancestor failed. There is probably"
                                                    " a loop in the graph";
      ++count;
      const nnvm::Node* top = s.top();
      s.pop();
      if (top == ancestor) {
        return true;
      }
      for (const auto& entry : top->inputs) {
        // when searching for the ancestor, the path cannot cross any subgraph node
        auto it = std::find(snodes.begin(), snodes.end(), entry.node.get());
        if (it == snodes.end()) {
          s.push(entry.node.get());
        }
      }
    }
    return false;
  };
  std::sort(non_subgraph_nodes.begin(), non_subgraph_nodes.end(), node_cmp);
  int excluded_node_id = -1;
  for (size_t i = 0; i < non_subgraph_nodes.size(); ++i) {
    auto it1 = non_subgraph_node_map.find(non_subgraph_nodes[i]);
    CHECK(it1 != non_subgraph_node_map.end());
    auto& output_nodes = it1->second.first;  // has been top sorted
    auto& input_nodes = it1->second.second;  // has been top sorted
    if (!output_nodes.empty() && !input_nodes.empty()) {
      // there is a loop between node i and the subgraph
      const auto node_id = std::max(indexed_graph.node_id(output_nodes.back()),
                                    indexed_graph.node_id(input_nodes.back()));
      excluded_node_id = std::max(excluded_node_id, static_cast<int>(node_id));
    } else if (!input_nodes.empty()) {
      // node i is an input to the subgraph, find out if there is a node j
      // which is an output of the subgraph and also a child of node i.
      for (size_t j = i + 1; j < non_subgraph_nodes.size(); ++j) {
        auto it2 = non_subgraph_node_map.find(non_subgraph_nodes[j]);
        CHECK(it2 != non_subgraph_node_map.end());
        // i is topologically before j, j might be a direct/indirect output node of i
        CHECK_LT(indexed_graph.node_id(it1->first), indexed_graph.node_id(it2->first));
        if (!it2->second.first.empty() && is_ancestor(it1->first, it2->first, *subgraph_nodes)) {
          // found a loop
          const auto node_id = std::max(indexed_graph.node_id(input_nodes.back()),
                                        indexed_graph.node_id(it2->second.first.back()));
          excluded_node_id = std::max(excluded_node_id, static_cast<int>(node_id));
        }
      }
    }
  }

  if (excluded_node_id != -1) {
    CHECK_LT(excluded_node_id, static_cast<int>(simple_nodes.size()));
    CHECK_NE(excluded_node_id, static_cast<int>(snid))
      << "A cycle is found in the computational graph between nodes "
      << simple_nodes[excluded_node_id]->node->attrs.name << " and "
      << simple_nodes[snid]->node->attrs.name;
    excluded_nodes->insert(simple_nodes[excluded_node_id]->node);
    ResetNodeLabelsRS(g, simple_nodes, subgraph_nodes);
    return false;
  }
  std::sort(subgraph_nodes->begin(), subgraph_nodes->end(), node_cmp);
  return true;
}

/*!
 * \brief Finds all the nodes that need to be processed given a seed node.
 * \param g the whole graph
 * \subgraph_selector determines whether the visited node should be choosen or not
 * \snid node id of the seed simple node
 * \simple_nodes all simple nodes in the top sorted order
 * \subgraph_nodes all the nodes belonging to the same subgraph of seed node
 * \return Subgraph node candidates sorted in the topological order
 */
void PreSelectSubgraphNodesRS(const Graph& g,
                            SubgraphSelectorPtr subgraph_selector,
                            const size_t snid,
                            const std::vector<SimpleNodePtr>& simple_nodes,
                            std::vector<nnvm::Node*>* subgraph_nodes) {
  int label = 1;
  std::unordered_set<const nnvm::Node*> excluded_nodes;
  const size_t max_num_retry = simple_nodes.size() * simple_nodes.size();
  size_t count = 0;
  bool success = false;
  while (!success && count < max_num_retry) {
    success = LabelSubgraphRS(g, subgraph_selector, label, snid, simple_nodes,
                            subgraph_nodes, &excluded_nodes);
    if (!success) {
      CHECK(!excluded_nodes.empty());
      std::string excluded_node_names;
      for (auto node : excluded_nodes) {
        excluded_node_names += node->attrs.name + ", ";
      }
      LOG(INFO) << "Found a cycle when BFS from node " << simple_nodes[snid]->node->attrs.name
                << ". Excluding nodes " << excluded_node_names << "and retrying";
    }
    ++count;
  }
  if (!success) {
    LOG(INFO) << "Tried " << count << " times of finding subgraphs starting from node "
              << simple_nodes[snid]->node->attrs.name << " without success because a loop "
                  "is always found between the subgraph and some other nodes. Will treat "
                  "seed node " << simple_nodes[snid]->node->attrs.name
              << "as a subgraph with one node";
    CHECK(subgraph_nodes->empty());
    simple_nodes[snid]->label = label;
    subgraph_nodes->push_back(simple_nodes[snid]->node);
  }
}

NodePtr CreateNodeRS(std::string op_name, std::string node_name) {
  NodePtr node = Node::Create();
  node->attrs.name = node_name;
  if (op_name == "nullptr") {
    node->attrs.op = nullptr;
    // ugly workaround because VariableParam is not exposed
    node->attrs.parsed =
      nnvm::Symbol::CreateVariable(node->attrs.name).outputs[0].node->attrs.parsed;
  } else {
    node->attrs.op = Op::Get(op_name);
  }
  return node;
}

/*!
 * \brief Find all nodes that meet certain criteria.
 * All nodes processed are marked with the same label.
 * simple_nodes here functions as an undirected copy of the original graph and is not
 * updated after graph changes
 */
void ModifyGraphRS(Graph* g,
                   const SubgraphProperty &subg_prop,
                   const std::vector<SimpleNodePtr>& simple_nodes) {
  const auto& indexed_graph = g->indexed_graph();
  CHECK_EQ(indexed_graph.num_nodes(), simple_nodes.size());
  auto node_cmp = [&] (const nnvm::Node* node1, const nnvm::Node* node2) {
    return indexed_graph.node_id(node1) < indexed_graph.node_id(node2);
  };

  //vector to save vector attributes to be changed
  std::vector<nnvm::NodeEntry*> entries;
  for (size_t i = 0; i < simple_nodes.size(); ++i) {
    SimpleNode* simple_node_ptr = simple_nodes[i].get();
    auto subgraph_selector = subg_prop.CreateSubgraphSelector();
    if ((dynamic_cast<SgRemoveSparseSelector*>(subgraph_selector.get()))
            ->Select(*simple_node_ptr) &&
        simple_nodes[i]->label == -1) {
      // pre-select nodes that are in the effeted path
      std::vector<nnvm::Node*> preselected_nodes;
      PreSelectSubgraphNodesRS(*g, subgraph_selector, i, simple_nodes, &preselected_nodes);
      // modify nodes, filter and mark convs, manually mark other
      // influenced original convs. Save nodes that need to be changed
      std::vector<nnvm::Node*> filtered_nodes = (dynamic_cast<SgRemoveSparseSelector*>(subgraph_selector.get()))->Filter(g, preselected_nodes, entries,simple_nodes);
      // make sure filtered_nodes is a subset of preselected_nodes
      for (const auto n : filtered_nodes) {
        const auto nit = std::find(preselected_nodes.begin(), preselected_nodes.end(), n);
        CHECK(nit != preselected_nodes.end())
          << "Node " << n->attrs.name << " is not found in the pre-selected subgraph nodes."
             " Please make sure that no new nodes were added in your subgraph"
             " selector's Filter function";
      }

      // make sure nodes are sorted
      std::sort(filtered_nodes.begin(), filtered_nodes.end(), node_cmp);

      // reset node labels that are not in filtered nodes
      for (const auto n : preselected_nodes) {
        const auto nit = std::find(filtered_nodes.begin(), filtered_nodes.end(), n);
        if (nit == filtered_nodes.end()) {
          simple_nodes[indexed_graph.node_id(n)]->label = -1;
        }
      }

    } //end of if
  }//end of for loop

  //change graphs
  for (const auto node_entry_ptr : entries) {
    NodePtr pooling_node = CreateNodeRS("Pooling", node_entry_ptr->node->attrs.name + "_RS_pooling");
    pooling_node->inputs.emplace_back(*(node_entry_ptr));
    pooling_node->attrs.dict["stride"] = "(2, 2)";
    pooling_node->attrs.dict["kernel"] = "(1, 1)";
    pooling_node->attrs.dict["pad"] = "(0, 0)";
    pooling_node->attrs.dict["pool_type"] = "max";
    pooling_node->op()->attr_parser(&(pooling_node->attrs));
    // modify original input node to new pooling node
    *(node_entry_ptr) = NodeEntry{pooling_node, 0, 0};
  }
}
}  // namespace sg

Graph PartitionGraphRemoveSparse(Graph&& g) {
  using namespace sg;
  SubgraphPropertyPtr subg_prop = std::make_shared<SgRemoveSparseProperty>();
  // Create undirected graph for ease of finding subgraphs
  std::vector<SimpleNodePtr> simple_nodes;
  CreateSimpleGraphRS(g, &simple_nodes);
  ModifyGraphRS(&g, *subg_prop, simple_nodes);

  return g;
}

NNVM_REGISTER_PASS(PartitionGraphRemoveSparse)
.describe("Partition a graph according to the user defined rules "
          "in a derived class of SubgraphProperty")
.set_body(PartitionGraphRemoveSparse)
.set_change_graph(true);
}  // namespace op
}  // namespace mxnet
