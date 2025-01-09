/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _TREE_VISITOR_H_INCLUDED_
#define _TREE_VISITOR_H_INCLUDED_

#include "daal4py.h"
#include "daal4py_defines.h"
#include <daal.h>
#include <vector>
#include <algorithm>

#define TERMINAL_NODE -1
#define NO_FEATURE    -2

// cython will convert this struct into an numpy structured array
// This is the layout that sklearn expects for its tree traversal mechanics
struct skl_tree_node
{
    Py_ssize_t left_child;
    Py_ssize_t right_child;
    Py_ssize_t feature;
    double threshold;
    double impurity;
    Py_ssize_t n_node_samples;
    double weighted_n_node_samples;
    unsigned char missing_go_to_left;

    skl_tree_node() : left_child(TERMINAL_NODE), right_child(TERMINAL_NODE), feature(NO_FEATURE), threshold(NO_FEATURE), impurity(get_nan64()), n_node_samples(0), weighted_n_node_samples(0.0), missing_go_to_left(false) {}
};

// We'd like the Models to have the descriptor typedefs in the class
// For now we provide a meat-class to map Models to descriptors
// Models might need an explicit instantiation providing visitor_type, leaf_desc_type and split_desc_type
// This is the default template for models using regression visitors
template <typename M>
struct TNVT
{
    typedef daal::algorithms::tree_utils::regression::TreeNodeVisitor visitor_type;
    typedef daal::algorithms::tree_utils::regression::LeafNodeDescriptor leaf_desc_type;
    typedef daal::algorithms::tree_utils::regression::SplitNodeDescriptor split_desc_type;
};

// Decision forest classification uses classification vistors
template <>
struct TNVT<daal::algorithms::decision_forest::classification::Model>
{
    typedef daal::algorithms::tree_utils::classification::TreeNodeVisitor visitor_type;
    typedef daal::algorithms::tree_utils::classification::LeafNodeDescriptor leaf_desc_type;
    typedef daal::algorithms::tree_utils::classification::SplitNodeDescriptor split_desc_type;
};

// Decision tree classification uses classification vistors
template <>
struct TNVT<daal::algorithms::decision_tree::classification::Model> : public TNVT<daal::algorithms::decision_forest::classification::Model>
{};

// our tree visitor for counting nodes
// TODO: Needs to store leaf-node response, and split-node impurity/sample_counts values
template <typename M>
class NodeDepthCountNodeVisitor : public TNVT<M>::visitor_type
{
public:
    NodeDepthCountNodeVisitor();
    virtual bool onLeafNode(const typename TNVT<M>::leaf_desc_type & desc);
    virtual bool onSplitNode(const typename TNVT<M>::split_desc_type & desc);

    size_t n_nodes;
    size_t depth;
    size_t n_leaf_nodes;
};

// equivalent for numpy arange
template <typename T>
std::vector<T> arange(T start, T stop, T step = 1)
{
    std::vector<T> res;
    for (T i = start; i < stop; i += step) res.push_back(i);
    return res;
}

// We only expose the minimum information to cython
struct TreeState
{
    skl_tree_node * node_ar;
    double * value_ar;
    size_t max_depth;
    size_t node_count;
    size_t leaf_count;
    size_t class_count;
};

// our tree visitor for getting tree state
template <typename M>
class toSKLearnTreeObjectVisitor : public TNVT<M>::visitor_type, public TreeState
{
public:
    toSKLearnTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes);
    virtual bool onSplitNode(const typename TNVT<M>::split_desc_type & desc);
    virtual bool onLeafNode(const typename TNVT<M>::leaf_desc_type & desc);

protected:
    // generic leaf node handling
    bool _onLeafNode(const daal::algorithms::tree_utils::NodeDescriptor & desc);
    // implementation of inLeafNode for regression visitors
    bool _onLeafNode(const typename TNVT<M>::leaf_desc_type & desc, std::false_type);
    // implementation of inLeafNode for classification visitors
    bool _onLeafNode(const typename TNVT<M>::leaf_desc_type & desc, std::true_type);

    size_t node_id;
    size_t max_n_classes;
    std::vector<Py_ssize_t> parents;
};

// This is the function for getting the tree state from a forest which we use in cython
// we will have different model types, so it's a template
// Note: the caller will own the memory of the 2 returned arrays!
template <typename M>
TreeState _getTreeState(M * model, size_t iTree, size_t n_classes)
{
    // First count nodes
    NodeDepthCountNodeVisitor<typename M::ElementType> ncv;
    (*model)->traverseDFS(iTree, ncv);
    // then do the final tree traversal
    toSKLearnTreeObjectVisitor<typename M::ElementType> tsv(ncv.depth, ncv.n_nodes, ncv.n_leaf_nodes, n_classes);
    (*model)->traverseDFS(iTree, tsv);

    return TreeState(tsv);
}

// This is the function for getting the tree state frmo a tree which we use in cython
// we will have different model types, so it's a template
// Note: the caller will own the memory of the 2 returned arrays!
template <typename M>
TreeState _getTreeState(M * model, size_t n_classes)
{
    // First count nodes
    NodeDepthCountNodeVisitor<typename M::ElementType> ncv;
    (*model)->traverseDFS(ncv);
    // then do the final tree traversal
    toSKLearnTreeObjectVisitor<typename M::ElementType> tsv(ncv.depth, ncv.n_nodes, ncv.n_leaf_nodes, n_classes);
    (*model)->traverseDFS(tsv);

    return TreeState(tsv);
}

// ****************************************************
// ****************************************************
// Visitor implementation
// ****************************************************
// ****************************************************

template <typename M>
NodeDepthCountNodeVisitor<M>::NodeDepthCountNodeVisitor() : n_nodes(0), depth(0), n_leaf_nodes(0)
{}

// TODO: Needs to store leaf-node response, and split-node impurity/sample_counts values
template <typename M>
bool NodeDepthCountNodeVisitor<M>::onLeafNode(const typename TNVT<M>::leaf_desc_type & desc)
{
    ++n_nodes;
    ++n_leaf_nodes;
    depth = std::max((const size_t)depth, desc.level);
    return true;
}

template <typename M>
bool NodeDepthCountNodeVisitor<M>::onSplitNode(const typename TNVT<M>::split_desc_type & desc)
{
    ++n_nodes;
    depth = std::max((const size_t)depth, desc.level);
    return true;
}

template <typename M>
toSKLearnTreeObjectVisitor<M>::toSKLearnTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes) : node_id(0), parents(arange<Py_ssize_t>(-1, _depth - 1))
{
    max_n_classes = _max_n_classes;
    node_count    = _n_nodes;
    max_depth     = _depth;
    leaf_count    = _n_leafs;
    class_count   = _max_n_classes;
    node_ar       = new skl_tree_node[node_count];
    value_ar      = new double[node_count * 1 * class_count](); // oneDAL only supports scalar responses for now
}

template <typename M>
bool toSKLearnTreeObjectVisitor<M>::onSplitNode(const typename TNVT<M>::split_desc_type & desc)
{
    if (desc.level > 0)
    {
        // has parents
        Py_ssize_t parent = parents[desc.level - 1];
        if (node_ar[parent].left_child > 0)
        {
            assert(node_ar[node_id].right_child < 0);
            node_ar[parent].right_child = node_id;
        }
        else
        {
            node_ar[parent].left_child = node_id;
        }
    }

    parents[desc.level]                      = node_id;
    node_ar[node_id].feature                 = desc.featureIndex;
    node_ar[node_id].threshold               = desc.featureValue;
    node_ar[node_id].impurity                = desc.impurity;
    node_ar[node_id].n_node_samples          = desc.nNodeSampleCount;
    node_ar[node_id].weighted_n_node_samples = desc.nNodeSampleCount;
    node_ar[node_id].missing_go_to_left      = false;

    // wrap-up
    ++node_id;
    return true;
}

template <typename M>
bool toSKLearnTreeObjectVisitor<M>::onLeafNode(const typename TNVT<M>::leaf_desc_type & desc)
{
    // we use somewhat complicated C++'11 construct to determine if the descriptor is for classification
    // The actual implementation is the overloaded _onLeafNode which depends on integral_constant types true_type or false_type
    // we might want to make this dependent on a more meaningful type than bool
    return _onLeafNode(desc, typename std::integral_constant<bool, std::is_base_of<daal::algorithms::tree_utils::classification::LeafNodeDescriptor, typename TNVT<M>::leaf_desc_type>::value>());
}

// stuff that is done for all leaf node types
template <typename M>
bool toSKLearnTreeObjectVisitor<M>::_onLeafNode(const daal::algorithms::tree_utils::NodeDescriptor & desc)
{
    if (desc.level)
    {
        Py_ssize_t parent = parents[desc.level - 1];
        if (node_ar[parent].left_child > 0)
        {
            assert(node_ar[node_id].right_child < 0);
            node_ar[parent].right_child = node_id;
        }
        else
        {
            node_ar[parent].left_child = node_id;
        }
    }

    node_ar[node_id].impurity                = desc.impurity;
    node_ar[node_id].n_node_samples          = desc.nNodeSampleCount;
    node_ar[node_id].weighted_n_node_samples = desc.nNodeSampleCount;
    node_ar[node_id].missing_go_to_left      = false;

    return true;
}

template <typename M>
bool toSKLearnTreeObjectVisitor<M>::_onLeafNode(const typename TNVT<M>::leaf_desc_type & desc, std::false_type)
{
    _onLeafNode(desc);
    DAAL4PY_OVERFLOW_CHECK_BY_MULTIPLICATION(int, node_id, class_count);
    value_ar[node_id * 1 * class_count] = desc.response;

    // wrap-up
    ++node_id;
    return true;
}

template <typename M>
bool toSKLearnTreeObjectVisitor<M>::_onLeafNode(const typename TNVT<M>::leaf_desc_type & desc, std::true_type)
{
    if (desc.level > 0)
    {
        size_t depth = desc.level - 1;
        while (depth >= 0)
        {
            size_t id = parents[depth];
            value_ar[id * 1 * class_count + desc.label] += desc.nNodeSampleCount;
            if (depth == 0)
            {
                break;
            }
            --depth;
        }
    }
    _onLeafNode(desc);
    DAAL4PY_OVERFLOW_CHECK_BY_ADDING(int, node_id * 1 * class_count, desc.label);
    value_ar[node_id * 1 * class_count + desc.label] += desc.nNodeSampleCount;

    // wrap-up
    ++node_id;
    return true;
}

#endif // _TREE_VISITOR_H_INCLUDED_
