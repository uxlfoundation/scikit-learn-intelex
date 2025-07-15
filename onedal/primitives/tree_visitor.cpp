/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/common.hpp"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "numpy/arrayobject.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>
#include <vector>

#define ONEDAL_PY_TERMINAL_NODE -1
#define ONEDAL_PY_NO_FEATURE    -2

namespace py = pybind11;

namespace dal = oneapi::dal;
namespace df = dal::decision_forest;

using namespace pybind11::literals;

namespace oneapi::dal::python {

inline static const double get_nan64() {
    return std::numeric_limits<double>::quiet_NaN();
}

// equivalent for numpy arange
template <typename T>
inline std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> res;
    for (T i = start; i < stop; i += step)
        res.push_back(i);
    return res;
}

// This is the layout that sklearn expects for its tree traversal mechanics
class skl_tree_node {
public:
    Py_ssize_t left_child;
    Py_ssize_t right_child;
    Py_ssize_t feature;
    double threshold;
    double impurity;
    Py_ssize_t n_node_samples;
    double weighted_n_node_samples;
    unsigned char missing_go_to_left;

    skl_tree_node()
            : left_child(ONEDAL_PY_TERMINAL_NODE),
              right_child(ONEDAL_PY_TERMINAL_NODE),
              feature(ONEDAL_PY_NO_FEATURE),
              threshold(ONEDAL_PY_NO_FEATURE),
              impurity(get_nan64()),
              n_node_samples(0),
              weighted_n_node_samples(0.0),
              missing_go_to_left(false) {}
};

// We only expose the minimum information to python
template <typename T>
struct tree_state {
    py::array_t<skl_tree_node> node_ar;
    py::array_t<double> value_ar;
    std::size_t max_depth;
    std::size_t node_count;
    std::size_t leaf_count;
    std::size_t class_count;
};
// Declaration and implementation.
template <typename Task>
class node_count_visitor {
public:
    node_count_visitor() : n_nodes(0), depth(0), n_leaf_nodes(0) {}
    bool call(const df::leaf_node_info<Task>& info) {
        ++n_nodes;
        ++n_leaf_nodes;
        depth = std::max(depth, static_cast<const std::size_t>(info.get_level()));
        return true;
    }
    bool call(const df::split_node_info<Task>& info) {
        ++n_nodes;
        depth = std::max(depth, static_cast<const std::size_t>(info.get_level()));
        return true;
    }

    std::size_t n_nodes;
    std::size_t depth;
    std::size_t n_leaf_nodes;
};

template <typename Task, typename Impl>
class node_visitor {
public:
    Impl* const p_impl;
    node_visitor(Impl* impl) : p_impl{ impl } {
        assert(impl != nullptr);
    }

    node_visitor(node_visitor&&) = default;

    bool operator()(const df::leaf_node_info<Task>& info) {
        return p_impl->call(info);
    }
    bool operator()(const df::split_node_info<Task>& info) {
        return p_impl->call(info);
    }
};

// our tree visitor for getting tree state
template <typename Task>
class to_sklearn_tree_object_visitor : public tree_state<Task> {
public:
    to_sklearn_tree_object_visitor(std::size_t _depth,
                                   std::size_t _n_nodes,
                                   std::size_t _n_leafs,
                                   std::size_t _max_n_classes);
    bool call(const df::leaf_node_info<Task>& info);
    bool call(const df::split_node_info<Task>& info);
    double* value_ar_ptr;
    skl_tree_node* node_ar_ptr;

protected:
    std::size_t node_id;
    std::size_t max_n_classes;
    std::vector<Py_ssize_t> parents;
    void _onLeafNode(const df::leaf_node_info<Task>& info);
};

template <typename Task>
to_sklearn_tree_object_visitor<Task>::to_sklearn_tree_object_visitor(std::size_t _depth,
                                                                     std::size_t _n_nodes,
                                                                     std::size_t _n_leafs,
                                                                     std::size_t _max_n_classes)
        : node_id(0),
          parents(arange<Py_ssize_t>(-1, _depth - 1)) {
    this->max_n_classes = _max_n_classes;
    this->node_count = _n_nodes;
    this->max_depth = _depth;
    this->leaf_count = _n_leafs;
    this->class_count = _max_n_classes;

    auto node_ar_shape = py::array::ShapeContainer({ this->node_count });
    auto node_ar_strides = py::array::StridesContainer({ sizeof(skl_tree_node) });

    auto value_ar_shape = py::array::ShapeContainer({ static_cast<Py_ssize_t>(this->node_count),
                                                      1,
                                                      static_cast<Py_ssize_t>(this->class_count) });
    auto value_ar_strides = py::array::StridesContainer(
        { this->class_count * sizeof(double), this->class_count * sizeof(double), sizeof(double) });

    OVERFLOW_CHECK_BY_MULTIPLICATION(std::size_t, this->node_count, this->class_count);

    this->node_ar_ptr = new skl_tree_node[this->node_count];
    this->value_ar_ptr = new double[this->node_count * this->class_count]();

    // array_t doesn't initialize the underlying memory with the object's constructor
    // so the values will not match what is defined above, must be done on C++ side

    py::capsule free_value_ar(this->value_ar_ptr, [](void* f) {
        double* value_ar_ptr = reinterpret_cast<double*>(f);
        delete[] value_ar_ptr;
    });

    py::capsule free_node_ar(this->node_ar_ptr, [](void* f) {
        skl_tree_node* node_ar_ptr = reinterpret_cast<skl_tree_node*>(f);
        delete[] node_ar_ptr;
    });

    this->node_ar =
        py::array_t<skl_tree_node>(node_ar_shape, node_ar_strides, this->node_ar_ptr, free_node_ar);
    this->value_ar =
        py::array_t<double>(value_ar_shape, value_ar_strides, this->value_ar_ptr, free_value_ar);
}

template <typename Task>
bool to_sklearn_tree_object_visitor<Task>::call(const df::split_node_info<Task>& info) {
    if (info.get_level() > 0) {
        // has parents
        Py_ssize_t parent = parents[info.get_level() - 1];
        if (this->node_ar_ptr[parent].left_child > 0) {
            assert(this->node_ar_ptr[node_id].right_child < 0);
            this->node_ar_ptr[parent].right_child = node_id;
        }
        else {
            this->node_ar_ptr[parent].left_child = node_id;
        }
    }

    parents[info.get_level()] = node_id;
    this->node_ar_ptr[node_id].feature = info.get_feature_index();
    this->node_ar_ptr[node_id].threshold = info.get_feature_value();
    this->node_ar_ptr[node_id].impurity = info.get_impurity();
    this->node_ar_ptr[node_id].n_node_samples = info.get_sample_count();
    this->node_ar_ptr[node_id].weighted_n_node_samples =
        static_cast<double>(info.get_sample_count());
    this->node_ar_ptr[node_id].missing_go_to_left = false;

    // wrap-up
    ++node_id;
    return true;
}

// stuff that is done for all leaf node types
template <typename Task>
void to_sklearn_tree_object_visitor<Task>::_onLeafNode(const df::leaf_node_info<Task>& info) {
    if (info.get_level()) {
        Py_ssize_t parent = parents[info.get_level() - 1];
        if (this->node_ar_ptr[parent].left_child > 0) {
            assert(this->node_ar_ptr[node_id].right_child < 0);
            this->node_ar_ptr[parent].right_child = node_id;
        }
        else {
            this->node_ar_ptr[parent].left_child = node_id;
        }
    }

    this->node_ar_ptr[node_id].impurity = info.get_impurity();
    this->node_ar_ptr[node_id].n_node_samples = info.get_sample_count();
    this->node_ar_ptr[node_id].weighted_n_node_samples =
        static_cast<double>(info.get_sample_count());
    this->node_ar_ptr[node_id].missing_go_to_left = false;
}

template <>
bool to_sklearn_tree_object_visitor<df::task::regression>::call(
    const df::leaf_node_info<df::task::regression>& info) {
    _onLeafNode(info);
    OVERFLOW_CHECK_BY_MULTIPLICATION(std::size_t, node_id, class_count);

    this->value_ar_ptr[node_id * this->class_count] = info.get_response();

    // wrap-up
    ++node_id;
    return true;
}

template <>
bool to_sklearn_tree_object_visitor<df::task::classification>::call(
    const df::leaf_node_info<df::task::classification>& info) {
    std::size_t depth = static_cast<const std::size_t>(info.get_level());
    const std::size_t label =
        info.get_response(); // these may be a slow accesses due to oneDAL abstraction
    const double nNodeSampleCount =
        static_cast<const double>(info.get_sample_count()); // do them only once

    while (depth--) {
        const std::size_t id = parents[depth];
        const std::size_t row = id * this->class_count;
        this->value_ar_ptr[row + label] += nNodeSampleCount;
    }
    _onLeafNode(info);
    this->value_ar_ptr[node_id * this->class_count + label] += nNodeSampleCount;

    // wrap-up
    ++node_id;
    return true;
}

template <typename Task>
void init_get_tree_state(py::module_& m) {
    using namespace decision_forest;
    using model_t = model<Task>;
    using tree_state_t = tree_state<Task>;

    // TODO:
    // create one instance for cls and reg.
    py::class_<tree_state_t>(m, "get_tree_state")
        .def(py::init([](const model_t& model, std::size_t iTree, std::size_t n_classes) {
            // First count nodes
            node_count_visitor<Task> ncv;
            node_visitor<Task, decltype(ncv)> ncv_decorator{ &ncv };

            model.traverse_depth_first(iTree, std::move(ncv_decorator));
            // then do the final tree traversal
            to_sklearn_tree_object_visitor<Task> tsv(ncv.depth,
                                                     ncv.n_nodes,
                                                     ncv.n_leaf_nodes,
                                                     n_classes);
            node_visitor<Task, decltype(tsv)> tsv_decorator{ &tsv };
            model.traverse_depth_first(iTree, std::move(tsv_decorator));
            tree_state_t output = tree_state_t(tsv);
            // convert the value_ar to fractional values, rather than total ones
            // the last axis (2) is the n_classes, which must be summed for the
            // fraction
            if (output.class_count > 1) {
                output.value_ar = output.value_ar /
                                  output.value_ar.attr("sum")("axis"_a = 2, "keepdims"_a = true);
            }
            return output;
        }))
        .def_readwrite("node_ar", &tree_state_t::node_ar, py::return_value_policy::take_ownership)
        .def_readwrite("value_ar", &tree_state_t::value_ar, py::return_value_policy::take_ownership)
        .def_readwrite("max_depth", &tree_state_t::max_depth)
        .def_readwrite("node_count", &tree_state_t::node_count)
        .def_readwrite("leaf_count", &tree_state_t::leaf_count)
        .def_readwrite("class_count", &tree_state_t::class_count);
}

ONEDAL_PY_TYPE2STR(decision_forest::task::classification, "classification");
ONEDAL_PY_TYPE2STR(decision_forest::task::regression, "regression");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_get_tree_state);

ONEDAL_PY_INIT_MODULE(get_tree) {
    using namespace decision_forest;
    using namespace dal::detail;

    PYBIND11_NUMPY_DTYPE(skl_tree_node,
                         left_child,
                         right_child,
                         feature,
                         threshold,
                         impurity,
                         n_node_samples,
                         weighted_n_node_samples,
                         missing_go_to_left);

    using task_list = types<task::classification, task::regression>;
    auto sub = m.def_submodule("get_tree");
#ifndef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_get_tree_state, sub, task_list);
#endif
}
} // namespace oneapi::dal::python
