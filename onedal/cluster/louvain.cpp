/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <type_traits>
#include <iostream>
#include "oneapi/dal/algo/louvain.hpp"

#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/detail/memory.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"
#include "onedal/datatypes/numpy/data_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Float>
using graph_t = dal::preview::undirected_adjacency_vector_graph<std::int32_t, Float>;

template <typename Type>
inline void _table_checks(const table& input, Type* &ptr, std::int64_t &length) {
    std::cout << "start_table_check\n";
    if (input.get_kind() == dal::homogen_table::kind()){
        const auto &homogen_input = static_cast<const dal::homogen_table &>(input);
        // verify matching datatype
#define CHECK_DTYPE(CType) if (!std::is_same<Type, CType>::value) std::invalid_argument("Incorrect dtype");

        SET_CTYPE_FROM_DAL_TYPE(homogen_input.get_metadata().get_data_type(0),
                                CHECK_DTYPE,
                                std::invalid_argument("Unknown table dtype"))

        
        // verify only one column
        if (homogen_input.get_column_count() != 1){
            throw std::invalid_argument("Incorrect dimensions.");
        }

        // get length
        length = static_cast<std::int64_t>(homogen_input.get_row_count());

        // get pointer
        auto bytes_array = dal::detail::get_original_data(homogen_input);
        const bool is_mutable = bytes_array.has_mutable_data();

        ptr = is_mutable ? reinterpret_cast<Type *>(bytes_array.get_mutable_data())
                         : const_cast<Type *>(reinterpret_cast<const Type *>(bytes_array.get_data()));

    } else {
        throw std::invalid_argument("Non-homogen table input.");
    }
}

template <typename Float>
graph_t<Float> tables_to_undirected_graph(const table& data, const table& indices, const table& indptr){
// because oneDAL graphs do not allow have the ability to call python capsule destructors
// graphs cannot be directly created from numpy array types. Conversion from oneDAL
// tables makes a simple and consitent interface which matches other estimators. The
// csr table cannot be used because of the data type of the indicies and indptr which
// are hardcoded to int64 and because they are 1 indexed.
    graph_t<Float> res;

    Float *edge_ptr;
    std::int32_t *cols;
    std::int64_t *rows, data_count, col_count, vertex_count;

    _table_checks<Float>(data, edge_ptr, data_count);
    _table_checks<std::int32_t>(indices, cols, col_count);
    _table_checks<std::int64_t>(indptr, rows, vertex_count);
    
    // verify data and indices are same lengths
    if (data_count != col_count){
        throw std::invalid_argument("Got invalid csr object.");
    }
    // -1 needed to match oneDAL graph inputs
    vertex_count--;
    
    // Undirected graphs in oneDAL do not check for self-loops.  This will iterate through
    // the data to verify that nothing along the diagonal is stored in the csr format.
    // This closely resembles scipy.sparse
    std::int64_t N = col_count < vertex_count ? col_count : vertex_count;
    std::cout << "access_problems\n";

    for(std::int64_t u=0; u < N; ++u) {
        std::int64_t row_begin = rows[u];
        std::int64_t row_end = rows[u + 1];
        for(std::int64_t j = row_begin; j < row_end; ++j){
            if (cols[j] == u) {
                throw std::invalid_argument(
                    "Self-loops are not allowed.\n");
            }
        }
    }
    
    auto& graph_impl = dal::detail::get_impl(res);  
    using vertex_set_t = typename dal::preview::graph_traits<graph_t<Float>>::vertex_set;
    dal::preview::detail::rebinded_allocator ra(graph_impl._vertex_allocator);
    auto [degrees_array, degrees] = ra.template allocate_array<vertex_set_t>(vertex_count);
    for (std::int64_t u = 0; u < vertex_count; u++) {
        degrees[u] = rows[u + 1] - rows[u];
    }

    graph_impl.set_topology(vertex_count, col_count/2, rows, cols, col_count, degrees);
    graph_impl.set_edge_values(edge_ptr, col_count/2);
    std::cout << "graph_generated\n";
    
    return res;
}

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace preview::louvain;

        const auto method = params["method"].cast<std::string>();

        ONEDAL_PARAM_DISPATCH_VALUE(method, "fast", ops, Float, method::fast);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::preview::louvain;

        auto desc = descriptor<Float, Method, Task>();
        desc.set_resolution(params["resolution"].cast<double>());
        desc.set_accuracy_threshold(params["accuracy_threshold"].cast<double>());
        desc.set_max_iteration_count(params["max_iteration_count"].cast<std::int64_t>());

        return desc;
    }
};

template <typename Task>
void init_vertex_partitioning_ops(py::module_& m) {
    m.def("vertex_partitioning",
          [](const py::dict& params,
             const table& data,
             const table& indices,
             const table& indptr,
             const table& initial_partition) {
              using namespace preview::louvain;
              using input_t = vertex_partitioning_input<graph_t<double>, Task>;
              // create graphs from oneDAL tables
              graph_t<double> graph;
              // only int and double topologies are currently exported to the oneDAL shared object
              
              graph = tables_to_undirected_graph<double>(data, indices, indptr);
              std::cout << "running ops\n";
              vertex_partitioning_ops ops(input_t{ graph, initial_partition}, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
    m.def("vertex_partitioning",
          [](const py::dict& params,
             const table& data,
             const table& indices,
             const table& indptr) {
              using namespace preview::louvain;
              using input_t = vertex_partitioning_input<graph_t<double>, Task>;
              graph_t<double> graph;
              // only int and double topologies are currently exported to the oneDAL shared object
              graph = tables_to_undirected_graph<double>(data, indices, indptr);
              std::cout << "running ops\n";
              vertex_partitioning_ops ops(input_t{ graph }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_vertex_partitioning_result(py::module_& m) {
    using namespace preview::louvain;
    using result_t = vertex_partitioning_result<Task>;

    py::class_<result_t>(m, "vertex_paritioning_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(labels, result_t)
        .DEF_ONEDAL_PY_PROPERTY(modularity, result_t)
        .DEF_ONEDAL_PY_PROPERTY(community_count, result_t);
}

ONEDAL_PY_TYPE2STR(preview::louvain::task::vertex_partitioning, "vertex_partitioning");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_vertex_partitioning_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_vertex_partitioning_result);

ONEDAL_PY_INIT_MODULE(louvain) {
    using namespace dal::detail;
    using namespace dal::preview::louvain;

    using task_list = types<task::vertex_partitioning>;
    auto sub = m.def_submodule("louvain");

    ONEDAL_PY_INSTANTIATE(init_vertex_partitioning_ops, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_vertex_partitioning_result, sub, task_list);
}

} // namespace oneapi::dal::python
