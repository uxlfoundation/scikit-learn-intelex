/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20260100

#include "oneapi/dal/algo/hdbscan.hpp"

#include <regex>

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Task, typename Ops>
struct hdbscan_method2t {
    hdbscan_method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace hdbscan;

        const auto method = params["method"].cast<std::string>();

        ONEDAL_PARAM_DISPATCH_VALUE(method, "brute_force", ops, Float, method::brute_force);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "kd_tree", ops, Float, method::kd_tree);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "ball_tree", ops, Float, method::ball_tree);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

static auto get_hdbscan_result_options(const py::dict& params) {
    using namespace dal::hdbscan;

    auto result_options_str = params["result_options"].cast<std::string>();
    result_option_id onedal_options;

    try {
        std::regex re("\\w+");
        const std::sregex_iterator last{};
        const std::sregex_iterator first( //
            result_options_str.begin(),
            result_options_str.end(),
            re);

        for (std::sregex_iterator it = first; it != last; ++it) {
            std::smatch match = *it;
            if (match.str() == "responses") {
                onedal_options = onedal_options | result_options::responses;
            }
            else if (match.str() == "core_observation_indices") {
                onedal_options = onedal_options | result_options::core_observation_indices;
            }
            else if (match.str() == "core_observations") {
                onedal_options = onedal_options | result_options::core_observations;
            }
            else if (match.str() == "core_flags") {
                onedal_options = onedal_options | result_options::core_flags;
            }
            else if (match.str() == "cluster_centers") {
                onedal_options = onedal_options | result_options::cluster_centers;
            }
            else if (match.str() == "medoid_centers") {
                onedal_options = onedal_options | result_options::medoid_centers;
            }
            else
                ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_options);
        }
    }
    catch (std::regex_error& e) {
        (void)e;
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_options);
    }

    return onedal_options;
}

static hdbscan::distance_metric parse_metric(const std::string& metric) {
    using namespace hdbscan;
    if (metric == "euclidean")
        return distance_metric::euclidean;
    if (metric == "manhattan")
        return distance_metric::manhattan;
    if (metric == "minkowski")
        return distance_metric::minkowski;
    if (metric == "chebyshev")
        return distance_metric::chebyshev;
    if (metric == "cosine")
        return distance_metric::cosine;
    throw std::runtime_error("Invalid value for parameter <metric>: " + metric);
}

struct hdbscan_params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::hdbscan;

        const auto min_cluster_size = params["min_cluster_size"].cast<std::int64_t>();
        const auto min_samples = params["min_samples"].cast<std::int64_t>();
        auto desc = descriptor<Float, Method, Task>(min_cluster_size, min_samples);
        desc.set_result_options(get_hdbscan_result_options(params));

        if (params.contains("metric")) {
            desc.set_metric(parse_metric(params["metric"].cast<std::string>()));
        }
        if (params.contains("degree")) {
            desc.set_degree(params["degree"].cast<double>());
        }
        if (params.contains("cluster_selection")) {
            const auto cs = params["cluster_selection"].cast<std::string>();
            if (cs == "leaf")
                desc.set_cluster_selection(cluster_selection_method::leaf);
            else
                desc.set_cluster_selection(cluster_selection_method::eom);
        }
        if (params.contains("allow_single_cluster")) {
            desc.set_allow_single_cluster(params["allow_single_cluster"].cast<bool>());
        }
        if (params.contains("cluster_selection_epsilon")) {
            desc.set_cluster_selection_epsilon(params["cluster_selection_epsilon"].cast<double>());
        }
        if (params.contains("max_cluster_size")) {
            desc.set_max_cluster_size(params["max_cluster_size"].cast<std::int64_t>());
        }
        if (params.contains("alpha")) {
            desc.set_alpha(params["alpha"].cast<double>());
        }
        if (params.contains("leaf_size")) {
            desc.set_leaf_size(params["leaf_size"].cast<std::int64_t>());
        }
        if (params.contains("store_centers")) {
            const auto sc = params["store_centers"].cast<std::string>();
            if (sc == "centroid")
                desc.set_store_centers(store_centers_method::centroid);
            else if (sc == "medoid")
                desc.set_store_centers(store_centers_method::medoid);
            else if (sc == "both")
                desc.set_store_centers(store_centers_method::both);
            else
                desc.set_store_centers(store_centers_method::none);
        }

        return desc;
    }
};

template <typename Policy, typename Task>
void init_hdbscan_compute_ops(py::module_& m) {
    m.def("compute", [](const Policy& policy, const py::dict& params, const table& data) {
        using namespace hdbscan;
        using input_t = compute_input<Task>;

        compute_ops ops(policy, input_t{ data }, hdbscan_params2desc{});
        return fptype2t{ hdbscan_method2t{ Task{}, ops } }(params);
    });
}

template <typename Task>
void init_hdbscan_compute_result(py::module_& m) {
    using namespace hdbscan;
    using result_t = compute_result<Task>;

    py::class_<result_t>(m, "compute_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(responses, result_t)
        .DEF_ONEDAL_PY_PROPERTY(core_flags, result_t)
        .DEF_ONEDAL_PY_PROPERTY(core_observation_indices, result_t)
        .DEF_ONEDAL_PY_PROPERTY(core_observations, result_t)
        .DEF_ONEDAL_PY_PROPERTY(result_options, result_t)
        .DEF_ONEDAL_PY_PROPERTY(cluster_count, result_t)
        .DEF_ONEDAL_PY_PROPERTY(cluster_centers, result_t)
        .DEF_ONEDAL_PY_PROPERTY(medoid_centers, result_t);
}

ONEDAL_PY_TYPE2STR(hdbscan::task::clustering, "clustering");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_hdbscan_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_hdbscan_compute_result);

ONEDAL_PY_INIT_MODULE(hdbscan) {
    using namespace dal::detail;
    using namespace hdbscan;
    using namespace dal::hdbscan;

    using task_list = types<task::clustering>;
    auto sub = m.def_submodule("hdbscan");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_hdbscan_compute_ops, sub, policy_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_hdbscan_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_hdbscan_compute_result, sub, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

} // namespace oneapi::dal::python

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20260100
