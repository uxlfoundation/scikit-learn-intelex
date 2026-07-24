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

#include "oneapi/dal/algo/kmeans_init.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#include <type_traits>
#include <regex>

namespace py = pybind11;

namespace oneapi::dal::python {

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200

namespace kmeans_init {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::kmeans_init;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "random_dense", ops, Float, method::random_dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "plus_plus_dense", ops, Float, method::plus_plus_dense);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
        ONEDAL_PARAM_DISPATCH_VALUE(method, "random_csr", ops, Float, method::random_csr);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "plus_plus_csr", ops, Float, method::plus_plus_csr);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240700
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Float, typename Method, typename Task>
struct descriptor_creator;

template <typename Float>
struct descriptor_creator<Float, dal::kmeans_init::method::dense, dal::kmeans_init::task::init> {
    static auto get() {
        return dal::kmeans_init::
            descriptor<Float, dal::kmeans_init::method::dense, dal::kmeans_init::task::init>{};
    }
};

template <typename Float>
struct descriptor_creator<Float,
                          dal::kmeans_init::method::random_dense,
                          dal::kmeans_init::task::init> {
    static auto get() {
        return dal::kmeans_init::descriptor<Float,
                                            dal::kmeans_init::method::random_dense,
                                            dal::kmeans_init::task::init>{};
    }
};

template <typename Float>
struct descriptor_creator<Float,
                          dal::kmeans_init::method::plus_plus_dense,
                          dal::kmeans_init::task::init> {
    static auto get() {
        return dal::kmeans_init::descriptor<Float,
                                            dal::kmeans_init::method::plus_plus_dense,
                                            dal::kmeans_init::task::init>{};
    }
};

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
template <typename Float>
struct descriptor_creator<Float,
                          dal::kmeans_init::method::random_csr,
                          dal::kmeans_init::task::init> {
    static auto get() {
        return dal::kmeans_init::
            descriptor<Float, dal::kmeans_init::method::random_csr, dal::kmeans_init::task::init>{};
    }
};

template <typename Float>
struct descriptor_creator<Float,
                          dal::kmeans_init::method::plus_plus_csr,
                          dal::kmeans_init::task::init> {
    static auto get() {
        return dal::kmeans_init::descriptor<Float,
                                            dal::kmeans_init::method::plus_plus_csr,
                                            dal::kmeans_init::task::init>{};
    }
};
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240700

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        using namespace dal::kmeans_init;

        const auto cluster_count = params["cluster_count"].cast<std::int64_t>();

        auto desc = descriptor_creator<Float, Method, Task>::get() //
                        .set_cluster_count(cluster_count);

        if constexpr (!std::is_same_v<Method, dal::kmeans_init::method::dense>) {
            const auto seed = params["seed"].cast<std::int64_t>();
            desc.set_seed(seed);
        }

        if constexpr (std::is_same_v<Method, dal::kmeans_init::method::plus_plus_dense>) {
            const auto local_trials_count = params["local_trials_count"].cast<std::int64_t>();
            desc.set_local_trials_count(local_trials_count);
        }
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
        if constexpr (std::is_same_v<Method, dal::kmeans_init::method::plus_plus_csr>) {
            const auto local_trials_count = params["local_trials_count"].cast<std::int64_t>();
            desc.set_local_trials_count(local_trials_count);
        }
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240700
        return desc;
    }
};

template <typename Policy, typename Task>
struct init_compute_ops_dispatcher {};

template <typename Policy>
struct init_compute_ops_dispatcher<Policy, dal::kmeans_init::task::init> {
    void operator()(py::module_& m) {
        using Task = dal::kmeans_init::task::init;
        m.def("compute", [](const Policy& policy, const py::dict& params, const table& data) {
            using namespace dal::kmeans_init;
            using input_t = compute_input<Task>;

            compute_ops ops(policy, input_t{ data }, params2desc{});
            return fptype2t{ method2t{ Task{}, ops } }(params);
        });
    }
};

template <typename Policy, typename Task>
void init_compute_ops(py::module& m) {
    init_compute_ops_dispatcher<Policy, Task>{}(m);
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace dal::kmeans_init;
    using result_t = compute_result<Task>;

    auto cls = py::class_<result_t>(m, "kmeans_init")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(centroids, result_t);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);

} // namespace kmeans_init

ONEDAL_PY_INIT_MODULE(kmeans_init) {
    using namespace dal::detail;
    using namespace kmeans_init;
    using namespace dal::kmeans_init;

    using task_list = types<task::init>;
    auto sub = m.def_submodule("kmeans_init");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

ONEDAL_PY_TYPE2STR(dal::kmeans_init::task::init, "init");

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20230200

} // namespace oneapi::dal::python
