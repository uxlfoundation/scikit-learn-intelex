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

#include "oneapi/dal/algo/kmeans.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#include <regex>

namespace py = pybind11;

namespace oneapi::dal::python {

namespace kmeans {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::kmeans;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "lloyd_dense", ops, Float, method::lloyd_dense);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
        ONEDAL_PARAM_DISPATCH_VALUE(method, "lloyd_csr", ops, Float, method::lloyd_csr);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Float, typename Method, typename Task>
struct descriptor_creator {};

template <typename Float, typename Method>
struct descriptor_creator<Float, Method, dal::kmeans::task::clustering> {
    static auto get() {
        return dal::kmeans::descriptor<Float, Method, dal::kmeans::task::clustering>{};
    }
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        using namespace dal::kmeans;

        auto desc = descriptor_creator<Float, Method, Task>::get();

        desc.set_cluster_count(params["cluster_count"].cast<std::int64_t>());
        desc.set_accuracy_threshold(params["accuracy_threshold"].cast<Float>());
        desc.set_max_iteration_count(params["max_iteration_count"].cast<std::int64_t>());
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240200
        auto result_options = params["result_options"].cast<std::string>();
        if (result_options == "compute_exact_objective_function") {
            desc.set_result_options(result_options::compute_exact_objective_function);
        }
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240200
        return desc;
    }
};

template <typename Policy, typename Task>
struct init_train_ops_dispatcher {};

template <typename Policy>
struct init_train_ops_dispatcher<Policy, dal::kmeans::task::clustering> {
    void operator()(py::module_& m) {
        using Task = dal::kmeans::task::clustering;
        m.def("train",
              [](const Policy& policy,
                 const py::dict& params,
                 const table& data,
                 const table& initial_centroids) {
                  using namespace dal::kmeans;
                  using input_t = train_input<Task>;

                  train_ops ops(policy, input_t{ data, initial_centroids }, params2desc{});
                  return fptype2t{ method2t{ Task{}, ops } }(params);
              });
    }
};

template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
    init_train_ops_dispatcher<Policy, Task>{}(m);
}

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const dal::kmeans::model<Task>& model,
             const table& data) {
              using namespace dal::kmeans;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ model, data }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_model(py::module_& m) {
    using namespace dal::kmeans;
    using model_t = model<Task>;

    auto cls = py::class_<model_t>(m, "model")
                   .def(py::init())
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
                   .def(py::pickle(
                       [](const model_t& m) {
                           return serialize(m);
                       },
                       [](const py::bytes& bytes) {
                           return deserialize<model_t>(bytes);
                       }))
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
                   .DEF_ONEDAL_PY_PROPERTY(centroids, model_t);
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::kmeans;
    using result_t = train_result<Task>;

    auto cls = py::class_<result_t>(m, "train_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(model, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(responses, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(iteration_count, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(objective_function_value, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::kmeans;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(responses, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(objective_function_value, result_t);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);

} // namespace kmeans

ONEDAL_PY_INIT_MODULE(kmeans) {
    using namespace kmeans;
    using namespace dal::detail;
    using namespace dal::kmeans;

    using task_list = types<task::clustering>;
    auto sub = m.def_submodule("kmeans");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_spmd, task_list);
    // kmeans infer has no host SPMD export in oneDAL; keep it dpc-only.
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_spmd_dpc, task_list);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

ONEDAL_PY_TYPE2STR(dal::kmeans::task::clustering, "clustering");

} // namespace oneapi::dal::python
