/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/knn.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"
#include "onedal/primitives/pairwise_distances.hpp"
#include <regex>

namespace py = pybind11;

namespace oneapi::dal::python {

namespace neighbors {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::knn;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "brute", ops, Float, method::brute_force);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "kd_tree", ops, Float, method::kd_tree);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Ops>
struct metric2t {
    metric2t(const Ops& ops) : ops(ops) {}

    template <typename Float, typename Method>
    auto operator()(const py::dict& params) {
        using namespace dal::knn;

        auto metric = params["metric"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(metric,
                                    "manhattan",
                                    ops,
                                    Float,
                                    Method,
                                    minkowski_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric,
                                    "minkowski",
                                    ops,
                                    Float,
                                    Method,
                                    minkowski_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric,
                                    "euclidean",
                                    ops,
                                    Float,
                                    Method,
                                    minkowski_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric,
                                    "chebyshev",
                                    ops,
                                    Float,
                                    Method,
                                    chebyshev_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric,
                                    "cosine",
                                    ops,
                                    Float,
                                    Method,
                                    cosine_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric,
                                    "correlation",
                                    ops,
                                    Float,
                                    Method,
                                    correlation_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(metric);
    }

    Ops ops;
};

auto get_onedal_voting_mode(const py::dict& params) {
    using namespace dal::knn;

    auto weights = params["vote_weights"].cast<std::string>();
    if (weights == "uniform") {
        return voting_mode::uniform;
    }
    else if (weights == "distance") {
        return voting_mode::distance;
    }
    else
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(weights);
}

auto get_onedal_result_options(const py::dict& params) {
    using namespace dal::knn;

    auto result_option = params["result_option"].cast<std::string>();
    result_option_id onedal_options;

    try {
        std::regex re("\\w+");
        std::sregex_iterator next(result_option.begin(), result_option.end(), re);
        std::sregex_iterator end;
        while (next != end) {
            std::smatch match = *next;
            if (match.str() == "responses") {
                onedal_options = onedal_options | result_options::responses;
            }
            else if (match.str() == "indices") {
                onedal_options = onedal_options | result_options::indices;
            }
            else if (match.str() == "distances") {
                onedal_options = onedal_options | result_options::distances;
            }
            else
                ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
            next++;
        }
    }
    catch (std::regex_error& e) {
        (void)e;
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
    }

    return onedal_options;
}

template <typename Float, typename Method, typename Task, typename Distance>
struct descriptor_creator;

template <typename Float, typename Method, typename Distance>
struct descriptor_creator<Float, Method, knn::task::classification, Distance> {
    static auto get(const std::int64_t class_count, const std::int64_t neighbor_count) {
        return knn::descriptor<Float, Method, knn::task::classification, Distance>(class_count,
                                                                                   neighbor_count);
    }
};

template <typename Float, typename Method, typename Distance>
struct descriptor_creator<Float, Method, knn::task::regression, Distance> {
    static auto get(const std::int64_t, const std::int64_t neighbor_count) {
        return knn::descriptor<Float, Method, knn::task::regression, Distance>(neighbor_count);
    }
};

template <typename Float, typename Method, typename Distance>
struct descriptor_creator<Float, Method, knn::task::search, Distance> {
    static auto get(const std::int64_t, const std::int64_t neighbor_count) {
        return knn::descriptor<Float, Method, knn::task::search, Distance>(neighbor_count);
    }
};

struct params2desc {
    template <typename Float, typename Method, typename Task, typename Distance>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::knn;

        constexpr bool is_bf = std::is_same_v<Method, method::brute_force>;
        const auto class_count = params["class_count"].cast<std::int64_t>();
        const auto neighbor_count = params["neighbor_count"].cast<std::int64_t>();

        auto desc =
            descriptor_creator<Float, Method, Task, Distance>::get(class_count, neighbor_count)
                .set_voting_mode(get_onedal_voting_mode(params))
                .set_result_options(get_onedal_result_options(params));
        if constexpr (is_bf) {
            desc.set_distance(get_distance_descriptor<Distance>(params));
        }
        return desc;
    }
};

template <typename Policy, typename Task>
struct init_train_ops_dispatcher {};

template <typename Policy>
struct init_train_ops_dispatcher<Policy, knn::task::classification> {
    void operator()(py::module_& m) {
        using Task = knn::task::classification;
        m.def("train",
              [](const Policy& policy,
                 const py::dict& params,
                 const table& data,
                 const table& responses) {
                  using namespace dal::knn;
                  using input_t = train_input<Task>;

                  train_ops ops(policy, input_t{ data, responses }, params2desc{});
                  return fptype2t{ method2t{ Task{}, metric2t{ ops } } }(params);
              });
    }
};

template <typename Policy>
struct init_train_ops_dispatcher<Policy, knn::task::regression> {
    void operator()(py::module_& m) {
        using Task = knn::task::regression;
        m.def("train",
              [](const Policy& policy,
                 const py::dict& params,
                 const table& data,
                 const table& responses) {
                  using namespace dal::knn;
                  using input_t = train_input<Task>;

                  train_ops ops(policy, input_t{ data, responses }, params2desc{});
                  return fptype2t{ method2t{ Task{}, metric2t{ ops } } }(params);
              });
    }
};

template <typename Policy>
struct init_train_ops_dispatcher<Policy, knn::task::search> {
    void operator()(py::module_& m) {
        using Task = knn::task::search;
        m.def("train", [](const Policy& policy, const py::dict& params, const table& data) {
            using namespace dal::knn;
            using input_t = train_input<Task>;

            train_ops ops(policy, input_t{ data }, params2desc{});
            return fptype2t{ method2t{ Task{}, metric2t{ ops } } }(params);
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
             const dal::knn::model<Task>& model,
             const table& data) {
              using namespace dal::knn;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ data, model }, params2desc{});
              return fptype2t{ method2t{ Task{}, metric2t{ ops } } }(params);
          });
}

template <typename Task>
void init_model(py::module_& m) {
    using namespace dal::knn;
    using model_t = model<Task>;

    auto cls = py::class_<model_t>(m, "model")
                   .def(py::init())
                   .def(py::pickle(
                       [](const model_t& m) {
                           return serialize(m);
                       },
                       [](const py::bytes& bytes) {
                           return deserialize<model_t>(bytes);
                       }));
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::knn;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result").def(py::init()).DEF_ONEDAL_PY_PROPERTY(model, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::knn;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(indices, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(distances, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(result_options, result_t);

    constexpr bool is_not_srch = !std::is_same_v<Task, task::search>;

    if constexpr (is_not_srch) {
        // workaround for gcc which cannot deduce setters directly passed to def_property()
        auto setter = &result_t::template set_responses<>;
        cls.def_property("responses", &result_t::get_responses, setter);
    }
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);

} // namespace neighbors

ONEDAL_PY_INIT_MODULE(neighbors) {
    using namespace neighbors;
    using namespace dal::knn;
    using namespace dal::detail;

    using task_list = types<task::classification, task::regression, task::search>;
    auto sub = m.def_submodule("neighbors");

#if defined(ONEDAL_DATA_PARALLEL_SPMD) && defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_spmd, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_spmd, task_list);
#else // defined(ONEDAL_DATA_PARALLEL_SPMD) && defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
#endif // defined(ONEDAL_DATA_PARALLEL_SPMD) && defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
}

ONEDAL_PY_TYPE2STR(dal::knn::task::classification, "classification");
ONEDAL_PY_TYPE2STR(dal::knn::task::regression, "regression");
ONEDAL_PY_TYPE2STR(dal::knn::task::search, "search");

} // namespace oneapi::dal::python
