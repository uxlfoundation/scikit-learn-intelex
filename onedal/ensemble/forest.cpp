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

#include "oneapi/dal/algo/decision_forest.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace decision_forest;

        const auto method = params["method"].cast<std::string>();

        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "hist", ops, Float, method::hist);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

std::vector<std::string> split(const std::string& str) {
    const auto size = str.size();
    std::vector<std::string> result;

    std::size_t start = 0;
    for (std::size_t i = 0; i < size; ++i) {
        if (str[i] == '|') {
            result.emplace_back(str.substr(start, i - start));
            start = i + 1;
        }
        else if (i == size - 1) {
            result.emplace_back(str.substr(start, i - start + 1));
        }
    }

    return result;
}

auto get_error_metric_mode(const py::dict& params) {
    using namespace decision_forest;

    auto mode = params["error_metric_mode"].cast<std::string>();
    auto modes = split(mode);
    const auto modes_num = modes.size();

    error_metric_mode result_mode = error_metric_mode::none;
    for (std::size_t i = 0; i < modes_num; ++i) {
        if (modes[i] == "none")
            result_mode |= error_metric_mode::none;
        else if (modes[i] == "out_of_bag_error")
            result_mode |= error_metric_mode::out_of_bag_error;
        else if (modes[i] == "out_of_bag_error_per_observation")
            result_mode |= error_metric_mode::out_of_bag_error_per_observation;
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230101
        else if (modes[i] == "out_of_bag_error_accuracy")
            result_mode |= error_metric_mode::out_of_bag_error_accuracy;
        else if (modes[i] == "out_of_bag_error_r2")
            result_mode |= error_metric_mode::out_of_bag_error_r2;
        else if (modes[i] == "out_of_bag_error_decision_function")
            result_mode |= error_metric_mode::out_of_bag_error_decision_function;
        else if (modes[i] == "out_of_bag_error_prediction")
            result_mode |= error_metric_mode::out_of_bag_error_prediction;
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20230101
        else
            ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(mode);
    }
    return result_mode;
}

auto get_infer_mode(const py::dict& params) {
    using namespace decision_forest;

    auto mode = params["infer_mode"].cast<std::string>();
    auto modes = split(mode);
    const auto modes_num = modes.size();

    infer_mode result_mode{};
    for (std::size_t i = 0; i < modes_num; ++i) {
        if (modes[i] == "class_responses")
            result_mode |= infer_mode::class_responses;
        else if (modes[i] == "class_probabilities")
            result_mode |= infer_mode::class_probabilities;
        else
            ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(mode);
    }
    return result_mode;
}

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230101
auto get_splitter_mode(const py::dict& params) {
    using namespace decision_forest;
    auto mode = params["splitter_mode"].cast<std::string>();
    if (mode == "best") {
        return splitter_mode::best;
    }
    else if (mode == "random") {
        return splitter_mode::random;
    }
    else
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(mode);
}
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20230101

auto get_variable_importance_mode(const py::dict& params) {
    using namespace decision_forest;

    auto mode = params["variable_importance_mode"].cast<std::string>();
    if (mode == "none") {
        return variable_importance_mode::none;
    }
    else if (mode == "mdi") {
        return variable_importance_mode::mdi;
    }
    else if (mode == "mda_raw") {
        return variable_importance_mode::mda_raw;
    }
    else if (mode == "mda_scaled") {
        return variable_importance_mode::mda_scaled;
    }
    else
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(mode);
}

auto get_voting_mode(const py::dict& params) {
    using namespace decision_forest;

    auto mode = params["voting_mode"].cast<std::string>();
    if (mode == "weighted") {
        return voting_mode::weighted;
    }
    else if (mode == "unweighted") {
        return voting_mode::unweighted;
    }
    else
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(mode);
}

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace decision_forest;

        constexpr bool is_cls = std::is_same_v<Task, task::classification>;

        auto desc = descriptor<Float, Method, Task>{}
                        .set_observations_per_tree_fraction(
                            params["observations_per_tree_fraction"].cast<double>())
                        .set_impurity_threshold(params["impurity_threshold"].cast<double>())
                        .set_min_weight_fraction_in_leaf_node(
                            params["min_weight_fraction_in_leaf_node"].cast<double>())
                        .set_min_impurity_decrease_in_split_node(
                            params["min_impurity_decrease_in_split_node"].cast<double>())
                        .set_tree_count(params["tree_count"].cast<std::int64_t>())
                        .set_features_per_node(params["features_per_node"].cast<std::int64_t>())
                        .set_max_tree_depth(params["max_tree_depth"].cast<std::int64_t>())
                        .set_min_observations_in_leaf_node(
                            params["min_observations_in_leaf_node"].cast<std::int64_t>())
                        .set_min_observations_in_split_node(
                            params["min_observations_in_split_node"].cast<std::int64_t>())
                        .set_max_leaf_nodes(params["max_leaf_nodes"].cast<std::int64_t>())
                        .set_max_bins(params["max_bins"].cast<std::int64_t>())
                        .set_min_bin_size(params["min_bin_size"].cast<std::int64_t>())
                        .set_memory_saving_mode(params["memory_saving_mode"].cast<bool>())
                        .set_bootstrap(params["bootstrap"].cast<bool>())
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230101
                        .set_splitter_mode(get_splitter_mode(params))
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230101
                        .set_error_metric_mode(get_error_metric_mode(params))
                        .set_variable_importance_mode(get_variable_importance_mode(params));

        if constexpr (is_cls) {
            desc.set_class_count(params["class_count"].cast<std::int64_t>());
            desc.set_infer_mode(get_infer_mode(params));
            desc.set_voting_mode(get_voting_mode(params));
        }

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
        desc.set_seed(params["seed"].cast<std::int64_t>());
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20250700
        if (params.contains("local_trees_mode")) {
            desc.set_local_trees_mode(params["local_trees_mode"].cast<bool>())
        }
        else {
            desc.set_local_trees_mode(false);
        }
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20250700

        return desc;
    }
};

template <typename Policy, typename Task>
void init_train_ops(py::module_& m) {
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const table& data,
             const table& responses,
             const table& weights) {
              using namespace decision_forest;
              using input_t = train_input<Task>;

              train_ops ops(policy, input_t{ data, responses, weights }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const table& data,
             const table& responses) {
              using namespace decision_forest;
              using input_t = train_input<Task>;

              train_ops ops(policy, input_t{ data, responses }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300
    using infer_hyperparams_t = decision_forest::detail::infer_parameters<Task>;
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const infer_hyperparams_t& hyperparams,
             const decision_forest::model<Task>& model,
             const table& data) {
              using namespace decision_forest;
              using input_t = infer_input<Task>;

              infer_ops_with_hyperparams ops(policy,
                                             input_t{ model, data },
                                             params2desc{},
                                             hyperparams);
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300

    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const decision_forest::model<Task>& model,
             const table& data) {
              using namespace decision_forest;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ model, data }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_model(py::module_& m) {
    using namespace decision_forest;
    using model_t = model<Task>;

    auto cls = py::class_<model_t>(m, "model")
                   .def(py::init())
                   .def(py::pickle(
                       [](const model_t& m) {
                           return serialize(m);
                       },
                       [](const py::bytes& bytes) {
                           return deserialize<model_t>(bytes);
                       }))
                   .def_property_readonly("tree_count", &model_t::get_tree_count);

    constexpr bool is_classification = std::is_same_v<Task, task::classification>;

    if constexpr (is_classification) {
        auto class_count_getter = &model_t::template get_class_count<>;
        cls.def_property_readonly("class_count", class_count_getter);
    }
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace decision_forest;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(model, result_t)
        .DEF_ONEDAL_PY_PROPERTY(oob_err, result_t)
        .DEF_ONEDAL_PY_PROPERTY(oob_err_per_observation, result_t)
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230101
        .DEF_ONEDAL_PY_PROPERTY(oob_err_accuracy, result_t)
        .DEF_ONEDAL_PY_PROPERTY(oob_err_r2, result_t)
        .DEF_ONEDAL_PY_PROPERTY(oob_err_decision_function, result_t)
        .DEF_ONEDAL_PY_PROPERTY(oob_err_prediction, result_t)
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20230101
        .DEF_ONEDAL_PY_PROPERTY(var_importance, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace decision_forest;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(responses, result_t);

    constexpr bool is_classification = std::is_same_v<Task, task::classification>;

    if constexpr (is_classification) {
        // workaround for gcc which cannot deduce setters directly passed to def_property()
        auto proba_getter = &result_t::template get_probabilities<>;
        auto proba_setter = &result_t::template set_probabilities<>;
        cls.def_property("probabilities", proba_getter, proba_setter);
    }
}

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300
template <typename Task>
void init_infer_hyperparameters(py::module_& m) {
    using namespace dal::decision_forest::detail;
    using infer_hyperparams_t = infer_parameters<Task>;

    auto cls = py::class_<infer_hyperparams_t>(m, "infer_hyperparameters")
                   .def(py::init())
                   .def("set_block_size",
                        [](infer_hyperparams_t& self, std::int64_t block_size) {
                            self.set_block_size(block_size);
                        })
                   .def("get_block_size",
                        [](const infer_hyperparams_t& self) -> std::int64_t {
                            return self.get_block_size();
                        })
                   .def("set_min_trees_for_threading",
                        [](infer_hyperparams_t& self, std::int64_t trees) {
                            self.set_min_trees_for_threading(trees);
                        })
                   .def("get_min_trees_for_threading",
                        [](const infer_hyperparams_t& self) -> std::int64_t {
                            return self.get_min_trees_for_threading();
                        })
                   .def("set_min_number_of_rows_for_vect_seq_compute",
                        [](infer_hyperparams_t& self, std::int64_t rows) {
                            self.set_min_number_of_rows_for_vect_seq_compute(rows);
                        })
                   .def("get_min_number_of_rows_for_vect_seq_compute",
                        [](const infer_hyperparams_t& self) -> std::int64_t {
                            return self.get_min_number_of_rows_for_vect_seq_compute();
                        })
                   .def("set_scale_factor_for_vect_parallel_compute",
                        [](infer_hyperparams_t& self, double factor) {
                            self.set_scale_factor_for_vect_parallel_compute(factor);
                        })
                   .def("get_scale_factor_for_vect_parallel_compute",
                        [](const infer_hyperparams_t& self) -> double {
                            return self.get_scale_factor_for_vect_parallel_compute();
                        });
}
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300

ONEDAL_PY_TYPE2STR(decision_forest::task::classification, "classification");
ONEDAL_PY_TYPE2STR(decision_forest::task::regression, "regression");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_hyperparameters);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300

ONEDAL_PY_INIT_MODULE(ensemble) {
    using namespace decision_forest;
    using namespace dal::detail;

    using task_list = types<task::classification, task::regression>;
    auto sub = m.def_submodule("decision_forest");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_spmd, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300
    ONEDAL_PY_INSTANTIATE(init_infer_hyperparameters, sub, task::classification);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300

#endif // ONEDAL_DATA_PARALLEL_SPMD
}

} // namespace oneapi::dal::python
