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

#include "oneapi/dal/algo/linear_regression.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#define NO_IMPORT_ARRAY // import_array called in table.cpp
#include "onedal/datatypes/data_conversion.hpp"

#include <regex>

namespace py = pybind11;

namespace oneapi::dal::python {

namespace linear_model {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::linear_regression;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "norm_eq", ops, Float, method::norm_eq);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::norm_eq);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

auto get_onedal_result_options(const py::dict& params) {
    using namespace dal::linear_regression;

    auto result_option = params["result_option"].cast<std::string>();
    result_option_id onedal_options;

    try {
        std::regex re("\\w+");
        const std::sregex_iterator last{};
        const std::sregex_iterator first( //
            result_option.begin(),
            result_option.end(),
            re);

        for (std::sregex_iterator it = first; it != last; ++it) {
            std::smatch match = *it;
            if (match.str() == "intercept") {
                onedal_options = onedal_options | result_options::intercept;
            }
            else if (match.str() == "coefficients") {
                onedal_options = onedal_options | result_options::coefficients;
            }
            else {
                ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
            }
        }
    }
    catch (std::regex_error& e) {
        (void)e;
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
    }

    return onedal_options;
}

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        using namespace dal::linear_regression;

        const auto intercept = params["intercept"].cast<bool>();

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240600
        const auto alpha = params["alpha"].cast<double>();
        auto desc = linear_regression::descriptor<Float, Method, Task>(intercept, alpha)
                        .set_result_options(get_onedal_result_options(params));
#else
        auto desc =
            linear_regression::descriptor<Float, Method, Task>(intercept).set_result_options(
                get_onedal_result_options(params));
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240600
        return desc;
    }
};

template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
    using train_hyperparams_t = dal::linear_regression::detail::train_parameters<Task>;
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const train_hyperparams_t& hyperparams,
             const table& data,
             const table& responses) {
              using namespace dal::linear_regression;
              using input_t = train_input<Task>;
              train_ops_with_hyperparams ops(policy,
                                             input_t{ data, responses },
                                             params2desc{},
                                             hyperparams);
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const table& data,
             const table& responses) {
              using namespace dal::linear_regression;
              using input_t = train_input<Task>;
              train_ops ops(policy, input_t{ data, responses }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
};

template <typename Policy, typename Task>
void init_partial_train_ops(py::module& m) {
    using prev_result_t = dal::linear_regression::partial_train_result<Task>;
    using train_hyperparams_t = dal::linear_regression::detail::train_parameters<Task>;
    m.def("partial_train",
          [](const Policy& policy,
             const py::dict& params,
             const train_hyperparams_t& hyperparams,
             const prev_result_t& prev,
             const table& data,
             const table& responses) {
              using namespace dal::linear_regression;
              using input_t = partial_train_input<Task>;
              partial_train_ops_with_hyperparams ops(policy,
                                                     input_t{ prev, data, responses },
                                                     params2desc{},
                                                     hyperparams);
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });

    using prev_result_t = dal::linear_regression::partial_train_result<Task>;
    m.def("partial_train",
          [](const Policy& policy,
             const py::dict& params,
             const prev_result_t& prev,
             const table& data,
             const table& responses) {
              using namespace dal::linear_regression;
              using input_t = partial_train_input<Task>;
              partial_train_ops ops(policy, input_t{ prev, data, responses }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
};

template <typename Policy, typename Task>
void init_finalize_train_ops(py::module& m) {
    using input_t = dal::linear_regression::partial_train_result<Task>;
    using train_hyperparams_t = dal::linear_regression::detail::train_parameters<Task>;

    m.def("finalize_train",
          [](const Policy& policy,
             const py::dict& params,
             const train_hyperparams_t& hyperparams,
             const input_t& data) {
              finalize_train_ops_with_hyperparams ops(policy, data, params2desc{}, hyperparams);
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });

    m.def("finalize_train", [](const Policy& policy, const py::dict& params, const input_t& data) {
        finalize_train_ops ops(policy, data, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
};

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const dal::linear_regression::model<Task>& model,
             const table& data) {
              using namespace dal::linear_regression;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ data, model }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_model(py::module_& m) {
    using namespace dal::linear_regression;
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
                   .DEF_ONEDAL_PY_PROPERTY(packed_coefficients, model_t);
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::linear_regression;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(model, result_t)
        .DEF_ONEDAL_PY_PROPERTY(intercept, result_t)
        .DEF_ONEDAL_PY_PROPERTY(coefficients, result_t)
        .DEF_ONEDAL_PY_PROPERTY(packed_coefficients, result_t)
        .DEF_ONEDAL_PY_PROPERTY(result_options, result_t);
}

template <typename Task>
void init_partial_train_result(py::module_& m) {
    using namespace dal::linear_regression;
    using result_t = partial_train_result<Task>;
    py::class_<result_t>(m, "partial_train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(partial_xtx, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_xty, result_t)
        .def(py::pickle(
            [](const result_t& res) {
                return py::make_tuple(
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_xtx())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_xty())));
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                result_t res;
                if (py::cast<int>(t[0].attr("size")) != 0)
                    res.set_partial_xtx(convert_to_table(t[0]));
                if (py::cast<int>(t[1].attr("size")) != 0)
                    res.set_partial_xty(convert_to_table(t[1]));
                return res;
            }));
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::linear_regression;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(responses, result_t);
}

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Task>
void init_train_hyperparameters(py::module_& m) {
    using namespace dal::linear_regression::detail;
    using train_hyperparams_t = train_parameters<Task>;

    auto cls = py::class_<train_hyperparams_t>(m, "train_hyperparameters")
                   .def(py::init())
                   .def("set_cpu_macro_block",
                        [](train_hyperparams_t& self, int64_t cpu_macro_block) {
                            self.set_cpu_macro_block(cpu_macro_block);
                        })
                   .def("set_gpu_macro_block",
                        [](train_hyperparams_t& self, int64_t gpu_macro_block) {
                            self.set_gpu_macro_block(gpu_macro_block);
                        })
                   .def("get_cpu_macro_block",
                        [](const train_hyperparams_t& self) {
                            return self.get_cpu_macro_block();
                        })
                   .def("get_gpu_macro_block", [](const train_hyperparams_t& self) {
                       return self.get_gpu_macro_block();
                   });
}

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_finalize_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_hyperparameters);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

} // namespace linear_model

ONEDAL_PY_INIT_MODULE(linear_model) {
    using namespace dal::detail;
    using namespace linear_model;
    using namespace dal::linear_regression;

    using task_list = types<task::regression>;
    auto sub = m.def_submodule("linear_model");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_spmd, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_spmd, task_list);
    ONEDAL_PY_INSTANTIATE(init_finalize_train_ops, sub, policy_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_partial_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_finalize_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_partial_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
    ONEDAL_PY_INSTANTIATE(init_train_hyperparameters, sub, task_list);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

ONEDAL_PY_TYPE2STR(dal::linear_regression::task::regression, "regression");

} // namespace oneapi::dal::python
