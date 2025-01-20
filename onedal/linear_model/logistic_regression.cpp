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

#include "onedal/common.hpp"
#include "onedal/version.hpp"
#include <regex>

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001

#include "oneapi/dal/algo/logistic_regression.hpp"
#include "onedal/primitives/optimizers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

namespace linear_model {

namespace logistic_regression {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::logistic_regression;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense_batch", ops, Float, method::dense_batch);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
        ONEDAL_PARAM_DISPATCH_VALUE(method, "sparse", ops, Float, method::sparse);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >=20240700
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Ops>
struct optimizer2t {
    optimizer2t(const Ops& ops) : ops(ops) {}

    template <typename Float, typename Method>
    auto operator()(const py::dict& params) {
        using namespace dal::logistic_regression;

        auto optimizer = params["optimizer"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(optimizer,
                                    "newton-cg",
                                    ops,
                                    Float,
                                    Method,
                                    newton_cg::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(optimizer);
    }

    Ops ops;
};

auto get_onedal_result_options(const py::dict& params) {
    using namespace dal::logistic_regression;

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
            else if (match.str() == "iterations_count") {
                onedal_options = onedal_options | result_options::iterations_count;
            }
#if ONEDAL_VERSION >= 20240300
            else if (match.str() == "inner_iterations_count") {
                onedal_options = onedal_options | result_options::inner_iterations_count;
            }
#endif
            else {
                ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
            }
        }
    }
    catch (std::regex_error&) {
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
    }

    return onedal_options;
}

template <typename Float, typename Method, typename Task, typename Optimizer>
struct descriptor_creator;

template <typename Float, typename Method, typename Optimizer>
struct descriptor_creator<Float,
                          Method,
                          dal::logistic_regression::task::classification,
                          Optimizer> {
    static auto get(bool intercept, double C) {
        return dal::logistic_regression::
            descriptor<Float, Method, dal::logistic_regression::task::classification>(intercept, C);
    }
};

struct params2desc {
    template <typename Float, typename Method, typename Task, typename Optimizer>
    auto operator()(const py::dict& params) {
        using namespace dal::logistic_regression;

        const auto intercept = params["intercept"].cast<bool>();
        const auto C = params["C"].cast<double>();

        auto desc = descriptor_creator<Float, Method, Task, Optimizer>::get(intercept, C)
                        .set_result_options(get_onedal_result_options(params));

        desc.set_optimizer(get_optimizer_descriptor<Optimizer>(params));

        return desc;
    }
};

template <typename Policy, typename Task>
struct init_train_ops_dispatcher {};

template <typename Policy>
struct init_train_ops_dispatcher<Policy, dal::logistic_regression::task::classification> {
    void operator()(py::module_& m) {
        using Task = dal::logistic_regression::task::classification;
        m.def("train",
              [](const Policy& policy,
                 const py::dict& params,
                 const table& data,
                 const table& responses) {
                  using namespace dal::logistic_regression;
                  using input_t = train_input<Task>;

                  train_ops ops(policy, input_t{ data, responses }, params2desc{});
                  return fptype2t{ method2t{ Task{}, optimizer2t{ ops } } }(params);
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
             const dal::logistic_regression::model<Task>& model,
             const table& data) {
              using namespace dal::logistic_regression;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ data, model }, params2desc{});
              return fptype2t{ method2t{ Task{}, optimizer2t{ ops } } }(params);
          });
}

template <typename Task>
void init_model(py::module_& m) {
    using namespace dal::logistic_regression;
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
    using namespace dal::logistic_regression;
    using result_t = train_result<Task>;

    auto cls = py::class_<result_t>(m, "train_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(model, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(intercept, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(coefficients, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(iterations_count, result_t)
#if ONEDAL_VERSION >= 20240300
                   .DEF_ONEDAL_PY_PROPERTY(inner_iterations_count, result_t)
#endif
                   .DEF_ONEDAL_PY_PROPERTY(packed_coefficients, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(result_options, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::logistic_regression;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(responses, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(probabilities, result_t);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);

} // namespace logistic_regression

} // namespace linear_model

ONEDAL_PY_INIT_MODULE(logistic_regression) {
    using namespace dal::detail;
    using namespace linear_model::logistic_regression;
    using namespace dal::logistic_regression;

    using task_list = types<task::classification>;
    auto sub = m.def_submodule("logistic_regression");

#if defined(ONEDAL_DATA_PARALLEL_SPMD)
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240100
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_spmd, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_spmd, task_list);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240100
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

ONEDAL_PY_TYPE2STR(dal::logistic_regression::task::classification, "classification");

} // namespace oneapi::dal::python

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >=20240001
