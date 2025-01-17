/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "onedal/version.hpp"

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/partial_compute.hpp"
#include "oneapi/dal/finalize_compute.hpp"
#include "oneapi/dal/partial_train.hpp"
#include "oneapi/dal/finalize_train.hpp"

#define ONEDAL_PARAM_DISPATCH_VALUE(value, value_case, ops, ...) \
    if (value == value_case) {                                   \
        return ops.template operator()<__VA_ARGS__>(params);     \
    }                                                            \
    else

#define ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(name) \
    { throw std::runtime_error("Invalid value for parameter <" #name ">"); }

namespace oneapi::dal::python {

template <typename Ops>
struct fptype2t {
    fptype2t(const Ops& ops) : ops(ops) {}

    auto operator()(const pybind11::dict& params) {
        // fptype needs to be a numpy dtype, which uses pybind11-native dtype checking
        const auto fptype = params["fptype"].cast<pybind11::dtype>().num();
        ONEDAL_PARAM_DISPATCH_VALUE(fptype, pybind11::detail::npy_api::NPY_FLOAT_, ops, float);
        ONEDAL_PARAM_DISPATCH_VALUE(fptype, pybind11::detail::npy_api::NPY_DOUBLE_, ops, double);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(fptype);
    }

    Ops ops;
};

template <typename Policy, typename Input, typename Ops>
struct compute_ops {
    using Task = typename Input::task_t;

    compute_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::compute(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Policy, typename Input, typename Ops, typename Hyperparams>
struct compute_ops_with_hyperparams {
    using Task = typename Input::task_t;

    compute_ops_with_hyperparams(const Policy& policy,
                                 const Input& input,
                                 const Ops& ops,
                                 const Hyperparams& hyperparams)
            : policy(policy),
              input(input),
              ops(ops),
              hyperparams(hyperparams) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::compute(policy, desc, hyperparams, input);
    }

    Policy policy;
    Input input;
    Ops ops;
    Hyperparams hyperparams;
};

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Policy, typename Input, typename Ops>
struct train_ops {
    using Task = typename Input::task_t;

    train_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::train(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Policy, typename Input, typename Ops, typename Hyperparams>
struct train_ops_with_hyperparams {
    using Task = typename Input::task_t;

    train_ops_with_hyperparams(const Policy& policy,
                               const Input& input,
                               const Ops& ops,
                               const Hyperparams& hyperparams)
            : policy(policy),
              input(input),
              ops(ops),
              hyperparams(hyperparams) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::train(policy, desc, hyperparams, input);
    }

    Policy policy;
    Input input;
    Ops ops;
    Hyperparams hyperparams;
};

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Policy, typename Input, typename Ops>
struct infer_ops {
    using Task = typename Input::task_t;

    infer_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::infer(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300

template <typename Policy, typename Input, typename Ops, typename Hyperparams>
struct infer_ops_with_hyperparams {
    using Task = typename Input::task_t;

    infer_ops_with_hyperparams(const Policy& policy,
                               const Input& input,
                               const Ops& ops,
                               const Hyperparams& hyperparams)
            : policy(policy),
              input(input),
              ops(ops),
              hyperparams(hyperparams) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::infer(policy, desc, hyperparams, input);
    }

    Policy policy;
    Input input;
    Ops ops;
    Hyperparams hyperparams;
};

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240300

template <typename Policy, typename Input, typename Ops>
struct partial_compute_ops {
    using Task = typename Input::task_t;
    partial_compute_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::partial_compute(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

template <typename Policy, typename Input, typename Ops>
struct finalize_compute_ops {
    using Task = typename Input::task_t;
    finalize_compute_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::finalize_compute(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

template <typename Policy, typename Input, typename Ops>
struct partial_train_ops {
    using Task = typename Input::task_t;
    partial_train_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::partial_train(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

template <typename Policy, typename Input, typename Ops, typename Hyperparams>
struct partial_train_ops_with_hyperparams {
    using Task = typename Input::task_t;
    partial_train_ops_with_hyperparams(const Policy& policy,
                                       const Input& input,
                                       const Ops& ops,
                                       const Hyperparams& hyperparams)
            : policy(policy),
              input(input),
              ops(ops),
              hyperparams(hyperparams) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::partial_train(policy, desc, hyperparams, input);
    }

    Policy policy;
    Input input;
    Ops ops;
    Hyperparams hyperparams;
};

template <typename Policy, typename Input, typename Ops>
struct finalize_train_ops {
    using Task = typename Input::task_t;
    finalize_train_ops(const Policy& policy, const Input& input, const Ops& ops)
            : policy(policy),
              input(input),
              ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::finalize_train(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

template <typename Policy, typename Input, typename Ops, typename Hyperparams>
struct finalize_train_ops_with_hyperparams {
    using Task = typename Input::task_t;
    finalize_train_ops_with_hyperparams(const Policy& policy,
                                        const Input& input,
                                        const Ops& ops,
                                        const Hyperparams& hyperparams)
            : policy(policy),
              input(input),
              ops(ops),
              hyperparams(hyperparams) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::finalize_train(policy, desc, hyperparams, input);
    }

    Policy policy;
    Input input;
    Ops ops;
    Hyperparams hyperparams;
};

} // namespace oneapi::dal::python
