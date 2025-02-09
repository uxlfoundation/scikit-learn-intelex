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

#pragma once

#include <pybind11/pybind11.h>

#include "oneapi/dal/algo/newton_cg/common.hpp"

#include "onedal/common.hpp"

namespace oneapi::dal::python {

template <typename Optimizer>
auto get_optimizer_descriptor(const pybind11::dict& params) {
    auto optimizer = Optimizer{};
    optimizer.set_tolerance(params["tol"].cast<double>());
    optimizer.set_max_iteration(params["max_iter"].cast<std::int64_t>());
    return optimizer;
}

template <typename Dense, typename Ops>
struct optimizer_method2t {
    optimizer_method2t(const Dense& dense_method, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const pybind11::dict& params) {
        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, Dense);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <template <typename F, typename M, typename T> typename Desc>
struct optimizer_params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        return get_optimizer_descriptor<Desc<Float, Method, Task>>(params);
    }
};

} // namespace oneapi::dal::python
