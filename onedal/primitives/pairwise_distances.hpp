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

#pragma once

#include <pybind11/pybind11.h>

#include "oneapi/dal/algo/chebyshev_distance/common.hpp"
#include "oneapi/dal/algo/correlation_distance/common.hpp"
#include "oneapi/dal/algo/cosine_distance/common.hpp"
#include "oneapi/dal/algo/minkowski_distance/common.hpp"

#include "onedal/common.hpp"

namespace oneapi::dal::python {

template <typename Distance>
auto get_distance_descriptor(const pybind11::dict& params) {
    using float_t = typename Distance::float_t;
    using method_t = typename Distance::method_t;
    using task_t = typename Distance::task_t;
    using minkowski_desc_t = minkowski_distance::descriptor<float_t, method_t, task_t>;
    using correlation_distance_desc_t = correlation_distance::descriptor<float_t, method_t, task_t>;
    using cosine_distance_desc_t = cosine_distance::descriptor<float_t, method_t, task_t>;

    auto distance = Distance{};
    if constexpr (std::is_same_v<Distance, minkowski_desc_t>) {
        distance.set_degree(params["p"].cast<double>());
    }

    return distance;
}

template <typename Dense, typename Ops>
struct distance_method2t {
    distance_method2t(const Dense& dense_method, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const pybind11::dict& params) {
        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, Dense);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <template <typename F, typename M, typename T> typename Desc>
struct distance_params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        return get_distance_descriptor<Desc<Float, Method, Task>>(params);
    }
};

} // namespace oneapi::dal::python
