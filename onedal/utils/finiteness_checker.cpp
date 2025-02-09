/*******************************************************************************
* Copyright 2024 Intel Corporation
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

// fix error with missing headers
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20250200
#include "oneapi/dal/algo/finiteness_checker.hpp"
#else
#include "oneapi/dal/algo/finiteness_checker/compute.hpp"
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20250200

#include "onedal/common.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace finiteness_checker;

        const auto method = params["method"].cast<std::string>();

        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::finiteness_checker;

        auto desc = descriptor<Float, Method, Task>();
        desc.set_allow_NaN(params["allow_nan"].cast<bool>());
        return desc;
    }
};

template <typename Policy, typename Task>
void init_compute_ops(py::module_& m) {
    m.def("compute", [](const Policy& policy, const py::dict& params, const table& data) {
        using namespace finiteness_checker;
        using input_t = compute_input<Task>;

        compute_ops ops(policy, input_t{ data }, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace finiteness_checker;
    using result_t = compute_result<Task>;

    py::class_<result_t>(m, "compute_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(finite, result_t);
}

ONEDAL_PY_TYPE2STR(finiteness_checker::task::compute, "compute");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);

ONEDAL_PY_INIT_MODULE(finiteness_checker) {
    using namespace dal::detail;
    using namespace finiteness_checker;
    using namespace dal::finiteness_checker;

    using task_list = types<task::compute>;
    auto sub = m.def_submodule("finiteness_checker");

#ifndef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task_list);
#endif
}

} // namespace oneapi::dal::python
