/*******************************************************************************
* Copyright Contributors to the oneDAL Project
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
#include "oendal/tests/fake_onedal.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"


namespace py = pybind11;

namespace oneapi::dal::python {

namespace dummy {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "generate", ops, Float, method::generate);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        using namespace dal::dummy;

        // conversion of the params dict to oneDAL params occurs here except
        // for the ``method`` and ``fptype`` parameters.  They are assigned
        // to the descriptor individually here before returning.
        return dummy::descriptor<Float, Method, Task>()
    }
};

template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const table& data,
             const table& responses) {
              using namespace dal::dummy;
              using input_t = train_input<Task>;
              train_ops ops(policy, input_t{ data, responses }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
};

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const table& constant,
             const table& data) {
              using namespace dal::dummy;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ data, constant }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::dummy;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(constant, result_t)
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::dummy;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(data, result_t);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);

} // namespace dummy

ONEDAL_PY_INIT_MODULE(dummy) {
    using namespace dal::detail;
    using namespace linear_model;
    using namespace dal::dummy;

    using task_list = types<task::generate>;
    auto sub = m.def_submodule("dummy");

    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
}

ONEDAL_PY_TYPE2STR(dal::linear_regression::task::regression, "regression");

} // namespace oneapi::dal::python
