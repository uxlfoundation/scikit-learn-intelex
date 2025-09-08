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
#include "onedal/tests/dummy_onedal.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

namespace dummy {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}
    // this functor converts the method param into a valid oneDAL task.
    // Tasks are specific to each algorithm, therefore method2t is often
    // defined for each algo.
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
    // This functor converts the params dictionary into a oneDAL descriptor
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        auto desc = dal::dummy::descriptor<Float, Method, Task>();

        // conversion of the params dict to oneDAL params occurs here except
        // for the ``method`` and ``fptype`` parameters.  They are assigned
        // to the descriptor individually here before returning.
        const auto constant = params["constant"].cast<double>();
        desc.set_constant(constant);

        return desc;
    }
};

// the following functions define the python interface methods for the
// oneDAL algorithms. They are templated for the policy (which may be host,
// dpc, or spmd), and task, which is defined per algorithm.  They are all
// defined using lambda functions (a common occurrence for pybind11), but
// that is not a requirement.
template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
    m.def("train", [](const Policy& policy, const py::dict& params, const table& data) {
        using namespace dal::dummy;
        using input_t = train_input<Task>;
        // while there is a train_ops defined for each oneDAL algorithm
        // which supports ``train``, this is the train_ops defined in
        // onedal/common/dispatch_utils.hpp
        train_ops ops(policy, input_t{ data }, params2desc{});
        // fptype2t is defined in common/dispatch_utils.hpp
        // which operates in a similar manner to the method2t functor
        // it selects the floating point datatype for the calculation
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
};

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def(
        "infer",
        [](const Policy& policy, const py::dict& params, const table& constant, const table& data) {
            using namespace dal::dummy;
            using input_t = infer_input<Task>;

            infer_ops ops(policy, input_t{ data, constant }, params2desc{});
            // with the use of functors the order of operations is as
            // follows: Task is generated, the ops is already created above,
            // method2t is constructed, and then fptype2t is constructed.
            // It is then evaluated in opposite order sequentially on the
            // params dict.
            return fptype2t{ method2t{ Task{}, ops } }(params);
        });
}

// This defines the result C++ objects for use in python via pybind11.
// Return types should be pybind11 native types or oneDAL tables.

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::dummy;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(constant, result_t);
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
    using namespace dal::dummy;

    using task_list = types<task::generate>;
    auto sub = m.def_submodule("dummy");

    // explicitly define the templates based off of the policy and task
    // lists. These instantiations lead to a cascade of fully-resolved
    // templates from oneDAL.  It begins by fully resolving functors defined
    // here and the oneDAL descriptor. It then fully specifies functors in
    // common/dispatch_utils.hpp, which starts resolving oneDAL objects
    // for the algorithm like the train_ops/infer_ops functors defined there.
    // This leads to a fair number of compile time work with oneDAL headers.
    // For example take init_train_ops in approximate reverse order
    // (to show how it goes from here to oneDAL):
    //
    // 0. Creates pybind11 interface
    // 1. Specifies lambda defined in init_train_ops
    // 2. Specifies fptype2t
    // 3. Specifies method2t
    // 4. Specifies train_ops defined in common/dispatch_utils.hpp
    // 5. Specifies train defined in oneapi/dal/train.hpp
    // 6. Specifies train_dispatch in oneapi/dal/detail/train_ops.hpp
    // 7. Specifies several functors in oneapi/dal/detail/ops_dispatcher.hpp
    // 8. Specifies train_ops defined in algorithm's train_ops.hpp
    // 9. Specifies oneDAL train_input, train_result and descriptor structs
    /**** finally hits objects compiled in oneDAL for the computation ****/
    // (train_ops_dispatcher for example)
    //
    // Its not clear how many layers of these indirections are compiled
    // versus optimized away. The namings in dispatch_utils.hpp are also
    // unfortunate and confusing.

    //
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
}

ONEDAL_PY_TYPE2STR(dal::dummy::task::generate, "generate");

} // namespace oneapi::dal::python
