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
#include "oneapi/dal/algo/pca.hpp"
#include "onedal/common.hpp"
#define NO_IMPORT_ARRAY // import_array called in table.cpp
#include "onedal/datatypes/data_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {
namespace decomposition {
struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        const auto n_components = params["n_components"].cast<std::int64_t>();
        bool whiten = params["whiten"].cast<bool>();
        // sign-flip feature is always used in scikit-learn
        bool is_deterministic = params["is_deterministic"].cast<bool>();

        auto desc = pca::descriptor<Float, Method>()
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240100
                        .set_whiten(whiten)
                        .set_normalization_mode(dal::pca::normalization::mean_center)
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240100
                        .set_component_count(n_components)
                        .set_deterministic(is_deterministic);
        return desc;
    }
};

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::pca;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "cov", ops, Float, method::cov);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "svd", ops, Float, method::svd);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "precomputed", ops, Float, method::precomputed);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Task, typename Ops>
struct incrementalmethod2t {
    incrementalmethod2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::pca;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "cov", ops, Float, method::cov);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Task>
void init_model(py::module_& m) {
    using namespace dal::pca;
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
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240100
                   .DEF_ONEDAL_PY_PROPERTY(eigenvalues, model_t)
                   .DEF_ONEDAL_PY_PROPERTY(means, model_t)
                   .DEF_ONEDAL_PY_PROPERTY(variances, model_t)
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240100
                   .DEF_ONEDAL_PY_PROPERTY(eigenvectors, model_t);
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::pca;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(model, result_t)
        .def_property_readonly("eigenvectors", &result_t::get_eigenvectors)
        .def_property_readonly("eigenvalues", &result_t::get_eigenvalues)
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240100
        .def_property_readonly("singular_values", &result_t::get_singular_values)
        .def_property_readonly("explained_variances_ratio",
                               &result_t::get_explained_variances_ratio)
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240100
        .def_property_readonly("means", &result_t::get_means)
        .def_property_readonly("variances", &result_t::get_variances);
}

template <typename Task>
void init_partial_train_result(py::module_& m) {
    using namespace dal::pca;
    using result_t = partial_train_result<Task>;

    py::class_<result_t>(m, "partial_train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(partial_n_rows, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_crossproduct, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum, result_t)
        .DEF_ONEDAL_PY_PROPERTY(auxiliary_table, result_t)
        .def_property_readonly("auxiliary_table_count", &result_t::get_auxiliary_table_count)
        .def(py::pickle(
            [](const result_t& res) {
                py::list auxiliary;
                int auxiliary_size = res.get_auxiliary_table_count();
                for (int i = 0; i < auxiliary_size; i++) {
                    auto aux_table = res.get_auxiliary_table(i);
                    auxiliary.append(py::cast<py::object>(convert_to_pyobject(aux_table)));
                }
                return py::make_tuple(
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_n_rows())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_crossproduct())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_sum())),
                    auxiliary);
            },
            [](py::tuple t) {
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");
                result_t res;
                if (py::cast<int>(t[0].attr("size")) != 0)
                    res.set_partial_n_rows(convert_to_table(t[0]));
                if (py::cast<int>(t[1].attr("size")) != 0)
                    res.set_partial_crossproduct(convert_to_table(t[1]));
                if (py::cast<int>(t[2].attr("size")) != 0)
                    res.set_partial_sum(convert_to_table(t[2]));
                py::list aux_list = t[3].cast<py::list>();
                for (int i = 0; i < aux_list.size(); i++) {
                    res.set_auxiliary_table(convert_to_table(aux_list[i]));
                }
                return res;
            }));
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::pca;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(transformed_data, result_t);
}

template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
    m.def("train", [](const Policy& policy, const py::dict& params, const table& data) {
        using namespace dal::pca;
        using input_t = train_input<Task>;

        train_ops ops(policy, input_t{ data }, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
}

template <typename Policy, typename Task>
void init_partial_train_ops(py::module& m) {
    using prev_result_t = dal::pca::partial_train_result<Task>;
    m.def("partial_train",
          [](const Policy& policy,
             const py::dict& params,
             const prev_result_t& prev,
             const table& data) {
              using namespace dal::pca;
              using input_t = partial_train_input<Task>;
              partial_train_ops ops(policy, input_t{ prev, data }, params2desc{});
              return fptype2t{ incrementalmethod2t{ Task{}, ops } }(params);
          });
};

template <typename Policy, typename Task>
void init_finalize_train_ops(py::module& m) {
    using input_t = dal::pca::partial_train_result<Task>;
    m.def("finalize_train", [](const Policy& policy, const py::dict& params, const input_t& data) {
        finalize_train_ops ops(policy, data, params2desc{});
        return fptype2t{ incrementalmethod2t{ Task{}, ops } }(params);
    });
};

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const pca::model<Task>& model,
             const table& data) {
              using namespace dal::pca;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ model, data }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_finalize_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);
} // namespace decomposition

ONEDAL_PY_INIT_MODULE(decomposition) {
    using namespace decomposition;
    using namespace dal::pca;
    using namespace dal::detail;

    using task_list = types<task::dim_reduction>;
    auto sub = m.def_submodule("decomposition");
#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_spmd, task_list);
    ONEDAL_PY_INSTANTIATE(init_finalize_train_ops, sub, policy_spmd, task_list);
#else
    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_partial_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_partial_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_finalize_train_ops, sub, policy_list, task_list);
#endif
}

ONEDAL_PY_TYPE2STR(dal::pca::task::dim_reduction, "dim_reduction");
} //namespace oneapi::dal::python
