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

#include "oneapi/dal/algo/basic_statistics.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#define NO_IMPORT_ARRAY // import_array called in table.cpp
#include "onedal/datatypes/data_conversion.hpp"

#include <string>
#include <regex>
#include <map>

namespace py = pybind11;

namespace oneapi::dal::python {

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100

namespace basic_statistics {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::basic_statistics;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "sparse", ops, Float, method::sparse);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

auto get_onedal_result_options(const py::dict& params) {
    using namespace dal::basic_statistics;

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
            if (match.str() == "max") {
                onedal_options = onedal_options | result_options::max;
            }
            else if (match.str() == "min") {
                onedal_options = onedal_options | result_options::min;
            }
            else if (match.str() == "sum") {
                onedal_options = onedal_options | result_options::sum;
            }
            else if (match.str() == "mean") {
                onedal_options = onedal_options | result_options::mean;
            }
            else if (match.str() == "variance") {
                onedal_options = onedal_options | result_options::variance;
            }
            else if (match.str() == "variation") {
                onedal_options = onedal_options | result_options::variation;
            }
            else if (match.str() == "sum_squares") {
                onedal_options = onedal_options | result_options::sum_squares;
            }
            else if (match.str() == "standard_deviation") {
                onedal_options = onedal_options | result_options::standard_deviation;
            }
            else if (match.str() == "sum_squares_centered") {
                onedal_options = onedal_options | result_options::sum_squares_centered;
            }
            else if (match.str() == "second_order_raw_moment") {
                onedal_options = onedal_options | result_options::second_order_raw_moment;
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
        auto desc =
            dal::basic_statistics::descriptor<Float, Method, dal::basic_statistics::task::compute>()
                .set_result_options(get_onedal_result_options(params));
        return desc;
    }
};

/// Only dense method is supported by incremental basic statistics
struct params2desc_incremental {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        auto desc = dal::basic_statistics::descriptor<Float,
                                                      dal::basic_statistics::method::dense,
                                                      dal::basic_statistics::task::compute>()
                        .set_result_options(get_onedal_result_options(params));
        return desc;
    }
};

template <typename Policy, typename Task>
void init_compute_ops(py::module& m) {
    m.def(
        "compute",
        [](const Policy& policy, const py::dict& params, const table& data, const table& weights) {
            using namespace dal::basic_statistics;
            using input_t = compute_input<Task>;

            compute_ops ops(policy, input_t{ data, weights }, params2desc{});
            return fptype2t{ method2t{ Task{}, ops } }(params);
        });
}

template <typename Policy, typename Task>
void init_partial_compute_ops(py::module& m) {
    using prev_result_t = dal::basic_statistics::partial_compute_result<Task>;
    m.def("partial_compute",
          [](const Policy& policy,
             const py::dict& params,
             const prev_result_t& prev,
             const table& data,
             const table& weights) {
              using namespace dal::basic_statistics;
              using input_t = partial_compute_input<Task>;
              partial_compute_ops ops(policy,
                                      input_t{ prev, data, weights },
                                      params2desc_incremental{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Policy, typename Task>
void init_finalize_compute_ops(pybind11::module_& m) {
    using namespace dal::basic_statistics;
    using input_t = partial_compute_result<Task>;
    m.def("finalize_compute",
          [](const Policy& policy, const pybind11::dict& params, const input_t& data) {
              finalize_compute_ops ops(policy, data, params2desc_incremental{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace dal::basic_statistics;
    using result_t = compute_result<Task>;

    py::class_<result_t>(m, "compute_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(min, result_t)
        .DEF_ONEDAL_PY_PROPERTY(max, result_t)
        .DEF_ONEDAL_PY_PROPERTY(sum, result_t)
        .DEF_ONEDAL_PY_PROPERTY(mean, result_t)
        .DEF_ONEDAL_PY_PROPERTY(variance, result_t)
        .DEF_ONEDAL_PY_PROPERTY(variation, result_t)
        .DEF_ONEDAL_PY_PROPERTY(sum_squares, result_t)
        .DEF_ONEDAL_PY_PROPERTY(standard_deviation, result_t)
        .DEF_ONEDAL_PY_PROPERTY(sum_squares_centered, result_t)
        .DEF_ONEDAL_PY_PROPERTY(second_order_raw_moment, result_t);
}

template <typename Task>
void init_partial_compute_result(py::module_& m) {
    using namespace dal::basic_statistics;
    using result_t = partial_compute_result<Task>;

    py::class_<result_t>(m, "partial_compute_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(partial_n_rows, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_min, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_max, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum_squares, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum_squares_centered, result_t)
        .def(py::pickle(
            [](const result_t& res) {
                return py::make_tuple(
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_n_rows())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_min())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_max())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_sum())),
                    py::cast<py::object>(convert_to_pyobject(res.get_partial_sum_squares())),
                    py::cast<py::object>(
                        convert_to_pyobject(res.get_partial_sum_squares_centered())));
            },
            [](py::tuple t) {
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");
                result_t res;
                if (py::cast<int>(t[0].attr("size")) != 0)
                    res.set_partial_n_rows(convert_to_table(t[0]));
                if (py::cast<int>(t[1].attr("size")) != 0)
                    res.set_partial_min(convert_to_table(t[1]));
                if (py::cast<int>(t[2].attr("size")) != 0)
                    res.set_partial_max(convert_to_table(t[2]));
                if (py::cast<int>(t[3].attr("size")) != 0)
                    res.set_partial_sum(convert_to_table(t[3]));
                if (py::cast<int>(t[4].attr("size")) != 0)
                    res.set_partial_sum_squares(convert_to_table(t[4]));
                if (py::cast<int>(t[5].attr("size")) != 0)
                    res.set_partial_sum_squares_centered(convert_to_table(t[5]));

                return res;
            }));
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_finalize_compute_ops);

} // namespace basic_statistics

ONEDAL_PY_INIT_MODULE(basic_statistics) {
    using namespace dal::detail;
    using namespace basic_statistics;
    using namespace dal::basic_statistics;

    auto sub = m.def_submodule("basic_statistics");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_spmd, task::compute);
    ONEDAL_PY_INSTANTIATE(init_finalize_compute_ops, sub, policy_spmd, task::compute);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task::compute);
    ONEDAL_PY_INSTANTIATE(init_partial_compute_ops, sub, policy_list, task::compute);
    ONEDAL_PY_INSTANTIATE(init_finalize_compute_ops, sub, policy_list, task::compute);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task::compute);
    ONEDAL_PY_INSTANTIATE(init_partial_compute_result, sub, task::compute);
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100

} // namespace oneapi::dal::python
