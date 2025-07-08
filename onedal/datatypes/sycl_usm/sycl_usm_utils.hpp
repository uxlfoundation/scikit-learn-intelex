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

#ifdef ONEDAL_DATA_PARALLEL
#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <string>
#include <utility>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/common/sycl_interfaces.hpp"
#include "onedal/datatypes/sycl_usm/data_conversion.hpp"
#include "onedal/datatypes/sycl_usm/dtype_conversion.hpp"

namespace oneapi::dal::python::sycl_usm {

dal::data_type get_sua_dtype(const py::dict& sua);

py::dict get_sua_interface(const py::object& obj);

py::tuple get_sua_data(const py::dict& sua);

std::uintptr_t get_sua_ptr(const py::dict& sua);

bool is_sua_readonly(const py::dict& sua);

py::tuple get_sua_shape(const py::dict& sua);

void report_problem_for_sua_iface(const char* clarification);

std::int64_t get_and_check_sua_iface_ndim(const py::dict& sua_dict);

std::pair<std::int64_t, std::int64_t> get_sua_iface_shape_by_values(const py::dict sua_dict,
                                                                    const std::int64_t ndim);

dal::data_layout get_sua_iface_layout(const py::dict& sua_dict,
                                      const std::int64_t& r_count,
                                      const std::int64_t& c_count);

void report_problem_to_sua_iface(const char* clarification);

py::tuple get_npy_strides(const dal::data_layout& data_layout,
                          npy_intp row_count,
                          npy_intp column_count);

} // namespace oneapi::dal::python::sycl_usm

#endif // ONEDAL_DATA_PARALLEL
