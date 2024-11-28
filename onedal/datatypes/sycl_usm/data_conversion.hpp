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

#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL ONEDAL_PY_ARRAY_API

#include <pybind11/pybind11.h>
#include <numpy/arrayobject.h>

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::python::sycl_usm {

namespace py = pybind11;

// Convert oneDAL table with zero-copy by use of `__sycl_usm_array_interface__` protocol.
dal::table convert_to_table(py::object obj);

// Create a dictionary for `__sycl_usm_array_interface__` protocol from oneDAL table properties.
py::dict construct_sua_iface(const dal::table& input);

// Adding `__sycl_usm_array_interface__` attribute to python oneDAL table, that representing
// USM allocations.
void define_sycl_usm_array_property(py::class_<dal::table>& t);

} // namespace oneapi::dal::python::sycl_usm
