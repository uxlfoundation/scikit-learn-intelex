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

#define PY_ARRAY_UNIQUE_SYMBOL ONEDAL_PY_ARRAY_API

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::python::numpy {

namespace py = pybind11;

PyObject *convert_to_pyobject(const dal::table &input);
dal::table convert_to_table(py::object inp_obj, py::object queue = py::none());

} // namespace oneapi::dal::python::numpy
