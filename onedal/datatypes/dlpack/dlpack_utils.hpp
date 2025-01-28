/*******************************************************************************
* Copyright Contributors to the oneDAL project
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

#include "onedal/datatypes/dlpack/dlpack.h"
#include "onedal/datatypes/dlpack/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::dlpack {

void dlpack_take_ownership(const py::capsule& caps);
std::int64_t get_ndim(tensor_t& caps);
dal::data_layout get_dlpack_layout(const DLTensor& tensor,
                                   const std::int64_t& r_count,
                                   const std::int64_t& c_count)
} // namespace oneapi::dal::python::dlpack
