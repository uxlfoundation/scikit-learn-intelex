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

#include <pybind11/pybind11.h>

#include "onedal/interop/utils/common.hpp"

#include "onedal/interop/dlpack/dlpack_utils.hpp"
#include "onedal/interop/dlpack/dlpack_helper.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::dlpack {

void instantiate_dlpack_helper(py::module& pm) {
    pm.def("get_shape", [](const py::capsule& caps) -> py::tuple {
        const std::int64_t dim_count = get_dim_count(caps);

        py::tuple result(dim_count);

        for (std::int64_t i = 0l; i != dim_count; ++i) {
            result[i] = py::cast(get_count_by_dim(i, caps));
        }

        return result;
    });
}

} // namespace oneapi::dal::python::interop::dlpack
