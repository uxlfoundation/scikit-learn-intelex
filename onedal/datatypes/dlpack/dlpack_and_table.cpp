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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/interop/utils/tensor_and_table.hpp"

#include "onedal/interop/dlpack/api/dlpack.h"
#include "onedal/interop/dlpack/dlpack_utils.hpp"
#include "onedal/interop/dlpack/dtype_conversion.hpp"
#include "onedal/interop/dlpack/dlpack_interface.hpp"
#include "onedal/interop/dlpack/device_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::dlpack {

py::object wrap_to_homogen_table(py::capsule buffer) {
    auto tensor = get_dlpack_interface<2l>(buffer);
    auto deleter = [tensor](auto* ptr) -> void {
        return;
    };
    return utils::wrap_to_homogen_table(*tensor, std::move(deleter));
}

void instantiate_wrap_to_homogen_table(py::module& pm) {
    constexpr const char name[] = "wrap_to_homogen_table";
    pm.def(name, [](py::capsule capsule) -> py::object {
        return wrap_to_homogen_table(std::move(capsule));
    });
    pm.def(name, [](py::object object) -> py::object {
        auto dlpack = object.attr("__dlpack__");
        auto capsule = dlpack().cast<py::capsule>();
        return wrap_to_homogen_table(std::move(capsule));
    });
}

void instantiate_dlpack_and_table(py::module& pm) {
    instantiate_wrap_to_homogen_table(pm);
}

} // namespace oneapi::dal::python::interop::dlpack
