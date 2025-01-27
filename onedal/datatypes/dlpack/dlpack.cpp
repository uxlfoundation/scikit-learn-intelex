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

#include "onedal/interop/dlpack/dlpack.hpp"
#include "onedal/interop/dlpack/dlpack_helper.hpp"
#include "onedal/interop/dlpack/dlpack_and_array.hpp"
#include "onedal/interop/dlpack/dlpack_and_table.hpp"
#include "onedal/interop/dlpack/device_conversion.hpp"

namespace oneapi::dal::python::interop {

void instantiate_dlpack_interop(py::module& pm) {
    auto sub_module = pm.def_submodule("dlpack");
    dlpack::instantiate_dlpack_helper(sub_module);
    dlpack::instantiate_dlpack_and_array(sub_module);
    dlpack::instantiate_dlpack_and_table(sub_module);
    dlpack::instantiate_convert_to_policy(sub_module);
}

} // namespace oneapi::dal::python::interop