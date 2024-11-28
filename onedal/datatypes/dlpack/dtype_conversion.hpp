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

#include <string>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/dlpack/dlpack.h"

namespace py = pybind11;

namespace oneapi::dal::python::interop::dlpack {

// DLDataType is only 64 bits in size. Not expensive
dal::data_type convert_dlpack_to_dal_type(DLDataType dt);
DLDataType convert_dal_to_dlpack_type(dal::data_type dt);

template <typename Type>
inline DLDataType make_data_type(Type tag = {}) {
    constexpr auto dt = detail::make_data_type<Type>();
    return convert_dlpack_to_dal_type(dt);
}

} // namespace oneapi::dal::python::interop::dlpack
