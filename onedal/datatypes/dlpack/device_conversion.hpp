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

#pragma once

#include <optional>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#else // ONEDAL_DATA_PARALLEL
namespace sycl {
    class queue;
} //namespace sycl
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "oneapi/dal/array.hpp"

#include "onedal/interop/dlpack/api/dlpack.h"

namespace py = pybind11;

namespace oneapi::dal::python::interop::dlpack {

DLDevice get_cpu_device();

// Passing by value is done for a reason
// These structures are extremely small
// and will not create any perf overheads
bool is_cpu_device(DLDevice device);
bool is_oneapi_device(DLDevice device);
bool is_unknown_device(DLDevice device);

#ifdef ONEDAL_DATA_PARALLEL

std::optional<DLDevice> convert_from_sycl(sycl::device device);
std::optional<sycl::device> convert_to_sycl(DLDevice device);

#endif // ONEDAL_DATA_PARALLEL

template <typename Type>
inline DLDevice make_device(const dal::array<Type>& arr) {
#ifdef ONEDAL_DATA_PARALLEL
    if (auto queue = arr.get_queue()) {
        auto device = queue.value().get_device();
        return convert_from_sycl(device).value();
    }
#endif // ONEDAL_DATA_PARALLEL
    return get_cpu_device();
}

DLDevice get_device(std::shared_ptr<sycl::queue> ptr);
std::shared_ptr<sycl::queue> get_queue(DLDevice device);

void instantiate_convert_to_policy(py::module& pm);

} // namespace oneapi::dal::python::interop::dlpack
