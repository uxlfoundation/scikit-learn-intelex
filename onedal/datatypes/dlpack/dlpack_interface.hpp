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

#include <array>
#include <cstdint>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#else // ONEDAL_DATA_PARALLEL
namespace sycl {
    class queue;
} // namespace sycl
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "onedal/interop/dlpack/api/dlpack.h"
#include "onedal/interop/dlpack/dlpack_utils.hpp"
#include "onedal/interop/dlpack/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::dlpack {

using tensor_t = const DLTensor;
using managed_t = const DLManagedTensor;

template <std::int64_t dim>
struct dlpack_interface {
    dal::data_type dtype;
    std::uint64_t offset;
    std::shared_ptr<sycl::queue> queue;
    std::array<std::int64_t, dim> shape;
    std::array<std::int64_t, dim> strides;
    std::pair<std::uintptr_t, bool> data;
};

template <std::int64_t dim>
DLTensor produce_unmanaged(std::shared_ptr<dlpack_interface<dim>> ptr);

template <std::int64_t dim>
std::shared_ptr<dlpack_interface<dim>> get_dlpack_interface(py::capsule caps); 

} // namespace oneapi::dal::python::interop::dlpack