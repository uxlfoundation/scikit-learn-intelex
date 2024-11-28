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

#include <pybind11/pybind11.h>

#include "onedal/datatypes/dlpack/dlpack.h"
#include "onedal/datatypes/dlpack/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::dlpack {

using tensor_t = const DLTensor;
using managed_t = const DLManagedTensor;

bool check_dlpack(const py::capsule& caps);
void assert_dlpack(const py::capsule& caps);

void* get_raw_ptr(const py::capsule& caps);
dal::data_type get_dtype(const py::capsule& caps);
std::int64_t get_dim_count(const py::capsule& caps);
std::int64_t get_count_by_dim(std::int64_t dim, const py::capsule& caps);
std::int64_t get_stride_by_dim(std::int64_t dim, const py::capsule& caps);

void* get_raw_ptr(tensor_t& caps);
dal::data_type get_dtype(tensor_t& caps);
std::int64_t get_dim_count(tensor_t& caps);
std::int64_t get_count_by_dim(std::int64_t dim, tensor_t& caps);
std::int64_t get_stride_by_dim(std::int64_t dim, tensor_t& caps);

void delete_dlpack(const py::capsule& caps);

void assert_pointer(dal::data_type dt, void* ptr, const py::capsule& caps);

template <typename Type>
inline void assert_pointer(Type* ptr, const py::capsule& caps) {
    constexpr auto dt = detail::make_data_type<Type>();
    auto* raw = reinterpret_cast<void*>(ptr);
    return assert_pointer(dt, raw, caps);
}

template <typename Type>
inline void assert_pointer(const Type* ptr, const py::capsule& caps) {
    return assert_pointer<Type>(const_cast<Type*>(ptr), caps);
}

template <typename Type>
inline Type* get_ptr(tensor_t& tensor) {
    void* raw = get_raw_ptr(tensor);
    return reinterpret_cast<Type*>(raw);
}

template <typename Type>
inline Type* get_ptr(const py::capsule& caps) {
    void* raw = get_raw_ptr(caps);
    Type* ptr = reinterpret_cast<Type*>(raw);
    assert_pointer<Type>(ptr, caps);
    return ptr;
}

} // namespace oneapi::dal::python::dlpack