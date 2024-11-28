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

namespace py = pybind11;

#define SET_CTYPE_FROM_DAL_TYPE(_T, _FUNCT, _EXCEPTION) \
    switch (_T) {                                       \
        case dal::data_type::float32: {                 \
            _FUNCT(float);                              \
            break;                                      \
        }                                               \
        case dal::data_type::float64: {                 \
            _FUNCT(double);                             \
            break;                                      \
        }                                               \
        case dal::data_type::int32: {                   \
            _FUNCT(std::int32_t);                       \
            break;                                      \
        }                                               \
        case dal::data_type::int64: {                   \
            _FUNCT(std::int64_t);                       \
            break;                                      \
        }                                               \
        default: _EXCEPTION;                            \
    };

namespace oneapi::dal::python::sycl_usm {

dal::data_type convert_sua_to_dal_type(std::string dtype);
std::string convert_dal_to_sua_type(dal::data_type dtype);

} // namespace oneapi::dal::python::sycl_usm
