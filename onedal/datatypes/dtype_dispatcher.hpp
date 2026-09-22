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
#include <cstdint>

#include "onedal/common.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/detail/dtype_dispatcher.hpp"

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

namespace oneapi::dal::python {

using supported_types_t = std::tuple<float,
                                     double,
                                     std::int8_t,
                                     std::uint8_t,
                                     std::int16_t,
                                     std::uint16_t,
                                     std::int32_t,
                                     std::uint32_t,
                                     std::int64_t,
                                     std::uint64_t>;
} // namespace oneapi::dal::python
