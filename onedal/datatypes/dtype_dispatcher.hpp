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

// TODO: Using includes should be the primary path
#if defined(ONEDAL_VERSION) && (20240400 < ONEDAL_VERSION)

#include "oneapi/dal/detail/dtype_dispatcher.hpp"

#else // Version check

#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::detail {

template <typename Op, typename OnUnknown>
inline constexpr auto dispatch_by_data_type(data_type dtype, Op&& op, OnUnknown&& on_unknown) {
    switch (dtype) {
        case data_type::int8: return op(std::int8_t{});
        case data_type::uint8: return op(std::uint8_t{});
        case data_type::int16: return op(std::int16_t{});
        case data_type::uint16: return op(std::uint16_t{});
        case data_type::int32: return op(std::int32_t{});
        case data_type::uint32: return op(std::uint32_t{});
        case data_type::int64: return op(std::int64_t{});
        case data_type::uint64: return op(std::uint64_t{});
        case data_type::float32: return op(float{});
        case data_type::float64: return op(double{});
        default: return on_unknown(dtype);
    }
}

template <typename Op, typename ResultType = std::invoke_result_t<Op, float>>
inline constexpr ResultType dispatch_by_data_type(data_type dtype, Op&& op) {
    // Necessary to make the return type conformant with
    // other dispatch branches
    const auto on_unknown = [](data_type) -> ResultType {
        using msg = oneapi::dal::detail::error_messages;
        throw unimplemented{ msg::unsupported_conversion_types() };
    };

    return dispatch_by_data_type(dtype, std::forward<Op>(op), on_unknown);
}

} // namespace oneapi::dal::detail

#endif // Version check

// TODO: Using includes should be the primary path
#if defined(ONEDAL_VERSION) && (ONEDAL_VERSION < 20240000)

namespace oneapi::dal::detail {

template <typename... Types, typename Op>
constexpr inline void apply(Op&& op) {
    ((void)op(Types{}), ...);
}

template <typename Op, typename... Args>
constexpr inline void apply(Op&& op, Args&&... args) {
    ((void)op(std::forward<Args>(args)), ...);
}

} //namespace oneapi::dal::detail

#endif // Version check

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
