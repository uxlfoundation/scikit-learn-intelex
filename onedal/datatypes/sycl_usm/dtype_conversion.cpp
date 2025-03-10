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

#include <unordered_map>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/datatypes/sycl_usm/dtype_conversion.hpp"
#include "onedal/datatypes/dtype_dispatcher.hpp"

namespace oneapi::dal::python::sycl_usm {

using fwd_map_t = std::unordered_map<std::string, dal::data_type>;
using inv_map_t = std::unordered_map<dal::data_type, std::string>;

inline void unknown_type() {
    throw std::runtime_error("Unknown type");
}

// Get the basic type character codes supported on oneDAL backend side
// for a string providing the basic type of the homogeneous array.
template <typename Type>
constexpr inline char type_desc() {
    if constexpr (std::is_integral_v<Type>) {
        if (std::is_unsigned_v<Type>) {
            // Unsigned integer
            return 'u';
        }
        else {
            // Integer
            return 'i';
        }
    }
    else {
        if (std::is_floating_point_v<Type>) {
            // Floating point
            return 'f';
        }
        else {
            unknown_type();
        }
    }
}

template <typename Type>
constexpr inline char type_size() {
    switch (sizeof(Type)) {
        case 1ul: return '1';
        case 2ul: return '2';
        case 4ul: return '4';
        case 8ul: return '8';
        default: unknown_type();
    };
}

// Get a string encoding elemental data type of the array.
template <typename Type>
inline std::string describe(char e = '<') {
    constexpr auto s = type_size<Type>();
    constexpr auto d = type_desc<Type>();
    return std::string{ { e, d, s } };
}

// oneDAL works only on little-endian hardware.
const char end = '<';

template <typename... Types>
inline auto make_fwd_map(const std::tuple<Types...>* const = nullptr) {
    fwd_map_t result(3ul * sizeof...(Types));

    dal::detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            constexpr auto dal_v = detail::make_data_type<type_t>();
            result.emplace(describe<type_t>(end), dal_v);
            result.emplace(describe<type_t>('='), dal_v);
            result.emplace(describe<type_t>('|'), dal_v);
        },
        Types{}...);

    return result;
}

template <typename... Types>
inline auto make_inv_map(const std::tuple<Types...>* const = nullptr) {
    inv_map_t result(sizeof...(Types));

    dal::detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            constexpr auto dal_v = detail::make_data_type<type_t>();
            result.emplace(dal_v, describe<type_t>('|'));
        },
        Types{}...);

    return result;
}

// The map, that provides translation from `__sycl_usm_array_interface__['typestr']`
// a string encoding elemental data type of the array to oneDAL table data type.
static const fwd_map_t& get_fwd_map() {
    constexpr const supported_types_t* types = nullptr;
    static const fwd_map_t body = make_fwd_map(types);
    return body;
}

// The map, that provides translation from oneDAL table data type to
// `__sycl_usm_array_interface__['typestr']` a string encoding elemental data type
// of the array.
static const inv_map_t& get_inv_map() {
    constexpr const supported_types_t* types = nullptr;
    static const inv_map_t body = make_inv_map(types);
    return body;
}

// Convert a string encoding elemental data type of the array to oneDAL homogen table data type.
dal::data_type convert_sua_to_dal_type(std::string dtype) {
    if (get_fwd_map().count(dtype)) {
        return get_fwd_map().at(dtype);
    }
    else {
        throw py::type_error("Found unsupported array type");
    }
}

// Convert oneDAL homogen table data type to a string encoding elemental data type of the array.
std::string convert_dal_to_sua_type(dal::data_type dtype) {
    if (get_inv_map().count(dtype)) {
        return get_inv_map().at(dtype);
    }
    else {
        throw py::type_error("Found unsupported oneDAL type");
    }
}
} // namespace oneapi::dal::python::sycl_usm
