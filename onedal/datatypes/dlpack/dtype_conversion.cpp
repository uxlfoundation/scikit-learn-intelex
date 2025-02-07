/*******************************************************************************
* Copyright Contributors to the oneDAL project
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

#include <type_traits>
#include <unordered_map>

#include "onedal/datatypes/dlpack/dtype_conversion.hpp"

namespace oneapi::dal::python::dlpack {

union type_hash_union {
    std::uint8_t bytes[4];
    std::uint16_t halfs[2];
    std::uint32_t full = 0u;

    constexpr type_hash_union(const DLDataType& t) {
        bytes[0] = t.code;
        bytes[1] = t.bits;
        halfs[1] = t.lanes;
    }
};

struct dlpack_type_hash {
    using type_t = DLDataType;
    constexpr std::uint32_t operator()(const type_t& type) const {
        return type_hash_union(type).full;
    }
};

struct dlpack_type_equal {
    using type_t = DLDataType;
    constexpr bool operator()(const type_t& lhs, const type_t& rhs) const {
        const auto l = type_hash_union(lhs).full;
        const auto r = type_hash_union(rhs).full;
        return l == r;
    }
};

using fwd_map_t = std::unordered_map<DLDataType,
                                     dal::data_type, //
                                     dlpack_type_hash,
                                     dlpack_type_equal>;
using inv_map_t = std::unordered_map<dal::data_type,
                                     DLDataType, //
                                     std::hash<dal::data_type>,
                                     std::equal_to<dal::data_type>>;

template <typename Type>
constexpr inline DLDataTypeCode make_dlpack_type_code() {
    if constexpr (std::is_integral_v<Type>) {
        if constexpr (std::is_unsigned_v<Type>) {
            return DLDataTypeCode::kDLUInt;
        }
        else {
            return DLDataTypeCode::kDLInt;
        }
    }
    else {
        if constexpr (std::is_floating_point_v<Type>) {
            return DLDataTypeCode::kDLFloat;
        }
        else {
            throw std::runtime_error("Unsupported");
            return static_cast<DLDataTypeCode>(-1);
        }
    }
}

template <typename Type>
constexpr inline DLDataType make_dlpack_type(std::uint16_t lanes = 0) {
    constexpr auto code = make_dlpack_type_code<Type>();
    constexpr auto size = std::size_t(8) * sizeof(Type);
    return DLDataType{ code, size, lanes };
}

template <typename... Types>
inline auto make_inv_map(const std::tuple<Types...>* const = nullptr) {
    inv_map_t result(1ul * sizeof...(Types));

    dal::detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            constexpr auto dal_v = detail::make_data_type<type_t>();
            result.emplace(dal_v, make_dlpack_type<type_t>(1));
        },
        Types{}...);

    return result;
}

template <typename... Types>
inline auto make_fwd_map(const std::tuple<Types...>* const = nullptr) {
    fwd_map_t result(2ul * sizeof...(Types));

    dal::detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            constexpr auto dal_v = detail::make_data_type<type_t>();
            result.emplace(make_dlpack_type<type_t>(0), dal_v);
            result.emplace(make_dlpack_type<type_t>(1), dal_v);
        },
        Types{}...);

    return result;
}

static const inv_map_t& get_inv_map() {
    constexpr const supported_types_t* types = nullptr;
    static const inv_map_t body = make_inv_map(types);
    return body;
}

static const fwd_map_t& get_fwd_map() {
    constexpr const supported_types_t* types = nullptr;
    static const fwd_map_t body = make_fwd_map(types);
    return body;
}

dal::data_type convert_dlpack_to_dal_type(DLDataType dtype) {
    if (get_fwd_map().count(dtype)) {
        return get_fwd_map().at(dtype);
    }
    else {
        throw py::type_error("Found unsupported tensor type");
    }
}

DLDataType convert_dal_to_dlpack_type(dal::data_type dtype) {
    if (get_inv_map().count(dtype)) {
        return get_inv_map().at(dtype);
    }
    else {
        throw py::type_error("Found unsupported dal type");
    }
}

} // namespace oneapi::dal::python::dlpack
