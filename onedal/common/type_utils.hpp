/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#define ONEDAL_PY_TYPE2STR(type, name) \
    template <>                        \
    struct type_to_str<type> {         \
        std::string operator()() {     \
            return name;               \
        }                              \
    }

namespace oneapi::dal::python {

template <class Head, class... Tail>
struct types {
    using head = Head;
    using last = types<Tail...>;
    static constexpr bool has_last = true;
};

template <typename T>
struct types<T> {
    using head = T;
    using last = types<T>;
    static constexpr bool has_last = false;
};

template <typename... Args>
struct is_type_list : public std::false_type {};

template <typename... Args>
struct is_type_list<types<Args...>> : public std::true_type {};

template <typename T>
struct type_to_str;

#ifdef ONEDAL_DATA_PARALLEL_SPMD
ONEDAL_PY_TYPE2STR(dal::detail::spmd_policy<dal::detail::data_parallel_policy>, "");
using policy_spmd = types<dal::detail::spmd_policy<dal::detail::data_parallel_policy>>;
#else
ONEDAL_PY_TYPE2STR(dal::detail::host_policy, "");
#ifdef ONEDAL_DATA_PARALLEL
ONEDAL_PY_TYPE2STR(dal::detail::data_parallel_policy, "");
using policy_list = types<dal::detail::host_policy, dal::detail::data_parallel_policy>;
#else
using policy_list = types<dal::detail::host_policy>;
#endif
#endif

} // namespace oneapi::dal::python
