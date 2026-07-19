/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <cstddef>
#include <string_view>

namespace oneapi::dal::python::result_option_detail {

inline constexpr bool is_ascii_word_character(char value) noexcept {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9') || value == '_';
}

// Result-option names are ASCII identifiers. Avoid std::regex("\\w+"), whose
// libstdc++ locale cache is not safe for concurrent first use without the GIL.
template <typename Callback>
void for_each_result_option(std::string_view value, Callback&& callback) {
    std::size_t position = 0;
    while (position < value.size()) {
        while (position < value.size() && !is_ascii_word_character(value[position])) {
            ++position;
        }

        const auto token_begin = position;
        while (position < value.size() && is_ascii_word_character(value[position])) {
            ++position;
        }

        if (token_begin != position) {
            callback(value.substr(token_begin, position - token_begin));
        }
    }
}

} // namespace oneapi::dal::python::result_option_detail
