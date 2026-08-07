/*
 * Copyright contributors to the oneDAL project
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
 */

#pragma once

#include <cstddef>
#include <string_view>

namespace oneapi::dal::python::result_option_detail {

inline constexpr bool is_ascii_word_character(char value) noexcept {
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9') || value == '_';
}

// Split a result-option string into its identifier tokens and invoke the
// callback for each one, in order. This replaces the std::regex("\\w+") scan
// that each estimator wrapper used to open-code, which had three problems: it
// constructed and compiled the regex on every call, on a path that runs once
// per fit/predict; std::regex is imbued with the global locale, so its notion
// of \w depended on process-wide state that Python code can change; and it
// pulled <regex> into five translation units for a scan that a dozen lines of
// straight-line code express exactly.
//
// The token definition matches ECMAScript \w, i.e. [A-Za-z0-9_], so the
// accepted and rejected option strings are unchanged. std::isalnum() is not
// used, as it is locale-sensitive and does not accept '_'.
//
// The callback may throw - the estimators use that to report an unrecognized
// option - and the exception propagates out unchanged.
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
