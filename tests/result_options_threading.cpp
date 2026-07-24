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

#include "onedal/common/result_options.hpp"

#include <atomic>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace result_option_detail = oneapi::dal::python::result_option_detail;

std::vector<std::string> tokenize(std::string_view value) {
    std::vector<std::string> tokens;
    result_option_detail::for_each_result_option(value, [&](std::string_view token) {
        tokens.emplace_back(token);
    });
    return tokens;
}

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("result-option parser check failed");
    }
}

int main() {
    require(result_option_detail::is_ascii_word_character('_'));
    require(!result_option_detail::is_ascii_word_character(static_cast<char>(0xe9)));
    require(!result_option_detail::is_ascii_word_character(static_cast<char>(0xff)));
    require(tokenize("").empty());
    require(tokenize("|, ").empty());
    require(tokenize("max|max") == std::vector<std::string>({ "max", "max" }));
    require(tokenize("alpha_beta") == std::vector<std::string>({ "alpha_beta" }));
    require(tokenize("max|unknown|min") ==
            std::vector<std::string>({ "max", "unknown", "min" }));
    require(tokenize(std::string_view("max\0min", 7)) ==
            std::vector<std::string>({ "max", "min" }));
    require(tokenize(std::string_view("max\xffmin", 7)) ==
            std::vector<std::string>({ "max", "min" }));

    bool exception_propagated = false;
    try {
        result_option_detail::for_each_result_option("max|unknown", [](std::string_view option) {
            if (option == "unknown") {
                throw std::invalid_argument("unknown result option");
            }
        });
    }
    catch (const std::invalid_argument&) {
        exception_propagated = true;
    }
    require(exception_propagated);

    std::atomic<bool> concurrent_parse_ok{ true };
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&]() {
            for (int iteration = 0; iteration < 1000; ++iteration) {
                if (tokenize("intercept|coefficients") !=
                    std::vector<std::string>({ "intercept", "coefficients" })) {
                    concurrent_parse_ok.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    require(concurrent_parse_ok.load(std::memory_order_relaxed));
}
