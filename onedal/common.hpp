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

#define OVERFLOW_CHECK_BY_ADDING(type, op1, op2)                    \
    {                                                               \
        volatile type r = (op1) + (op2);                            \
        r -= (op1);                                                 \
        if (!(r == (op2)))                                          \
            throw std::runtime_error("Integer overflow by adding"); \
    }

#define OVERFLOW_CHECK_BY_MULTIPLICATION(type, op1, op2)                        \
    {                                                                           \
        if (!(0 == (op1)) && !(0 == (op2))) {                                   \
            volatile type r = (op1) * (op2);                                    \
            r /= (op1);                                                         \
            if (!(r == (op2)))                                                  \
                throw std::runtime_error("Integer overflow by multiplication"); \
        }                                                                       \
    }

#include "onedal/common/dispatch_utils.hpp"
#include "onedal/common/instantiate_utils.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/common/serialization.hpp"
