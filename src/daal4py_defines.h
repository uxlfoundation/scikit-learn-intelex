/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef _DAAL4PY_DEFINES_H_
#define _DAAL4PY_DEFINES_H_

#define DAAL4PY_OVERFLOW_CHECK_BY_MULTIPLICATION(type, op1, op2)                         \
    {                                                                                    \
        if (!(0 == (op1)) && !(0 == (op2)))                                              \
        {                                                                                \
            volatile type r = (op1) * (op2);                                             \
            r /= (op1);                                                                  \
            if (!(r == (op2))) throw std::runtime_error("Buffer size integer overflow"); \
        }                                                                                \
    }

#define DAAL4PY_OVERFLOW_CHECK_BY_ADDING(type, op1, op2)                             \
    {                                                                                \
        volatile type r = (op1) + (op2);                                             \
        r -= (op1);                                                                  \
        if (!(r == (op2))) throw std::runtime_error("Buffer size integer overflow"); \
    }

#define DAAL4PY_CHECK(cond, error) \
    if (!(cond)) throw std::runtime_error(error);

#define DAAL4PY_CHECK_BAD_CAST(cond) \
    if (!(cond)) throw std::runtime_error("Bad casting");

#define DAAL4PY_CHECK_MALLOC(cond) \
    if (!(cond)) throw std::bad_alloc();

#endif
