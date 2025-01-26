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

#include <map>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "oneapi/dal/common.hpp"

#define SET_CTYPE_NPY_FROM_DAL_TYPE(_T, _FUNCT, _EXCEPTION) \
    switch (_T) {                                           \
        case dal::data_type::float32: {                     \
            _FUNCT(NPY_FLOAT32);                            \
            break;                                          \
        }                                                   \
        case dal::data_type::float64: {                     \
            _FUNCT(NPY_FLOAT64);                            \
            break;                                          \
        }                                                   \
        case dal::data_type::int32: {                       \
            _FUNCT(NPY_INT32);                              \
            break;                                          \
        }                                                   \
        case dal::data_type::int64: {                       \
            _FUNCT(NPY_INT64);                              \
            break;                                          \
        }                                                   \
        default: _EXCEPTION;                                \
    };

#define SET_CTYPES_NPY_FROM_DAL_TYPE(_T, _FUNCT, _EXCEPTION) \
    switch (_T) {                                            \
        case dal::data_type::float32: {                      \
            _FUNCT(NPY_FLOAT32, float);                      \
            break;                                           \
        }                                                    \
        case dal::data_type::float64: {                      \
            _FUNCT(NPY_FLOAT64, double);                     \
            break;                                           \
        }                                                    \
        case dal::data_type::int32: {                        \
            _FUNCT(NPY_INT32, std::int32_t);                 \
            break;                                           \
        }                                                    \
        case dal::data_type::int64: {                        \
            _FUNCT(NPY_INT64, std::int64_t);                 \
            break;                                           \
        }                                                    \
        default: _EXCEPTION;                                 \
    };

#define SET_NPY_FEATURE(_T, _S, _FUNCT, _EXCEPTION) \
    switch (_T) {                                   \
        case NPY_FLOAT:                             \
        case NPY_CFLOAT:                            \
        case NPY_FLOATLTR:                          \
        case NPY_CFLOATLTR: {                       \
            _FUNCT(float);                          \
            break;                                  \
        }                                           \
        case NPY_DOUBLE:                            \
        case NPY_CDOUBLE:                           \
        case NPY_DOUBLELTR:                         \
        case NPY_CDOUBLELTR: {                      \
            _FUNCT(double);                         \
            break;                                  \
        }                                           \
        case NPY_INTLTR:                            \
        case NPY_INT32: {                           \
            _FUNCT(std::int32_t);                   \
            break;                                  \
        }                                           \
        case NPY_UINTLTR:                           \
        case NPY_UINT32: {                          \
            _FUNCT(std::uint32_t);                  \
            break;                                  \
        }                                           \
        case NPY_LONGLONGLTR:                       \
        case NPY_INT64: {                           \
            _FUNCT(std::int64_t);                   \
            break;                                  \
        }                                           \
        case NPY_ULONGLONGLTR:                      \
        case NPY_UINT64: {                          \
            _FUNCT(std::uint64_t);                  \
            break;                                  \
        }                                           \
        case NPY_LONGLTR: {                         \
            if (_S == 4) {                          \
                _FUNCT(std::int32_t);               \
            }                                       \
            else if (_S == 8) {                     \
                _FUNCT(std::int64_t);               \
            }                                       \
            else {                                  \
                _EXCEPTION;                         \
            }                                       \
            break;                                  \
        }                                           \
        case NPY_ULONGLTR: {                        \
            if (_S == 4) {                          \
                _FUNCT(std::uint32_t);              \
            }                                       \
            else if (_S == 8) {                     \
                _FUNCT(std::uint64_t);              \
            }                                       \
            else {                                  \
                _EXCEPTION;                         \
            }                                       \
            break;                                  \
        }                                           \
        default: _EXCEPTION;                        \
    };

#define is_array(a)          ((a) && PyArray_Check(a))
#define array_type(a)        PyArray_TYPE((PyArrayObject *)a)
#define array_type_sizeof(a) PyArray_ITEMSIZE((PyArrayObject *)a)
#define array_is_behaved_C(a) \
    (PyArray_ISCARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_behaved_F(a) \
    (PyArray_ISFARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_native(a) (PyArray_ISNOTSWAPPED((PyArrayObject *)a))
#define array_numdims(a)   PyArray_NDIM((PyArrayObject *)a)
#define array_data(a)      PyArray_DATA((PyArrayObject *)a)
#define array_size(a, i)   PyArray_DIM((PyArrayObject *)a, i)

namespace oneapi::dal::python::numpy {

using npy_dtype_t = decltype(NPY_FLOAT);
using npy_to_dal_t = std::map<npy_dtype_t, dal::data_type>;
using dal_to_npy_t = std::map<dal::data_type, npy_dtype_t>;

const npy_to_dal_t &get_npy_to_dal_map();
const dal_to_npy_t &get_dal_to_npy_map();

dal::data_type convert_npy_to_dal_type(npy_dtype_t);
npy_dtype_t convert_dal_to_npy_type(dal::data_type);

} // namespace oneapi::dal::python::numpy
