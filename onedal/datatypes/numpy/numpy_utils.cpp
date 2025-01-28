/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "onedal/datatypes/numpy/numpy_utils.hpp"

namespace oneapi::dal::python::numpy {

template <typename Key, typename Value>
auto reverse_map(const std::map<Key, Value>& input) {
    std::map<Value, Key> output;
    for (const auto& [key, value] : input) {
        output.emplace(std::make_pair(value, key));
    }
    return output;
}

const npy_to_dal_t& get_npy_to_dal_map() {
    static npy_to_dal_t body = {
        { NPY_INT32, dal::data_type::int32 },
        { NPY_INT64, dal::data_type::int64 },
        { NPY_FLOAT32, dal::data_type::float32 },
        { NPY_FLOAT64, dal::data_type::float64 },
    };
    return body;
}

const dal_to_npy_t& get_dal_to_npy_map() {
    static dal_to_npy_t body = reverse_map(get_npy_to_dal_map());
    return body;
}

dal::data_type convert_npy_to_dal_type(npy_dtype_t type) {
    return get_npy_to_dal_map().at(type);
}

npy_dtype_t convert_dal_to_npy_type(dal::data_type type) {
    return get_dal_to_npy_map().at(type);
}

} // namespace oneapi::dal::python::numpy
