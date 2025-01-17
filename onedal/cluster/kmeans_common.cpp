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

#include <stdexcept>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "onedal/common/pybind11_helpers.hpp"

namespace oneapi::dal::python {

namespace kmeans {

bool is_same_clustering(const dal::table& left, const dal::table& right, std::int64_t n_clusters) {
    if (!left.has_data() || !right.has_data())
        throw std::invalid_argument("Empty input table");

    row_accessor<const std::int32_t> l_acc{ left };
    row_accessor<const std::int32_t> r_acc{ right };

    if (left.get_column_count() > 1 || right.get_column_count() > 1)
        throw std::length_error("Too many columns in input table");

    const auto l_arr = l_acc.pull({ 0, -1 });
    const auto r_arr = r_acc.pull({ 0, -1 });

    if (n_clusters < 1)
        throw std::invalid_argument("Invalid number of clusters");

    constexpr std::int32_t minus_one = -1;
    auto map = dal::array<std::int32_t>::full( //
        n_clusters,
        minus_one);

    auto* const m_ptr = map.get_mutable_data();

    const auto l_count = l_arr.get_count();
    const auto r_count = r_arr.get_count();

    if (l_count != r_count)
        throw std::length_error("Inconsistent number of labels");

    for (std::int64_t i = 0; i < l_count; ++i) {
        const auto l_label = l_arr[i];
        const auto r_label = r_arr[i];

        if (n_clusters <= l_label)
            throw std::out_of_range("Label is out of range");

        auto& l_map = m_ptr[l_label];

        if (l_map == minus_one) {
            l_map = r_label;
        }
        else if (l_map != r_label) {
            return false;
        }
    }

    return true;
}

} // namespace kmeans

ONEDAL_PY_INIT_MODULE(kmeans_common) {
    auto sub = m.def_submodule("kmeans_common");

    sub.def("_is_same_clustering", &kmeans::is_same_clustering);
} // ONEDAL_PY_INIT_MODULE(kmeans_common)

} // namespace oneapi::dal::python
