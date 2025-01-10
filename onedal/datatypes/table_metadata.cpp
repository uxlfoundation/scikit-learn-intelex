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

#include "oneapi/dal/table/common.hpp"

#include "onedal/datatypes/utils/numpy_helpers.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(table_metadata) {
    py::class_<table_metadata> table_metadata_obj(m, "table_metadata");
    table_metadata_obj.def(py::init());
    table_metadata_obj.def_property_readonly("feature_count", //
                                             &table_metadata::get_feature_count);

    table_metadata_obj.def("get_raw_dtype",
                           [](const table_metadata* const ptr, std::int64_t feature) {
                               return static_cast<std::int64_t>(ptr->get_data_type(feature));
                           });

    table_metadata_obj.def("get_npy_dtype",
                           [](const table_metadata* const ptr, std::int64_t feature) {
                               const auto npy_type =
                                   convert_dal_to_npy_type(ptr->get_data_type(feature));
                               return py::dtype(npy_type);
                           });

    m.def("get_table_metadata", [](const dal::table& t) {
        return t.get_metadata();
    });
}

} // namespace oneapi::dal::python
