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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "onedal/datatypes/sycl_usm/data_conversion.hpp"
#endif // ONEDAL_DATA_PARALLEL

#include "onedal/datatypes/numpy/data_conversion.hpp"
#include "onedal/datatypes/dlpack/data_conversion.hpp"
#include "onedal/datatypes/numpy/numpy_utils.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/version.hpp"

#if ONEDAL_VERSION <= 20230100
#include "oneapi/dal/table/detail/csr.hpp"
#else
#include "oneapi/dal/table/csr.hpp"
#endif

namespace py = pybind11;

namespace oneapi::dal::python {

#if ONEDAL_VERSION <= 20230100
typedef oneapi::dal::detail::csr_table csr_table_t;
#else
typedef oneapi::dal::csr_table csr_table_t;
#endif

static void* init_numpy() {
    import_array();
    return nullptr;
}

ONEDAL_PY_INIT_MODULE(table) {
    init_numpy();

    py::class_<table> table_obj(m, "table");
    table_obj.def(py::init());
    table_obj.def_property_readonly("has_data", &table::has_data);
    table_obj.def_property_readonly("column_count", &table::get_column_count);
    table_obj.def_property_readonly("row_count", &table::get_row_count);
    table_obj.def_property_readonly("kind", [](const table& t) {
        if (t.get_kind() == 0) { // TODO: expose empty table kind
            return "empty";
        }
        if (t.get_kind() == homogen_table::kind()) {
            return "homogen";
        }
        if (t.get_kind() == csr_table_t::kind()) {
            return "csr";
        }
        return "unknown";
    });
    table_obj.def_property_readonly("shape", [](const table& t) {
        const auto row_count = t.get_row_count();
        const auto column_count = t.get_column_count();
        return py::make_tuple(row_count, column_count);
    });
    table_obj.def_property_readonly("dtype", [](const table& t) {
        // returns a numpy dtype, even if source was not from numpy
        return py::dtype(numpy::convert_dal_to_npy_type(t.get_metadata().get_data_type(0)));
    });
    table_obj.def("__dlpack__", &dlpack::construct_dlpack);
    table_obj.def("__dlpack_device__", [](const table& t) {
        auto dlpack_device = dlpack::get_dlpack_device(t);
        return py::make_tuple(dlpack_device.device_type, dlpack_device.device_id);
    });

#ifdef ONEDAL_DATA_PARALLEL
    sycl_usm::define_sycl_usm_array_property(table_obj);
#endif // ONEDAL_DATA_PARALLEL

    m.def("to_table", [](py::object obj, py::object queue) {
        if (py::isinstance<py::array>(obj)) {
            return numpy::convert_to_table(obj, queue);
        }
#ifdef ONEDAL_DATA_PARALLEL
        if (py::hasattr(obj, "__sycl_usm_array_interface__")) {
            return sycl_usm::convert_to_table(obj);
        }
#endif // ONEDAL_DATA_PARALLEL
        if (py::hasattr(obj, "__dlpack__")) {
            return dlpack::convert_to_table(obj, queue);
        }
        // assume to be sparse (handled in numpy)
        return numpy::convert_to_table(obj, queue);
    });

    m.def("from_table", [](const dal::table& t) -> py::handle {
        auto* obj_ptr = numpy::convert_to_pyobject(t);
        return obj_ptr;
    });
    m.def("dlpack_memory_order", &dlpack::dlpack_memory_order);
    py::enum_<DLDeviceType>(m, "DLDeviceType")
        .value("kDLCPU", kDLCPU)
        .value("kDLOneAPI", kDLOneAPI);
}

} // namespace oneapi::dal::python
