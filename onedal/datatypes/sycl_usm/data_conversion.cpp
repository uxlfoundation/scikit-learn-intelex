/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL
#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <string>
#include <utility>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/common/sycl_interfaces.hpp"
#include "onedal/datatypes/sycl_usm/data_conversion.hpp"
#include "onedal/datatypes/sycl_usm/dtype_conversion.hpp"
#include "onedal/datatypes/sycl_usm/sycl_usm_utils.hpp"

namespace oneapi::dal::python::sycl_usm {

using namespace pybind11::literals;
// Please follow <https://intelpython.github.io/dpctl/latest/
// api_reference/dpctl/sycl_usm_array_interface.html#sycl-usm-array-interface-attribute>
// for the description of `__sycl_usm_array_interface__` protocol.

// Convert python object to oneDAL homogen table with zero-copy by use
// of `__sycl_usm_array_interface__` protocol.
template <typename Type>
dal::table convert_to_homogen_impl(py::object obj) {
    dal::table res{};

    // Get `__sycl_usm_array_interface__` dictionary representing USM allocations.
    auto sua_iface_dict = get_sua_interface(obj);

    // Python uses reference counting as its primary memory management technique.
    // Each object in Python has an associated reference count, representing the number
    // of references pointing to that object. When this count drops to zero, Python
    // automatically frees the memory occupied by the object.
    // Using as a deleter the convenience function for decreasing the reference count
    // of an instance and potentially deleting it when the count reaches zero.
    const auto deleter = [obj](const Type*) {
        obj.dec_ref();
    };

    // Get and check `__sycl_usm_array_interface__` number of dimensions.
    const auto ndim = get_and_check_sua_iface_ndim(sua_iface_dict);

    // Get the pair of row and column counts.
    const auto [r_count, c_count] = get_sua_iface_shape_by_values(sua_iface_dict, ndim);

    // Get oneDAL Homogen DataLayout enumeration from input object shape and strides.
    const auto layout = get_sua_iface_layout(sua_iface_dict, r_count, c_count);

    if (layout == dal::data_layout::unknown) {
        // NOTE: this will make a C-contiguous deep copy of the data
        // if possible, this is expected to be a special case
        py::object copy;
        if (py::hasattr(obj, "copy")) {
            copy = obj.attr("copy")();
        }
        else if (py::hasattr(obj, "__array_namespace__")) {
            const auto space = obj.attr("__array_namespace__")();
            copy = space.attr("asarray")(obj, "copy"_a = true);
        }
        else {
            throw std::runtime_error("Wrong strides");
        }
        res = convert_to_homogen_impl<Type>(copy);
        copy.dec_ref();
        return res;
    }

    // Get `__sycl_usm_array_interface__['data'][0]`, the first element of data entry,
    // which is a Python integer encoding USM pointer value.
    const auto* const ptr = reinterpret_cast<const Type*>(get_sua_ptr(sua_iface_dict));

    // Get SYCL object from `__sycl_usm_array_interface__["syclobj"]`.
    // syclobj: Python object from which SYCL context to which represented USM
    // allocation is bound.
    auto syclobj = sua_iface_dict["syclobj"].cast<py::object>();

    // Get sycl::queue from syclobj.
    const auto queue = get_queue_from_python(syclobj);

    // Use read-only accessor for onedal table.
    bool is_readonly = is_sua_readonly(sua_iface_dict);

    if (is_readonly) {
        res = dal::homogen_table(queue,
                                 ptr,
                                 r_count,
                                 c_count,
                                 deleter,
                                 std::vector<sycl::event>{},
                                 layout);
    }
    else {
        auto* const mut_ptr = const_cast<Type*>(ptr);
        res = dal::homogen_table(queue,
                                 mut_ptr,
                                 r_count,
                                 c_count,
                                 deleter,
                                 std::vector<sycl::event>{},
                                 layout);
    }

    // Towards the python object memory model increment the python object reference
    // count due to new reference by oneDAL table pointing to that object.
    obj.inc_ref();
    return res;
}

// Convert oneDAL table with zero-copy by use of `__sycl_usm_array_interface__` protocol.
dal::table convert_to_table(py::object obj) {
    // Get `__sycl_usm_array_interface__` dictionary representing USM allocations.
    auto sua_iface_dict = get_sua_interface(obj);

    // Convert a string encoding elemental data type of the array to oneDAL homogen table
    // data type.
    const auto type = get_sua_dtype(sua_iface_dict);

    dal::table res{};

#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType>(obj);

    SET_CTYPE_FROM_DAL_TYPE(type,
                            MAKE_HOMOGEN_TABLE,
                            throw py::type_error("Found unsupported array type"));

#undef MAKE_HOMOGEN_TABLE

    return res;
}

// Create a dictionary for `__sycl_usm_array_interface__` protocol from oneDAL table properties.
py::dict construct_sua_iface(const dal::table& input) {
    const auto kind = input.get_kind();
    if (kind != dal::homogen_table::kind())
        report_problem_to_sua_iface(": only homogen tables are supported");

    const auto& homogen_input = reinterpret_cast<const dal::homogen_table&>(input);
    const dal::data_type dtype = homogen_input.get_metadata().get_data_type(0);
    const dal::data_layout data_layout = homogen_input.get_data_layout();

    npy_intp row_count = dal::detail::integral_cast<npy_intp>(homogen_input.get_row_count());
    npy_intp column_count = dal::detail::integral_cast<npy_intp>(homogen_input.get_column_count());

    // `__sycl_usm_array_interface__` protocol is a Python dictionary with the following fields:
    // shape: tuple of int
    // typestr: string
    // typedescr: a list of tuples
    // data: (int, bool)
    // strides: tuple of int
    // offset: int
    // version: int
    // syclobj: dpctl.SyclQueue or dpctl.SyclContext object or `SyclQueueRef` PyCapsule that
    //          represents an opaque value of sycl::queue.

    py::tuple shape = py::make_tuple(row_count, column_count);
    py::list data_entry(2);

    auto bytes_array = dal::detail::get_original_data(homogen_input);
    auto has_queue = bytes_array.get_queue().has_value();
    // oneDAL returns tables without sycl context for CPU sycl queue inputs, that
    // breaks the compute-follows-data execution.
    // Currently not throwing runtime exception and __sycl_usm_array_interface__["syclobj"] None asigned
    // if no SYCL queue to allow workaround on python side.
    // if (!has_queue) {
    //     report_problem_to_sua_iface(": table has no queue");
    // }

    const bool is_mutable = bytes_array.has_mutable_data();

    // data: A 2-tuple whose first element is a Python integer encoding
    // USM pointer value. The second entry in the tuple is a read-only flag
    // (True means the data area is read-only).
    data_entry[0] = is_mutable ? reinterpret_cast<std::size_t>(bytes_array.get_mutable_data())
                               : reinterpret_cast<std::size_t>(bytes_array.get_data());
    data_entry[1] = is_mutable;

    py::dict iface;
    iface["data"] = data_entry;
    // shape: a tuple of integers describing dimensions of an N-dimensional array.
    // Note:  oneDAL supports only (r,1) for 1-D arrays. In python code after from_table conversion
    // for 1-D expected outputs xp.ravel or reshape(-1) is used.
    // TODO:
    // probably worth to update for 1-D arrays.
    iface["shape"] = shape;

    // strides: An optional tuple of integers describing number of array elements needed to jump
    // to the next array element in the corresponding dimensions.
    iface["strides"] = get_npy_strides(data_layout, row_count, column_count);

    // Version of the `__sycl_usm_array_interface__`. At present, the only supported value is 1.
    iface["version"] = 1;

    // Convert oneDAL homogen table data type to a string encoding elemental data type of the array.
    iface["typestr"] = convert_dal_to_sua_type(dtype);

    // syclobj: Python object from which SYCL context to which represented USM allocation is bound.
    if (!has_queue) {
        iface["syclobj"] = py::none();
    }
    else {
        iface["syclobj"] =
            pack_queue(std::make_shared<sycl::queue>(bytes_array.get_queue().value()));
    }

    return iface;
}

// Adding `__sycl_usm_array_interface__` attribute to python oneDAL table, that representing
// USM allocations.
void define_sycl_usm_array_property(py::class_<dal::table>& table_obj) {
    // To enable native extensions to pass the memory allocated by a native SYCL library to SYCL-aware
    // Python extension without making a copy, the class must provide `__sycl_usm_array_interface__`
    // attribute representing USM allocations. The `__sycl_usm_array_interface__` attribute is used
    // for constructing DPCTL usm_ndarray or DPNP ndarray with zero-copy on python level.
    table_obj.def_property_readonly("__sycl_usm_array_interface__", &construct_sua_iface);
}

} // namespace oneapi::dal::python::sycl_usm

#endif // ONEDAL_DATA_PARALLEL
