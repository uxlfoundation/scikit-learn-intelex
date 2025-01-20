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

#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <string>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/datatypes/data_conversion.hpp"
#include "onedal/datatypes/utils/numpy_helpers.hpp"
#include "onedal/version.hpp"

#if ONEDAL_VERSION <= 20230100
#include "oneapi/dal/table/detail/csr.hpp"
#else
#include "oneapi/dal/table/csr.hpp"
#endif

namespace oneapi::dal::python {

#if ONEDAL_VERSION <= 20230100
typedef oneapi::dal::detail::csr_table csr_table_t;
#else
typedef oneapi::dal::csr_table csr_table_t;
#endif

template <typename T>
static dal::array<T> transfer_to_host(const dal::array<T> &array) {
#ifdef ONEDAL_DATA_PARALLEL
    auto opt_queue = array.get_queue();
    if (opt_queue.has_value()) {
        auto device = opt_queue->get_device();
        if (!device.is_cpu()) {
            const auto *device_data = array.get_data();

            auto memory_kind = sycl::get_pointer_type(device_data, opt_queue->get_context());
            if (memory_kind == sycl::usm::alloc::unknown) {
                throw std::runtime_error("[convert_to_numpy] Unknown memory type");
            }
            if (memory_kind == sycl::usm::alloc::device) {
                auto host_array = dal::array<T>::empty(array.get_count());
                opt_queue->memcpy(host_array.get_mutable_data(), device_data, array.get_size())
                    .wait_and_throw();
                return host_array;
            }
        }
    }
#endif

    return array;
}

template <typename T>
inline dal::homogen_table convert_to_homogen_impl(PyArrayObject *np_data) {
    std::int64_t column_count = 1;

    if (array_numdims(np_data) > 2) {
        throw std::runtime_error("Input array has wrong dimensionality (must be 2d).");
    }
    T *const data_pointer = reinterpret_cast<T *const>(array_data(np_data));
    // TODO: check safe cast from int to std::int64_t
    const std::int64_t row_count = static_cast<std::int64_t>(array_size(np_data, 0));
    if (array_numdims(np_data) == 2) {
        // TODO: check safe cast from int to std::int64_t
        column_count = static_cast<std::int64_t>(array_size(np_data, 1));
    }
    // If both array_is_behaved_C(np_data) and array_is_behaved_F(np_data) are true
    // (for example, if the array has only one column), then row-major layout will be chosen
    // which is default on oneDAL side.
    const auto layout =
        array_is_behaved_C(np_data) ? dal::data_layout::row_major : dal::data_layout::column_major;
    auto res_table = dal::homogen_table(
        data_pointer,
        row_count,
        column_count,
        [np_data](const T *data) {
            Py_DECREF(np_data);
        },
        layout);

    // we need to increment the ref-count as we use the input array in-place
    Py_INCREF(np_data);
    return res_table;
}

template <typename T>
inline csr_table_t convert_to_csr_impl(PyObject *py_data,
                                       PyObject *py_column_indices,
                                       PyObject *py_row_indices,
                                       std::int64_t row_count,
                                       std::int64_t column_count) {
    PyArrayObject *np_data = reinterpret_cast<PyArrayObject *>(py_data);
    PyArrayObject *np_column_indices = reinterpret_cast<PyArrayObject *>(py_column_indices);
    PyArrayObject *np_row_indices = reinterpret_cast<PyArrayObject *>(py_row_indices);

    const std::int64_t *row_indices_zero_based =
        static_cast<std::int64_t *>(array_data(np_row_indices));
    const std::int64_t row_indices_count = static_cast<std::int64_t>(array_size(np_row_indices, 0));

    auto row_indices_one_based = dal::array<std::int64_t>::empty(row_indices_count);
    auto row_indices_one_based_data = row_indices_one_based.get_mutable_data();

    for (std::int64_t i = 0; i < row_indices_count; ++i)
        row_indices_one_based_data[i] = row_indices_zero_based[i] + 1;

    const std::int64_t *column_indices_zero_based =
        static_cast<std::int64_t *>(array_data(np_column_indices));
    const std::int64_t column_indices_count =
        static_cast<std::int64_t>(array_size(np_column_indices, 0));

    auto column_indices_one_based = dal::array<std::int64_t>::empty(column_indices_count);
    auto column_indices_one_based_data = column_indices_one_based.get_mutable_data();

    for (std::int64_t i = 0; i < column_indices_count; ++i)
        column_indices_one_based_data[i] = column_indices_zero_based[i] + 1;

    const T *data_pointer = static_cast<T *>(array_data(np_data));
    const std::int64_t data_count = static_cast<std::int64_t>(array_size(np_data, 0));

    auto res_table = csr_table_t(
        dal::array<T>(data_pointer,
                      data_count,
                      [np_data](const T *) {
                          Py_DECREF(np_data);
                      }),
        column_indices_one_based,
        row_indices_one_based,
#if ONEDAL_VERSION <= 20230100
        // row_count parameter is present in csr_table's constructor only in older versions of oneDAL
        row_count,
#endif
        column_count);

    // we need to increment the ref-count as we use the input array in-place
    Py_INCREF(np_data);
    return res_table;
}

dal::table convert_to_table(py::object inp_obj, py::object queue) {
    dal::table res;
#ifdef ONEDAL_DATA_PARALLEL
    if (!queue.is(py::none()) && !queue.attr("sycl_device").attr("has_aspect_fp64").cast<bool>()) {
        // If the queue exists, doesn't have the fp64 aspect, and the data is float64
        // then cast it to float32
        int type = reinterpret_cast<PyArray_Descr *>(inp_obj.attr("dtype").ptr())->type_num;
        if (type == NPY_DOUBLE || type == NPY_DOUBLELTR) {
            PyErr_WarnEx(
                PyExc_RuntimeWarning,
                "Data will be converted into float32 from float64 because device does not support it",
                1);
            // use astype instead of PyArray_Cast in order to support scipy sparse inputs
            inp_obj = inp_obj.attr("astype")(py::dtype::of<float>());
            res = convert_to_table(
                inp_obj); // queue will be set to none, as this check is no longer necessary
            return res;
        }
    }
#endif // ONEDAL_DATA_PARALLEL

    PyObject *obj = inp_obj.ptr();

    if (obj == nullptr || obj == Py_None) {
        return res;
    }
    if (is_array(obj)) {
        PyArrayObject *ary = reinterpret_cast<PyArrayObject *>(obj);

        if (!PyArray_ISCARRAY_RO(ary) && !PyArray_ISFARRAY_RO(ary)) {
            // NOTE: this will make a C-contiguous deep copy of the data
            // this is expected to be a special case
            obj = reinterpret_cast<PyObject *>(PyArray_GETCONTIGUOUS(ary));
            if (obj) {
                res = convert_to_table(py::cast<py::object>(obj), queue);
                Py_DECREF(obj);
                return res;
            }
            else {
                throw std::invalid_argument(
                    "[convert_to_table] Numpy input could not be converted into onedal table.");
            }
        }
#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType>(ary);
        SET_NPY_FEATURE(array_type(ary),
                        array_type_sizeof(ary),
                        MAKE_HOMOGEN_TABLE,
                        throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
    }
    else if (strcmp(Py_TYPE(obj)->tp_name, "csr_matrix") == 0 ||
             strcmp(Py_TYPE(obj)->tp_name, "csr_array") == 0) {
        PyObject *py_data = PyObject_GetAttrString(obj, "data");
        PyObject *py_column_indices = PyObject_GetAttrString(obj, "indices");
        PyObject *py_row_indices = PyObject_GetAttrString(obj, "indptr");

        PyObject *py_shape = PyObject_GetAttrString(obj, "shape");
        if (!(is_array(py_data) && is_array(py_column_indices) && is_array(py_row_indices) &&
              array_numdims(py_data) == 1 && array_numdims(py_column_indices) == 1 &&
              array_numdims(py_row_indices) == 1)) {
            throw std::invalid_argument("[convert_to_table] Got invalid csr_matrix object.");
        }
        PyObject *np_data = PyArray_FROMANY(py_data, array_type(py_data), 0, 0, NPY_ARRAY_CARRAY);
        PyObject *np_column_indices =
            PyArray_FROMANY(py_column_indices,
                            NPY_UINT64,
                            0,
                            0,
                            NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
        PyObject *np_row_indices =
            PyArray_FROMANY(py_row_indices,
                            NPY_UINT64,
                            0,
                            0,
                            NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);

        PyObject *np_row_count = PyTuple_GetItem(py_shape, 0);
        PyObject *np_column_count = PyTuple_GetItem(py_shape, 1);
        if (!(np_data && np_column_indices && np_row_indices && np_row_count && np_column_count)) {
            throw std::invalid_argument(
                "[convert_to_table] Failed accessing csr data when converting csr_matrix.\n");
        }

        const std::int64_t row_count = static_cast<std::int64_t>(PyLong_AsSsize_t(np_row_count));
        const std::int64_t column_count =
            static_cast<std::int64_t>(PyLong_AsSsize_t(np_column_count));

#define MAKE_CSR_TABLE(CType)                           \
    res = convert_to_csr_impl<CType>(np_data,           \
                                     np_column_indices, \
                                     np_row_indices,    \
                                     row_count,         \
                                     column_count);
        SET_NPY_FEATURE(array_type(np_data),
                        array_type_sizeof(np_data),
                        MAKE_CSR_TABLE,
                        throw std::invalid_argument("Found unsupported data type in csr_matrix"));
#undef MAKE_CSR_TABLE
        Py_DECREF(np_column_indices);
        Py_DECREF(np_row_indices);
    }
    else {
        throw std::invalid_argument(
            "[convert_to_table] Not available input format for convert Python object to onedal table.");
    }
    return res;
}

static void free_capsule(PyObject *cap) {
    // TODO: check safe cast
    dal::base *stored_array = static_cast<dal::base *>(PyCapsule_GetPointer(cap, NULL));
    if (stored_array) {
        delete stored_array;
    }
}

template <int NpType, typename T = byte_t>
static PyObject *convert_to_numpy_impl(const dal::array<T> &array,
                                       std::int64_t row_count,
                                       std::int64_t column_count = 0) {
    const int size_dims = column_count == 0 ? 1 : 2;

    npy_intp dims[2] = { static_cast<npy_intp>(row_count), static_cast<npy_intp>(column_count) };
    auto host_array = transfer_to_host(array);
    host_array.need_mutable_data();
    auto *bytes = host_array.get_mutable_data();

    PyObject *obj = PyArray_SimpleNewFromData(size_dims, dims, NpType, static_cast<void *>(bytes));
    if (!obj)
        throw std::invalid_argument("Conversion to numpy array failed");

    void *opaque_value = static_cast<void *>(new dal::array<T>(host_array));
    PyObject *cap = PyCapsule_New(opaque_value, NULL, free_capsule);
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), cap);
    return obj;
}

#if ONEDAL_VERSION <= 20230100

// dal::detail::csr_table class is valid
// only one-based indeices are supported
template <int NpType, typename T>
static PyObject *convert_to_py_from_csr_impl(const detail::csr_table &table) {
    PyObject *result = PyTuple_New(3);
    const std::int64_t rows_indices_count = table.get_row_count() + 1;

    const std::int64_t *row_indices_one_based = table.get_row_indices();
    std::uint64_t *row_indices_zero_based_data =
        detail::host_allocator<std::uint64_t>().allocate(rows_indices_count);
    for (std::int64_t i = 0; i < rows_indices_count; ++i)
        row_indices_zero_based_data[i] = row_indices_one_based[i] - 1;

    auto row_indices_zero_based_array =
        dal::array<std::uint64_t>::wrap(row_indices_zero_based_data, rows_indices_count);
    PyObject *py_row =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(row_indices_zero_based_array,
                                                         rows_indices_count);
    PyTuple_SetItem(result, 2, py_row);

    const std::int64_t non_zero_count = row_indices_zero_based_data[rows_indices_count - 1];
    const T *data = reinterpret_cast<const T *>(table.get_data());
    auto data_array = dal::array<T>::wrap(data, non_zero_count);

    PyObject *py_data = convert_to_numpy_impl<NpType, T>(data_array, non_zero_count);
    PyTuple_SetItem(result, 0, py_data);

    const std::int64_t *column_indices_one_based = table.get_column_indices();
    std::uint64_t *column_indices_zero_based_data =
        detail::host_allocator<std::uint64_t>().allocate(non_zero_count);
    for (std::int64_t i = 0; i < non_zero_count; ++i)
        column_indices_zero_based_data[i] = column_indices_one_based[i] - 1;

    auto column_indices_zero_based_array =
        dal::array<std::uint64_t>::wrap(column_indices_zero_based_data, non_zero_count);
    PyObject *py_col =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(column_indices_zero_based_array,
                                                         non_zero_count);
    PyTuple_SetItem(result, 1, py_col);
    return result;
}

#else // ONEDAL_VERSION > 20230100

// dal::csr_table class is valid
// zero- and one-based indeices are supported
template <int NpType, typename T>
static PyObject *convert_to_py_from_csr_impl(const csr_table &table) {
    PyObject *result = PyTuple_New(3);
    const std::int64_t rows_indices_count = table.get_row_count() + 1;
    const std::int64_t non_zero_count = table.get_non_zero_count();
    const std::int64_t *row_offsets = table.get_row_offsets();
    const std::int64_t *column_indices = table.get_column_indices();

    std::uint64_t *column_indices_zero_based_data = nullptr;
    std::uint64_t *row_offsets_zero_based_data = nullptr;

    if (table.get_indexing() == sparse_indexing::zero_based) {
        column_indices_zero_based_data =
            const_cast<std::uint64_t *>(reinterpret_cast<const std::uint64_t *>(column_indices));
        row_offsets_zero_based_data =
            const_cast<std::uint64_t *>(reinterpret_cast<const std::uint64_t *>(row_offsets));
    }
    else { // table.get_indexing() == sparse_indexing::one_based
        column_indices_zero_based_data =
            detail::host_allocator<std::uint64_t>().allocate(non_zero_count);
        row_offsets_zero_based_data =
            detail::host_allocator<std::uint64_t>().allocate(rows_indices_count);

        for (std::int64_t i = 0; i < non_zero_count; ++i)
            column_indices_zero_based_data[i] = column_indices[i] - 1;

        for (std::int64_t i = 0; i < rows_indices_count; ++i)
            row_offsets_zero_based_data[i] = row_offsets[i] - 1;
    }

    const T *data = table.get_data<T>();
    auto data_array = dal::array<T>::wrap(data, non_zero_count);

    PyObject *py_data = convert_to_numpy_impl<NpType, T>(data_array, non_zero_count);
    PyTuple_SetItem(result, 0, py_data);

    auto column_indices_zero_based_array =
        dal::array<std::uint64_t>::wrap(column_indices_zero_based_data, non_zero_count);
    PyObject *py_col =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(column_indices_zero_based_array,
                                                         non_zero_count);
    PyTuple_SetItem(result, 1, py_col);
    auto row_indices_zero_based_array =
        dal::array<std::uint64_t>::wrap(row_offsets_zero_based_data, rows_indices_count);
    PyObject *py_row =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(row_indices_zero_based_array,
                                                         rows_indices_count);
    PyTuple_SetItem(result, 2, py_row);
    return result;
}

#endif // ONEDAL_VERSION <= 20230100

PyObject *convert_to_pyobject(const dal::table &input) {
    PyObject *res = nullptr;
    if (!input.has_data()) {
        npy_intp dims[1] = { static_cast<npy_intp>(0) };
        return PyArray_EMPTY(1, dims, NPY_INT32, 0);
    }
    if (input.get_kind() == dal::homogen_table::kind()) {
        const auto &homogen_input = static_cast<const dal::homogen_table &>(input);
        const dal::data_type dtype = homogen_input.get_metadata().get_data_type(0);

#define MAKE_NYMPY_FROM_HOMOGEN(NpType)                                        \
    {                                                                          \
        auto bytes_array = dal::detail::get_original_data(homogen_input);      \
        res = convert_to_numpy_impl<NpType>(bytes_array,                       \
                                            homogen_input.get_row_count(),     \
                                            homogen_input.get_column_count()); \
    }
        SET_CTYPE_NPY_FROM_DAL_TYPE(dtype,
                                    MAKE_NYMPY_FROM_HOMOGEN,
                                    throw std::invalid_argument("Unable to convert numpy object"));
#undef MAKE_NYMPY_FROM_HOMOGEN
    }
    else if (input.get_kind() == csr_table_t::kind()) {
        const auto &csr_input = static_cast<const csr_table_t &>(input);
        const dal::data_type dtype = csr_input.get_metadata().get_data_type(0);
#define MAKE_PY_FROM_CSR(NpType, T) \
    { res = convert_to_py_from_csr_impl<NpType, T>(csr_input); }
        SET_CTYPES_NPY_FROM_DAL_TYPE(
            dtype,
            MAKE_PY_FROM_CSR,
            throw std::invalid_argument("Unable to convert scipy csr object"));
#undef MAKE_PY_FROM_CSR
    }
    else {
        throw std::invalid_argument("Output oneDAL table doesn't have homogen or csr format");
    }
    return res;
}

} // namespace oneapi::dal::python
