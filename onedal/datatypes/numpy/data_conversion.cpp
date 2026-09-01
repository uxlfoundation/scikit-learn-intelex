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

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/datatypes/numpy/data_conversion.hpp"
#include "onedal/datatypes/numpy/numpy_utils.hpp"
#include "onedal/datatypes/common.hpp"
#include "onedal/version.hpp"

#if ONEDAL_VERSION <= 20230100
#include "oneapi/dal/table/detail/csr.hpp"
#else
#include "oneapi/dal/table/csr.hpp"
#endif

namespace oneapi::dal::python::numpy {

#if ONEDAL_VERSION <= 20230100
typedef oneapi::dal::detail::csr_table csr_table_t;
#else
typedef oneapi::dal::csr_table csr_table_t;
#endif

static std::shared_ptr<PyObject> make_python_owner(PyObject *obj) {
    Py_INCREF(obj);
    return std::shared_ptr<PyObject>(obj, [](PyObject *owner) {
        // Attaching a thread during interpreter shutdown can hang indefinitely.
        // Leaking this final reference is preferable to entering the runtime after
        // finalization has started; the process is already tearing down.
        if (!Py_IsInitialized()) {
            return;
        }
#if PY_VERSION_HEX >= 0x030D0000
        if (Py_IsFinalizing()) {
            return;
        }
#endif
        const PyGILState_STATE state = PyGILState_Ensure();
        Py_DECREF(owner);
        PyGILState_Release(state);
    });
}

template <typename T>
inline dal::homogen_table convert_to_homogen_impl(PyArrayObject *np_data) {
    std::int64_t column_count = 1;
    const std::int32_t ndims = array_numdims(np_data);
    if (ndims > 2) {
        throw std::length_error("Input array has wrong dimensionality (must be 2d).");
    }
    T *const data_pointer = reinterpret_cast<T *const>(array_data(np_data));
    // TODO: check safe cast from int to std::int64_t
    // if 0 dimensional numpy array, force to 2d
    const std::int64_t row_count = ndims ? static_cast<std::int64_t>(array_size(np_data, 0)) : 1l;
    if (ndims == 2) {
        // TODO: check safe cast from int to std::int64_t
        column_count = static_cast<std::int64_t>(array_size(np_data, 1));
    }
    // If both array_is_behaved_C(np_data) and array_is_behaved_F(np_data) are true
    // (for example, if the array has only one column), then row-major layout will be chosen
    // which is default on oneDAL side.
    const auto layout =
        array_is_behaved_C(np_data) ? dal::data_layout::row_major : dal::data_layout::column_major;
    // The table borrows the array's buffer, so the capture keeps the array alive
    // for as long as any copy of the table exists.
    return dal::homogen_table(
        data_pointer,
        row_count,
        column_count,
        [owner = make_python_owner(reinterpret_cast<PyObject *>(np_data))](const T *) {},
        layout);
}

// Widen `count` base-0 indices of type Src into base-1 std::int64_t, reading the
// source as Src directly.
//
// The caller guarantees every element is aligned for Src - see the dispatch in
// make_one_based_indices() - so `stride` arrives here already divided by
// sizeof(Src) and can be negative or zero. A dedicated unit-stride loop is worth
// keeping: it is the case scipy produces, and it is the only one the vectorizer
// can widen, because the general loop's step is a runtime value.
template <typename Src>
inline void rebase_indices_to_one(const Src *source,
                                  std::int64_t element_stride,
                                  std::int64_t *destination,
                                  std::int64_t count) {
    if (element_stride == 0) {
        // A broadcast view repeats one element, so load it once.
        if (count > 0) {
            const std::int64_t rebased = static_cast<std::int64_t>(*source) + 1;
            for (std::int64_t i = 0; i < count; ++i) {
                destination[i] = rebased;
            }
        }
        return;
    }

    if (element_stride == 1) {
        for (std::int64_t i = 0; i < count; ++i) {
            destination[i] = static_cast<std::int64_t>(source[i]) + 1;
        }
        return;
    }

    for (std::int64_t i = 0; i < count; ++i) {
        destination[i] = static_cast<std::int64_t>(source[i * element_stride]) + 1;
    }
}

// The same widening for a buffer whose elements are not aligned for Src, which
// NumPy permits: a view can be unit-stride and still start at an odd byte. A
// typed dereference would be undefined there, so this walks bytes and copies
// each element out with a fixed-size memcpy. It is the cold path - scipy does
// not produce such a buffer - so it is kept simple rather than specialized.
template <typename Src>
inline void rebase_indices_to_one_unaligned(const char *source_bytes,
                                            std::int64_t stride,
                                            std::int64_t *destination,
                                            std::int64_t count) {
    for (std::int64_t i = 0; i < count; ++i) {
        Src value;
        std::memcpy(&value, source_bytes + i * stride, sizeof(Src));
        destination[i] = static_cast<std::int64_t>(value) + 1;
    }
}

// Pick between the two loops above for one source dtype. `stride` is in bytes;
// the typed loop wants it in elements, which is exact only when the caller has
// established that, hence the flag rather than a second test here.
template <typename Src>
inline void rebase_indices(const char *source,
                           std::int64_t stride,
                           bool readable_as_source,
                           std::int64_t *destination,
                           std::int64_t count) {
    if (readable_as_source) {
        rebase_indices_to_one<Src>(reinterpret_cast<const Src *>(source),
                                   stride / static_cast<std::int64_t>(sizeof(Src)),
                                   destination,
                                   count);
    }
    else {
        rebase_indices_to_one_unaligned<Src>(source, stride, destination, count);
    }
}

enum class index_kind { unsupported, int32, uint32, int64, uint64 };

// Classify by width and signedness rather than by NumPy type number, because
// NPY_INT32 and NPY_INT64 are aliases whose targets differ per platform: an
// int64 array is NPY_LONG on Linux and NPY_LONGLONG on Windows, and a caller
// can hand over either spelling on either platform.
static index_kind classify_index_type(int npy_type, std::int64_t itemsize) {
    const bool is_signed = npy_type == NPY_INT || npy_type == NPY_LONG || npy_type == NPY_LONGLONG;
    const bool is_unsigned =
        npy_type == NPY_UINT || npy_type == NPY_ULONG || npy_type == NPY_ULONGLONG;
    if (itemsize == 4) {
        return is_signed ? index_kind::int32
                         : (is_unsigned ? index_kind::uint32 : index_kind::unsupported);
    }
    if (itemsize == 8) {
        return is_signed ? index_kind::int64
                         : (is_unsigned ? index_kind::uint64 : index_kind::unsupported);
    }
    return index_kind::unsupported;
}

// Build the base-1 index array oneDAL's csr_table expects, reading the caller's
// buffer in whatever integer dtype and stride it arrives in.
//
// Casting to a fixed dtype first - which is what this used to do - allocates
// twice per index array: once for the cast array, whose only purpose is to be
// read once and discarded, and once for the dal::array that outlives the call.
// scipy stores `indices` and `indptr` as int32 whenever the matrix fits in 32
// bits, so that was the common path, not a corner case: for nnz non-zeros it
// cost an extra 8*nnz bytes of peak allocation and two extra passes over them.
static dal::array<std::int64_t> make_one_based_indices(PyObject *py_indices) {
    PyArrayObject *np_indices = reinterpret_cast<PyArrayObject *>(py_indices);

    // The loops below read the source dtype directly, so they need native byte
    // order and a width they can widen. Anything else - a byte-swapped array,
    // or an index dtype scipy does not produce, such as int16 - goes through
    // numpy's own cast and then takes the fast path on the result, which is
    // native, contiguous int64 by construction. Recursion is therefore one
    // level deep.
    const std::int64_t itemsize = static_cast<std::int64_t>(array_type_sizeof(np_indices));
    const index_kind kind = classify_index_type(static_cast<int>(array_type(np_indices)), itemsize);
    if (!array_is_native(np_indices) || kind == index_kind::unsupported) {
        // No WRITEABLE in the flags: these buffers are only ever read, and
        // asking for writeability copies a read-only input for nothing.
        py::object cast_indices = py::reinterpret_steal<py::object>(
            PyArray_FROMANY(py_indices,
                            NPY_INT64,
                            0,
                            0,
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST));
        if (!cast_indices) {
            throw std::invalid_argument(
                "[convert_to_table] Could not convert csr_matrix indices to 64-bit integers.");
        }
        return make_one_based_indices(cast_indices.ptr());
    }

    const std::int64_t count = static_cast<std::int64_t>(array_size(np_indices, 0));
    auto one_based = dal::array<std::int64_t>::empty(count);
    std::int64_t *const destination = one_based.get_mutable_data();
    const char *const source = static_cast<const char *>(array_data(np_indices));
    const std::int64_t stride = static_cast<std::int64_t>(array_stride(np_indices, 0));

    // Whether the elements can be read as the source dtype rather than byte by
    // byte. NumPy's ALIGNED flag is computed over the data pointer and the
    // strides, so it is what makes a typed dereference well defined for every
    // element; the stride test is what makes `stride / itemsize` below exact,
    // which NumPy does not guarantee on its own for arrays of one element or
    // fewer. `itemsize` is 4 or 8 past the guard above, so it cannot be zero
    // here - but the guard has to stay ahead of the division for that to hold.
    const bool readable_as_source = array_is_aligned(np_indices) && stride % itemsize == 0;

    switch (kind) {
        case index_kind::int32:
            rebase_indices<std::int32_t>(source, stride, readable_as_source, destination, count);
            break;
        case index_kind::uint32:
            rebase_indices<std::uint32_t>(source, stride, readable_as_source, destination, count);
            break;
        case index_kind::int64:
            rebase_indices<std::int64_t>(source, stride, readable_as_source, destination, count);
            break;
        default:
            // classify_index_type() admits nothing else past the guard above.
            rebase_indices<std::uint64_t>(source, stride, readable_as_source, destination, count);
            break;
    }
    return one_based;
}

template <typename T>
inline csr_table_t convert_to_csr_impl(PyObject *py_data,
                                       PyObject *py_column_indices,
                                       PyObject *py_row_indices,
                                       std::int64_t row_count,
                                       std::int64_t column_count) {
    PyArrayObject *np_data = reinterpret_cast<PyArrayObject *>(py_data);

    auto row_indices_one_based = make_one_based_indices(py_row_indices);
    auto column_indices_one_based = make_one_based_indices(py_column_indices);

    const T *data_pointer = static_cast<T *>(array_data(np_data));
    const std::int64_t data_count = static_cast<std::int64_t>(array_size(np_data, 0));

    // Only the data buffer is borrowed; make_one_based_indices() rebased the
    // index arrays into dal::arrays it allocated itself.
    return csr_table_t(
        dal::array<T>(
            data_pointer,
            data_count,
            [owner = make_python_owner(reinterpret_cast<PyObject *>(np_data))](const T *) {}),
        column_indices_one_based,
        row_indices_one_based,
#if ONEDAL_VERSION <= 20230100
        // row_count parameter is present in csr_table's constructor only in older versions of oneDAL
        row_count,
#endif
        column_count);
}

dal::table convert_to_table(py::object inp_obj,
                            py::object queue,
                            bool recursed,
                            bool require_sparse_with_sorted_indices) {
    dal::table res;

    PyObject *obj = inp_obj.ptr();

    if (obj == nullptr || obj == Py_None) {
        return res;
    }

#ifdef ONEDAL_DATA_PARALLEL
    if (!queue.is(py::none()) && !queue.attr("sycl_device").attr("has_aspect_fp64").cast<bool>() &&
        hasattr(inp_obj, "dtype")) {
        // If the queue exists, doesn't have the fp64 aspect, and the data is float64
        // then cast it to float32
        int type = reinterpret_cast<PyArray_Descr *>(inp_obj.attr("dtype").ptr())->type_num;
        if (type == NPY_DOUBLE || type == NPY_DOUBLELTR) {
            // use astype instead of PyArray_Cast in order to support scipy sparse inputs
            if (!recursed) {
                PyErr_WarnEx(
                    PyExc_RuntimeWarning,
                    "Data will be converted into float32 from float64 because device does not support it",
                    1);
                inp_obj = inp_obj.attr("astype")(py::dtype::of<float>());
                res = convert_to_table(inp_obj, queue, true);
            }
            else {
                throw std::invalid_argument(
                    "[convert_to_table] Numpy input could not be converted into onedal table.");
            }
            return res;
        }
    }
#endif // ONEDAL_DATA_PARALLEL

    if (is_array(obj)) {
        PyArrayObject *ary = reinterpret_cast<PyArrayObject *>(obj);

        if (!PyArray_ISCARRAY_RO(ary) && !PyArray_ISFARRAY_RO(ary)) {
            // NOTE: this will make a C-contiguous deep copy of the data
            // this is expected to be a special case
            obj = reinterpret_cast<PyObject *>(PyArray_GETCONTIGUOUS(ary));
            if (obj && !recursed) {
                res = convert_to_table(py::cast<py::object>(obj), queue, true);
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
                        throw py::type_error("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
    }
    else if (strcmp(Py_TYPE(obj)->tp_name, "csr_matrix") == 0 ||
             strcmp(Py_TYPE(obj)->tp_name, "csr_array") == 0) {
        if (require_sparse_with_sorted_indices) {
            if (!py::getattr(obj, "has_sorted_indices").cast<bool>()) {
                py::reinterpret_borrow<py::object>(obj).attr("sort_indices")();
            }
        }
        // py::getattr returns an owning py::object and raises Python's own
        // AttributeError if the lookup fails, so no reference bookkeeping or
        // null check is needed here.
        py::object py_data = py::getattr(obj, "data");
        py::object py_column_indices = py::getattr(obj, "indices");
        py::object py_row_indices = py::getattr(obj, "indptr");
        py::object py_shape = py::getattr(obj, "shape");
        if (!(is_array(py_data.ptr()) && is_array(py_column_indices.ptr()) &&
              is_array(py_row_indices.ptr()) && array_numdims(py_data.ptr()) == 1 &&
              array_numdims(py_column_indices.ptr()) == 1 &&
              array_numdims(py_row_indices.ptr()) == 1)) {
            throw std::invalid_argument("[convert_to_table] Got invalid csr_matrix object.");
        }
        py::object np_data = py::reinterpret_steal<py::object>(
            PyArray_FROMANY(py_data.ptr(), array_type(py_data.ptr()), 0, 0, NPY_ARRAY_CARRAY));
        // The index arrays are handed over as they are. make_one_based_indices()
        // reads whatever integer dtype and stride they carry straight into the
        // base-1 dal::array it allocates, so there is no cast array to create
        // here and immediately throw away.

        PyObject *np_row_count = PyTuple_GetItem(py_shape.ptr(), 0);
        PyObject *np_column_count = PyTuple_GetItem(py_shape.ptr(), 1);
        if (!(np_data && np_row_count && np_column_count)) {
            throw std::invalid_argument(
                "[convert_to_table] Failed accessing csr data when converting csr_matrix.\n");
        }

        const std::int64_t row_count = static_cast<std::int64_t>(PyLong_AsSsize_t(np_row_count));
        const std::int64_t column_count =
            static_cast<std::int64_t>(PyLong_AsSsize_t(np_column_count));

#define MAKE_CSR_TABLE(CType)                                 \
    res = convert_to_csr_impl<CType>(np_data.ptr(),           \
                                     py_column_indices.ptr(), \
                                     py_row_indices.ptr(),    \
                                     row_count,               \
                                     column_count);
        SET_NPY_FEATURE(array_type(np_data.ptr()),
                        array_type_sizeof(np_data.ptr()),
                        MAKE_CSR_TABLE,
                        throw py::type_error("Found unsupported data type in csr_matrix"));
#undef MAKE_CSR_TABLE
    }
    else {
        throw std::invalid_argument(
            "[convert_to_table] Not available input format for convert Python object to onedal table.");
    }
    return res;
}

template <class T>
void free_capsule(PyObject *cap) {
    // TODO: check safe cast
    dal::array<T> *stored_array = static_cast<dal::array<T> *>(PyCapsule_GetPointer(cap, NULL));
    if (stored_array) {
        delete stored_array;
    }
}

template <int NpType, typename T = byte_t>
static PyObject *convert_to_numpy_impl(
    const dal::array<T> &array,
    std::int64_t row_count,
    std::int64_t column_count = 0,
    const dal::data_layout &layout = dal::data_layout::row_major) {
    const int size_dims = column_count == 0 ? 1 : 2;
    npy_intp dims[2] = { static_cast<npy_intp>(row_count), static_cast<npy_intp>(column_count) };

    auto host_array = transfer_to_host(array);
    host_array.need_mutable_data();
    auto *bytes = host_array.get_mutable_data();
    // assumes that the array has writeable data (not clear if that is the case in oneDAL)
    int flags = layout == dal::data_layout::row_major ? NPY_ARRAY_CARRAY : NPY_ARRAY_FARRAY;
    PyObject *obj = PyArray_New(&PyArray_Type,
                                size_dims,
                                dims,
                                NpType,
                                NULL,
                                static_cast<void *>(bytes),
                                0,
                                flags,
                                NULL);
    if (!obj)
        throw std::invalid_argument("Conversion to numpy array failed");

    void *opaque_value = static_cast<void *>(new dal::array<T>(host_array));
    PyObject *cap = PyCapsule_New(opaque_value, NULL, free_capsule<T>);
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
    const std::int64_t rows_indices_count = table.get_row_count() + 1;
    const std::int64_t non_zero_count = table.get_non_zero_count();
    const std::int64_t *row_offsets = table.get_row_offsets();
    const std::int64_t *column_indices = table.get_column_indices();

    std::uint64_t *column_indices_zero_based_data = nullptr;
    std::uint64_t *row_offsets_zero_based_data = nullptr;

    dal::array<std::uint64_t> column_indices_zero_based_array;
    dal::array<std::uint64_t> row_offsets_zero_based_array;

    if (table.get_indexing() == sparse_indexing::zero_based) {
        column_indices_zero_based_data =
            const_cast<std::uint64_t *>(reinterpret_cast<const std::uint64_t *>(column_indices));
        row_offsets_zero_based_data =
            const_cast<std::uint64_t *>(reinterpret_cast<const std::uint64_t *>(row_offsets));

        column_indices_zero_based_array =
            dal::array<std::uint64_t>::wrap(column_indices_zero_based_data, non_zero_count);
        row_offsets_zero_based_array =
            dal::array<std::uint64_t>::wrap(row_offsets_zero_based_data, rows_indices_count);
    }
    else { // table.get_indexing() == sparse_indexing::one_based
        column_indices_zero_based_array = dal::array<std::uint64_t>::empty(non_zero_count);
        row_offsets_zero_based_array = dal::array<std::uint64_t>::empty(rows_indices_count);

        column_indices_zero_based_data = column_indices_zero_based_array.get_mutable_data();
        row_offsets_zero_based_data = row_offsets_zero_based_array.get_mutable_data();

        for (std::int64_t i = 0; i < non_zero_count; ++i)
            column_indices_zero_based_data[i] = column_indices[i] - 1;

        for (std::int64_t i = 0; i < rows_indices_count; ++i)
            row_offsets_zero_based_data[i] = row_offsets[i] - 1;
    }

    const T *data = table.get_data<T>();
    auto data_array = dal::array<T>::wrap(data, non_zero_count);

    PyObject *py_data = convert_to_numpy_impl<NpType, T>(data_array, non_zero_count);

    PyObject *py_col =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(column_indices_zero_based_array,
                                                         non_zero_count);
    PyObject *py_row =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(row_offsets_zero_based_array,
                                                         rows_indices_count);
    PyObject *result = PyTuple_New(3);
    PyTuple_SetItem(result, 0, py_data);
    PyTuple_SetItem(result, 1, py_col);
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

#define MAKE_NYMPY_FROM_HOMOGEN(NpType)                                       \
    {                                                                         \
        auto bytes_array = dal::detail::get_original_data(homogen_input);     \
        res = convert_to_numpy_impl<NpType>(bytes_array,                      \
                                            homogen_input.get_row_count(),    \
                                            homogen_input.get_column_count(), \
                                            homogen_input.get_data_layout()); \
    }
        SET_CTYPE_NPY_FROM_DAL_TYPE(dtype,
                                    MAKE_NYMPY_FROM_HOMOGEN,
                                    throw std::invalid_argument("Unable to convert numpy object"));
#undef MAKE_NYMPY_FROM_HOMOGEN
    }
    else if (input.get_kind() == csr_table_t::kind()) {
        const auto &csr_input = static_cast<const csr_table_t &>(input);
        const dal::data_type dtype = csr_input.get_metadata().get_data_type(0);
#define MAKE_PY_FROM_CSR(NpType, T)                              \
    {                                                            \
        res = convert_to_py_from_csr_impl<NpType, T>(csr_input); \
    }
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

} // namespace oneapi::dal::python::numpy
