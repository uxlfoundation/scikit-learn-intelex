/*******************************************************************************
* Copyright 2014 Intel Corporation
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
#include <limits>
#include <Python.h>
#include "daal4py.h"
#include "npy4daal.h"
#include "daal4py_defines.h"

#if NPY_ABI_VERSION < 0x02000000
    #define PyDataType_NAMES(descr) ((descr)->names)
#endif

// ************************************************************************************
// ************************************************************************************
// Numpy type conversion code, taken from numpy.i (SWIG typemap-code)
// ************************************************************************************
// ************************************************************************************

#define is_array(a)           ((a) && PyArray_Check(a))
#define array_type(a)         PyArray_TYPE((PyArrayObject *)a)
#define array_is_behaved_C(a) (PyArray_ISCARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_behaved_F(a) (PyArray_ISFARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_native(a)    (PyArray_ISNOTSWAPPED((PyArrayObject *)a))
#define array_numdims(a)      PyArray_NDIM((PyArrayObject *)a)
#define array_data(a)         PyArray_DATA((PyArrayObject *)a)
#define array_size(a, i)      PyArray_DIM((PyArrayObject *)a, i)

// ************************************************************************************
// ************************************************************************************

class NumpyDeleter : public daal::services::DeleterIface
{
public:
    // constructor to initialize with ndarray
    NumpyDeleter(PyArrayObject * a) : _ndarray(a) {}
    // DeleterIface must be copy-constructible
    NumpyDeleter(const NumpyDeleter & o) : _ndarray(o._ndarray) {}
    // ref-count reached 0 -> decref reference to python object
    void operator()(const void * ptr) DAAL_C11_OVERRIDE
    {
        // We need to protect calls to python API
        // Note: at termination time, even when no threads are running, this breaks without the protection
        PyGILState_STATE gstate = PyGILState_Ensure();
        assert(static_cast<void *>(array_data(_ndarray)) == ptr);
        Py_DECREF(_ndarray);
        PyGILState_Release(gstate);
    }
    // We don't want this to be copied
    NumpyDeleter & operator=(const NumpyDeleter &) = delete;

private:
    PyArrayObject * _ndarray;
};

// define our own free functions for wrapping python objects holding our shared pointers
void daalsp_free_cap(PyObject * cap)
{
    VSP * sp = static_cast<VSP *>(PyCapsule_GetPointer(cap, NULL));
    if (sp)
    {
        delete sp;
        sp = NULL;
    }
}

// define our own free functions for wrapping python objects holding our raw pointers
void rawp_free_cap(PyObject * cap)
{
    void * rp = PyCapsule_GetPointer(cap, NULL);
    if (rp)
    {
        delete[] rp;
        rp = NULL;
    }
}

void set_rawp_base(PyArrayObject * ary, void * ptr)
{
    PyObject * cap = PyCapsule_New(ptr, NULL, rawp_free_cap);
    PyArray_SetBaseObject(ary, cap);
}

inline void py_err_check()
{
    if (PyErr_Occurred())
    {
        PyErr_Print();
        throw std::runtime_error("Python error");
    }
}

// *****************************************************************************

// Uses a shared pointer to a raw array (T*) for creating a nd-array
template <typename T, int NPTYPE>
static PyObject * _sp_to_nda(daal::services::SharedPtr<T> & sp, size_t nr, size_t nc)
{
    DAAL4PY_CHECK_BAD_CAST(nc <= std::numeric_limits<int>::max());
    DAAL4PY_CHECK_BAD_CAST(nr <= std::numeric_limits<int>::max());
    npy_intp dims[2] = { static_cast<npy_intp>(nr), static_cast<npy_intp>(nc) };
    PyObject * obj   = PyArray_SimpleNewFromData(2, dims, NPTYPE, static_cast<void *>(sp.get()));
    if (!obj) throw std::invalid_argument("conversion to numpy array failed");
    set_sp_base(reinterpret_cast<PyArrayObject *>(obj), sp);
    return obj;
}

// get a block of Rows from NT and then create nd-array from it
// oneDAL potentially makes a copy when creating the BlockDesriptor
template <typename T, int NPTYPE>
static PyObject * _make_nda_from_bd(daal::data_management::NumericTablePtr * ptr)
{
    daal::data_management::BlockDescriptor<T> block;
    (*ptr)->getBlockOfRows(0, (*ptr)->getNumberOfRows(), daal::data_management::readOnly, block);
    if (block.getNumberOfRows() != (*ptr)->getNumberOfRows() || block.getNumberOfColumns() != (*ptr)->getNumberOfColumns())
    {
        std::cerr << "Getting data ptr as block-of-rows failed.\nExpected shape: " << (*ptr)->getNumberOfRows() << "x" << (*ptr)->getNumberOfColumns() << "\nBlock shape:" << block.getNumberOfRows() << "x" << block.getNumberOfColumns() << std::endl;
        return NULL;
    }
    daal::services::SharedPtr<T> data_tmp = block.getBlockSharedPtr();
    if (!data_tmp)
    {
        std::cerr << "Unexpected null pointer from block descriptor.";
        return NULL;
    }
    return _sp_to_nda<T, NPTYPE>(data_tmp, block.getNumberOfRows(), block.getNumberOfColumns());
}

// Most efficient conversion if NT is a HomogenNumericTable
// We do not need to make a copy and can use the raw data pointer directly.
template <typename T, int NPTYPE>
static PyObject * _make_nda_from_homogen(daal::data_management::NumericTablePtr * ptr)
{
    auto dptr = dynamic_cast<daal::data_management::HomogenNumericTable<T> *>((*ptr).get());
    if (dptr)
    {
        daal::services::SharedPtr<T> data_tmp(dptr->getArraySharedPtr());
        return _sp_to_nda<T, NPTYPE>(data_tmp, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns());
    }
    return NULL;
}

template <typename T, int NPTYPE>
static PyObject * _make_npy_from_data(T * data, size_t n)
{
    DAAL4PY_CHECK_BAD_CAST(n <= std::numeric_limits<int>::max());
    npy_intp dims[1] = { static_cast<npy_intp>(n) };
    PyObject * obj   = PyArray_SimpleNewFromData(1, dims, NPTYPE, static_cast<void *>(data));
    if (!obj) throw std::invalid_argument("conversion to numpy array failed");
    return obj;
}

template <typename T, int NPTYPE>
static PyObject * _make_nda_from_csr(daal::data_management::NumericTablePtr * ptr)
{
    daal::data_management::CSRNumericTable * csr_ptr = dynamic_cast<daal::data_management::CSRNumericTable *>(const_cast<daal::data_management::NumericTable *>((*ptr).get()));
    if (csr_ptr)
    {
        T * data_ptr;
        size_t * col_indices_ptr;
        size_t * row_offsets_ptr;
        csr_ptr->getArrays<T>(&data_ptr, &col_indices_ptr, &row_offsets_ptr);
        size_t n      = csr_ptr->getDataSize();
        T * data_copy = static_cast<T *>(daal::services::daal_malloc(n * sizeof(T)));
        DAAL4PY_CHECK_MALLOC(data_copy);
        daal::services::internal::daal_memcpy_s(data_ptr, sizeof(T) * n, data_copy, sizeof(T) * n);
        PyObject * py_data        = _make_npy_from_data<T, NPTYPE>(data_copy, n);
        n                         = csr_ptr->getNumberOfColumns();
        size_t * col_indices_copy = static_cast<size_t *>(daal::services::daal_malloc(n * sizeof(size_t)));
        DAAL4PY_CHECK_MALLOC(col_indices_copy);
        for (size_t i = 0; i < n; ++i)
        {
            col_indices_copy[i] = col_indices_ptr[i] - 1;
        }
        PyObject * py_col         = _make_npy_from_data<size_t, NPTYPE>(col_indices_copy, n);
        n                         = csr_ptr->getNumberOfRows();
        size_t * row_offsets_copy = static_cast<size_t *>(daal::services::daal_malloc(n * sizeof(size_t)));
        DAAL4PY_CHECK_MALLOC(row_offsets_copy);
        for (size_t i = 0; i < n; ++i)
        {
            row_offsets_copy[i] = row_offsets_ptr[i] - 1;
        }
        PyObject * py_row = _make_npy_from_data<size_t, NPTYPE>(row_offsets_copy, n);
        PyObject * result = PyTuple_New(3);
        PyTuple_SetItem(result, 0, py_data);
        PyTuple_SetItem(result, 1, py_col);
        PyTuple_SetItem(result, 2, py_row);
        return result;
    }
    return NULL;
}

// Convert a oneDAL NT to a numpy nd-array
// tries to avoid copying the data, instead we try to share the memory with DAAL
PyObject * make_nda(daal::data_management::NumericTablePtr * ptr)
{
    if (!ptr || !(*ptr).get() || (*ptr)->getNumberOfRows() == 0 || (*ptr)->getNumberOfRows() == 0) return Py_None;

    PyObject * res = NULL;

    // Try to convert from homogen/dense type as given in first column of NT
    // first try HomogenNT, then via BlockDescriptor. The latter requires a copy.
    switch ((*(*ptr)->getDictionary())[0].indexType)
    {
    case daal::data_management::data_feature_utils::DAAL_FLOAT64:
        if ((res = _make_nda_from_homogen<double, NPY_FLOAT64>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_bd<double, NPY_FLOAT64>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_csr<double, NPY_FLOAT64>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_FLOAT32:
        if ((res = _make_nda_from_homogen<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_bd<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_csr<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT32_S:
        if ((res = _make_nda_from_homogen<int32_t, NPY_INT32>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_csr<int32_t, NPY_INT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT32_U:
        if ((res = _make_nda_from_homogen<uint32_t, NPY_UINT32>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_csr<uint32_t, NPY_UINT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT64_S:
        if ((res = _make_nda_from_homogen<int64_t, NPY_INT64>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_csr<int64_t, NPY_INT64>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT64_U:
        if ((res = _make_nda_from_homogen<uint64_t, NPY_UINT64>(ptr)) != NULL) return res;
        if ((res = _make_nda_from_csr<uint64_t, NPY_UINT64>(ptr)) != NULL) return res;
        break;
    }
    // Falling back to using block-desriptors and converting to double
    if ((res = _make_nda_from_bd<double, NPY_FLOAT64>(ptr)) != NULL) return res;

    throw std::invalid_argument("Got unsupported table type.");
}

// Create a list of numpy arrays
extern PyObject * make_nda(daal::data_management::DataCollectionPtr * coll)
{
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    if (!coll) return Py_None;
    auto n          = (*coll)->size();
    PyObject * list = PyList_New(n);
    for (auto i = 0; i < n; ++i)
    {
        daal::data_management::NumericTablePtr nt = daal::services::dynamicPointerCast<daal::data_management::NumericTable>((*coll)->get(i));
        PyList_SetItem(list, i, make_nda(&nt));
        py_err_check();
    }
    return list;
}

extern PyObject * make_nda(daal::data_management::KeyValueDataCollectionPtr * dict, const i2str_map_t & id2str)
{
    PyObject * pydict = PyDict_New();
    for (size_t i = 0; i < (*dict)->size(); ++i)
    {
        auto elem = (*dict)->getValueByIndex(i);
        auto tbl  = daal::services::dynamicPointerCast<daal::data_management::NumericTable>(elem);
        // There can be NULL elements in collection
        if (tbl || !elem)
        {
            PyObject * obj = tbl ? make_nda(&tbl) : Py_None;
            size_t key     = (*dict)->getKeyByIndex(i);
            auto strkey    = id2str.find(key);
            if (strkey != id2str.end())
            {
                PyObject * keyobj = PyString_FromString(strkey->second.c_str());
                PyDict_SetItem(pydict, keyobj, obj);
                Py_DECREF(keyobj);
            }
            else
            {
                Py_DECREF(pydict);
                throw std::invalid_argument(std::string("Unexpected key '") + std::to_string(key) + "' found in KeyValueDataCollectionPtr\n");
            }
        }
        else
        {
            Py_DECREF(pydict);
            throw std::invalid_argument("Unexpected object found in KeyValueDataCollectionPtr, expected NULL or NumericTable\n");
        }
    }
    return pydict;
}

template <typename T>
static daal::data_management::NumericTablePtr _make_hnt(PyObject * nda)
{
    daal::data_management::NumericTablePtr ptr;
    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(nda);

    assert(is_array(nda) && array_is_behaved_C(array));

    if (array_numdims(array) == 2)
    {
        // we provide the SharedPtr with a deleter which decrements the pyref
        ptr = daal::data_management::HomogenNumericTable<T>::create(daal::services::SharedPtr<T>(reinterpret_cast<T *>(array_data(array)), NumpyDeleter(array)), static_cast<size_t>(array_size(array, 1)), static_cast<size_t>(array_size(array, 0)));
        // we need it increment the ref-count if we use the input array in-place
        // if we copied/converted it we already own our own reference
        if (reinterpret_cast<PyObject *>(array) == nda) Py_INCREF(array);
    }
    else
    {
        throw std::invalid_argument("Input array has wrong dimensionality (must be 2d).\n");
    }

    return ptr;
}

static daal::data_management::NumericTablePtr _make_npynt(PyObject * nda)
{
    daal::data_management::NumericTable * ptr = NULL;

    assert(is_array(nda));

    PyArrayObject * array = reinterpret_cast<PyArrayObject *>(nda);
    if (array_numdims(array) == 2)
    {
        // the given numpy array is not well behaved C array but has right dimensionality
        try
        {
            ptr = new NpyNumericTable<NpyNonContigHandler>(array);
        }
        catch (...)
        {
            ptr = NULL;
        }
    }
    else if (array_numdims(nda) == 1)
    {
        PyArray_Descr * descr = PyArray_DESCR(array);
        if (PyDataType_NAMES(descr))
        {
            // the given array is a structured numpy array.
            ptr = new NpyNumericTable<NpyStructHandler>(array);
        }
        else
        {
            throw std::invalid_argument("Input array is neither well behaved and nor a structured array.\n");
        }
    }
    else
    {
        throw std::invalid_argument("Input array has wrong dimensionality (must be 2d).\n");
    }

    return daal::data_management::NumericTablePtr(ptr);
}

// Try to convert given object to oneDAL Table without copying. Currently supports
// * numpy contiguous, homogenous -> oneDAL HomogenNumericTable
// * numpy non-contiguous, homogenous -> NpyNumericTable
// * numpy structured, heterogenous -> NpyNumericTable
// * list of arrays, heterogen -> oneDAL SOANumericTable
// * scipy csr_matrix -> oneDAL CSRNumericTable
//   As long as oneDAL CSR is only 0-based we need to copy indices/offsets
daal::data_management::NumericTablePtr make_nt(PyObject * obj)
{
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    if (obj && obj != Py_None)
    {
        if (PyObject_HasAttrString(obj, "__2daalnt__"))
        {
            static daal::data_management::NumericTablePtr ntptr;
            if (true || !ntptr)
            {
                // special protocol assumes that python objects implement __2daalnt__
                // returning a pointer to a NumericTablePtr, we have to delete the shared-pointer
                PyObject * _obj = PyObject_CallMethod(obj, "__2daalnt__", NULL);
                py_err_check();
                void * _ptr = PyCapsule_GetPointer(_obj, NULL);
                py_err_check();
                Py_DECREF(_obj);
                auto nt = reinterpret_cast<daal::data_management::NumericTablePtr *>(_ptr);
                ntptr   = *nt;
                delete nt; // we delete the shared pointer-pointer
                nt = NULL;
            }

            return ntptr;
        }
        daal::data_management::NumericTablePtr ptr;
        if (is_array(obj))
        { // we got a numpy array
            PyArrayObject * ary = reinterpret_cast<PyArrayObject *>(obj);

            if (array_is_behaved_C(ary))
            {
#define MAKENT_(_T) ptr = _make_hnt<_T>(obj)
                SET_NPY_FEATURE(PyArray_DESCR(ary)->type, MAKENT_, throw std::invalid_argument("Found unsupported array type"));
#undef MAKENT_
            }
            else
            {
                if (array_is_behaved_F(ary) && (PyArray_NDIM(ary) == 2))
                {
                    int _axes           = 0;
                    npy_intp N          = PyArray_DIM(ary, 1); // number of columns
                    npy_intp column_len = PyArray_DIM(ary, 0);
                    int ary_numtype     = PyArray_TYPE(ary);
                    /*
		     * Input is 2D F-contiguous array: represent it as SOA numeric table
		     */
                    daal::data_management::SOANumericTablePtr soatbl;

                    // iterate over columns
                    PyObject * it = PyArray_IterAllButAxis(obj, &_axes);
                    if (it == NULL)
                    {
                        Py_XDECREF(it);
                        throw std::runtime_error("Creating oneDAL SOA table from F-contigous NumPy array failed: iterator could not be created");
                    }

                    soatbl = daal::data_management::SOANumericTable::create(N, column_len);

                    for (npy_intp i = 0; PyArray_ITER_NOTDONE(it); ++i)
                    {
                        PyArrayObject * slice = reinterpret_cast<PyArrayObject *>(PyArray_SimpleNewFromData(1, &column_len, ary_numtype, static_cast<void *>(PyArray_ITER_DATA(it))));
                        PyArray_SetBaseObject(slice, reinterpret_cast<PyObject *>(ary));
                        Py_INCREF(ary);
#define SETARRAY_(_T)                                                                                         \
    {                                                                                                         \
        daal::services::SharedPtr<_T> _tmp(reinterpret_cast<_T *>(PyArray_DATA(slice)), NumpyDeleter(slice)); \
        soatbl->setArray(_tmp, i);                                                                            \
    }
                        SET_NPY_FEATURE(PyArray_DESCR(ary)->type, SETARRAY_, throw std::invalid_argument("Found unsupported array type"));
#undef SETARRAY_
                        PyArray_ITER_NEXT(it);
                    }
                    Py_DECREF(it);

                    if (soatbl->getNumberOfColumns() != N)
                    {
                        throw std::runtime_error("Creating oneDAL SOA table from F-contigous NumPy array failed.");
                    }
                    ptr = soatbl;
                }
                else
                    ptr = _make_npynt(obj);
            }

            if (!ptr) throw std::runtime_error("Could not convert Python object to oneDAL table.\n");
        }
        else if (PyList_Check(obj) && PyList_Size(obj) > 0)
        { // a list of arrays for SOA?
            PyObject * first = PyList_GetItem(obj, 0);
            py_err_check();

            if (is_array(first))
            { // can handle only list of 1d arrays
                auto N = PyList_Size(obj);
                daal::data_management::SOANumericTablePtr soatbl;

                for (auto i = 0; i < N; i++)
                {
                    PyArrayObject * ary = reinterpret_cast<PyArrayObject *>(PyList_GetItem(obj, i));
                    py_err_check();
                    if (i == 0) soatbl = daal::data_management::SOANumericTable::create(N, PyArray_DIM(ary, 0));
                    if (PyArray_NDIM(ary) != 1)
                    {
                        throw std::runtime_error(std::string("Found wrong dimensionality (") + std::to_string(PyArray_NDIM(ary)) + ") of array in list when constructing SOA table (must be 1d)");
                    }

                    if (!array_is_behaved_C(ary))
                    {
                        throw std::runtime_error(std::string("Cannot operate on column: ") + std::to_string(i) + "  because it is non-contiguous. Please make it contiguous before passing it to daal4py\n");
                    }

#define SETARRAY_(_T)                                                                                     \
    {                                                                                                     \
        daal::services::SharedPtr<_T> _tmp(reinterpret_cast<_T *>(PyArray_DATA(ary)), NumpyDeleter(ary)); \
        soatbl->setArray(_tmp, i);                                                                        \
    }
                    SET_NPY_FEATURE(PyArray_DESCR(ary)->type, SETARRAY_, throw std::invalid_argument("Found unsupported array type"));
#undef SETARRAY_
                    Py_INCREF(ary);
                }
                if (soatbl->getNumberOfColumns() != N)
                {
                    throw std::runtime_error("Creating oneDAL SOA table from list failed.");
                }
                ptr = soatbl;
            } // else not a list of 1d arrays
        }     // else not a list of 1d arrays
        if (!ptr && ((strcmp(Py_TYPE(obj)->tp_name, "csr_matrix") == 0) || (strcmp(Py_TYPE(obj)->tp_name, "csr_array") == 0)))
        {
            daal::services::SharedPtr<daal::data_management::CSRNumericTable> ret;
            PyObject * vals = PyObject_GetAttrString(obj, "data");
            py_err_check();
            PyObject * indcs = PyObject_GetAttrString(obj, "indices");
            py_err_check();
            PyObject * roffs = PyObject_GetAttrString(obj, "indptr");
            py_err_check();
            PyObject * shape = PyObject_GetAttrString(obj, "shape");
            py_err_check();

            if (shape && PyTuple_Check(shape) && is_array(vals) && is_array(indcs) && is_array(roffs) && array_numdims(vals) == 1 && array_numdims(indcs) == 1 && array_numdims(roffs) == 1)
            {
                py_err_check();

                // As long as oneDAL does not support 0-based indexing we have to copy the indices and add 1 to each
                PyObject * np_indcs = PyArray_FROMANY(indcs, NPY_UINT64, 0, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
                py_err_check();
                PyObject * np_roffs = PyArray_FROMANY(roffs, NPY_UINT64, 0, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
                py_err_check();
                PyObject * np_vals = PyArray_FROMANY(vals, array_type(vals), 0, 0, NPY_ARRAY_CARRAY);
                py_err_check();

                PyObject * nr = PyTuple_GetItem(shape, 0);
                py_err_check();
                PyObject * nc = PyTuple_GetItem(shape, 1);
                py_err_check();

                if (np_indcs && np_roffs && np_vals && nr && nc)
                {
                    size_t * c_indcs           = static_cast<size_t *>(array_data(np_indcs));
                    size_t n                   = array_size(np_indcs, 0);
                    size_t * c_indcs_one_based = static_cast<size_t *>(daal::services::daal_malloc(n * sizeof(size_t)));
                    DAAL4PY_CHECK_MALLOC(c_indcs_one_based);
                    for (size_t i = 0; i < n; ++i) c_indcs_one_based[i] = c_indcs[i] + 1;
                    size_t * c_roffs           = static_cast<size_t *>(array_data(np_roffs));
                    n                          = array_size(np_roffs, 0);
                    size_t * c_roffs_one_based = static_cast<size_t *>(daal::services::daal_malloc((n + 1) * sizeof(size_t)));
                    DAAL4PY_CHECK_MALLOC(c_roffs_one_based);
                    for (size_t i = 0; i < n; ++i) c_roffs_one_based[i] = c_roffs[i] + 1;
                    size_t c_nc = static_cast<size_t>(PyInt_AsSsize_t(nc));
                    py_err_check();
                    size_t c_nr = static_cast<size_t>(PyInt_AsSsize_t(nr));
                    py_err_check();
#define MKCSR_(_T) ret = daal::data_management::CSRNumericTable::create(daal::services::SharedPtr<_T>(reinterpret_cast<_T *>(array_data(np_vals)), NumpyDeleter(reinterpret_cast<PyArrayObject *>(np_vals))), daal::services::SharedPtr<size_t>(c_indcs_one_based, daal::services::ServiceDeleter()), daal::services::SharedPtr<size_t>(c_roffs_one_based, daal::services::ServiceDeleter()), c_nc, c_nr)
                    SET_NPY_FEATURE(array_type(np_vals), MKCSR_, throw std::invalid_argument(std::string("Found unsupported data type in ") + Py_TYPE(obj)->tp_name + "\n"));
#undef MKCSR_
                }
                else
                    throw std::invalid_argument(std::string("Failed accessing csr data when converting ") + Py_TYPE(obj)->tp_name + "\n");
                Py_DECREF(np_indcs);
                Py_DECREF(np_roffs);
            }
            else
                throw std::invalid_argument("Got invalid csr_matrix or csr_array.\n");
            Py_DECREF(shape);
            Py_DECREF(roffs);
            Py_DECREF(indcs);
            Py_DECREF(vals);
            return daal::data_management::NumericTablePtr(ret);
        }
        return ptr;
    }

    return daal::data_management::NumericTablePtr();
}

extern daal::data_management::KeyValueDataCollectionPtr make_dnt(PyObject * dict, str2i_map_t & str2id)
{
    daal::data_management::KeyValueDataCollectionPtr dc(new daal::data_management::KeyValueDataCollection);
    if (dict && dict != Py_None)
    {
        if (PyDict_Check(dict))
        {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(dict, &pos, &key, &value))
            {
                const char * strkey = PyString_AsString(key);
                auto keyid          = str2id.find(strkey);
                if (keyid != str2id.end())
                {
                    daal::data_management::NumericTablePtr tbl = make_nt(value);
                    if (tbl)
                    {
                        (*dc)[keyid->second] = daal::services::staticPointerCast<daal::data_management::SerializationIface>(tbl);
                    }
                    else
                    {
                        throw std::invalid_argument(std::string("Unexpected object '") + Py_TYPE(value)->tp_name + "' found in dict, expected an array\n");
                    }
                }
                else
                {
                    throw std::invalid_argument(std::string("Unexpected key '") + Py_TYPE(key)->tp_name + "' found in dict, expected a string\n");
                }
            }
        }
        else
        {
            throw std::invalid_argument(std::string("Unexpected object '") + Py_TYPE(dict)->tp_name + "' found, expected dict\n");
        }
    }
    return dc;
}

data_or_file::data_or_file(PyObject * input)
{
    this->table.reset();
    this->file.resize(0);
    if (input == Py_None)
    {
        ;
    }
    else if (PyUnicode_Check(input))
    {
        //        this->file = PyUnicode_AsUTF8AndSize(input, &size);
        this->file = PyUnicode_AsUTF8(input);
    }
    else
    {
        auto tmp = make_nt(input);
        if (tmp)
        {
            this->table = tmp;
        }
        if (!this->table)
        {
            throw std::invalid_argument(std::string("Got type '") + Py_TYPE(input)->tp_name + "' when expecting string, array, or list of 1d-arrays.");
        }
    }
}

const daal::data_management::NumericTablePtr get_table(const data_or_file & t)
{
    if (t.table) return t.table;
    if (t.file.size()) return readCSV(t.file);
    throw std::invalid_argument("one and only one input per process allowed");
    return daal::data_management::NumericTablePtr();
}

const daal::data_management::NumericTablePtr readCSV(const std::string & fname)
{
    daal::data_management::FileDataSource<daal::data_management::CSVFeatureManager> dataSource(fname, daal::data_management::DataSource::doAllocateNumericTable, daal::data_management::DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();
    return dataSource.getNumericTable();
}

extern "C" void to_c_array(const daal::data_management::NumericTablePtr * ptr, void ** data, size_t * dims, char dtype)
{
    *data = NULL;
    if (ptr && ptr->get())
    {
        dims[0] = (*ptr)->getNumberOfRows();
        dims[1] = (*ptr)->getNumberOfColumns();
        switch (dtype)
        {
        case 0: *data = get_nt_data_ptr<double>(ptr); break;
        case 1: *data = get_nt_data_ptr<float>(ptr); break;
        case 2: *data = get_nt_data_ptr<int>(ptr); break;
        default: throw std::invalid_argument("Invalid data type specified.");
        }
        if (*data) return;
        throw std::invalid_argument("Data type and table type are incompatible.");
    }
    // ptr==NULL: no input data
    dims[0] = dims[1] = 0;
    return;
}

daal::data_management::DataCollectionPtr make_datacoll(PyObject * input)
{
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    if (input && input != Py_None && PyList_Check(input) && PyList_Size(input) > 0)
    {
        auto n                                      = PyList_Size(input);
        daal::data_management::DataCollection * res = new daal::data_management::DataCollection;
        res->resize(n);
        for (auto i = 0; i < n; i++)
        {
            PyObject * obj = PyList_GetItem(input, i);
            py_err_check();
            auto tmp = make_nt(obj);
            if (tmp)
                res->push_back(tmp);
            else
                throw std::runtime_error(std::string("Unexpected object '") + Py_TYPE(obj)->tp_name + "' found in list, expected an array");
        }
        return daal::data_management::DataCollectionPtr(res);
    }
    return daal::data_management::DataCollectionPtr();
}

static int64_t getval_(const std::string & str, const str2i_map_t & strmap)
{
    auto i = strmap.find(str);
    if (i == strmap.end()) throw std::invalid_argument(std::string("Encountered unexpected string-identifier '") + str + std::string("'"));
    return i->second;
}

int64_t string2enum(const std::string & str, str2i_map_t & strmap)
{
    int64_t r = 0;
    std::size_t current, previous = 0;
    while ((current = str.find('|', previous)) != std::string::npos)
    {
        r |= getval_(str.substr(previous, current - previous), strmap);
        previous = current + 1;
    }
    return (r | getval_(str.substr(previous, current - previous), strmap));
}

#ifdef _DIST_
    #include "transceiver.h"
#endif

extern "C"
{
    void c_daalinit(int nthreads)
    {
        if (nthreads > 0) daal::services::Environment::getInstance()->setNumberOfThreads(nthreads);
    }

    void c_daalfini()
    {
#ifdef _DIST_
        del_transceiver();
#endif
    }

    size_t c_num_threads()
    {
        return daal::services::Environment::getInstance()->getNumberOfThreads();
    }

    size_t c_num_procs()
    {
#ifdef _DIST_
        return get_transceiver()->nMembers();
#else
        return 1;
#endif
    }

    size_t c_my_procid()
    {
#ifdef _DIST_
        return get_transceiver()->me();
#else
        return 0;
#endif
    }

    void c_enable_thread_pinning(bool enabled)
    {
        daal::services::Environment::getInstance()->enableThreadPinning(enabled);
    }
} // extern "C"

bool c_assert_all_finite(const data_or_file & t, bool allowNaN, char dtype)
{
    bool result = false;
    auto tab    = get_table(t);
    switch (dtype)
    {
    case 0: result = daal::data_management::internal::allValuesAreFinite<double>(*tab, allowNaN); break;
    case 1: result = daal::data_management::internal::allValuesAreFinite<float>(*tab, allowNaN); break;
    default: throw std::invalid_argument("Invalid data type specified.");
    }
    return result;
}

void c_train_test_split(data_or_file & orig, data_or_file & train, data_or_file & test, data_or_file & train_idx, data_or_file & test_idx)
{
    auto origTable     = get_table(orig);
    auto trainTable    = get_table(train);
    auto testTable     = get_table(test);
    auto trainIdxTable = get_table(train_idx);
    auto testIdxTable  = get_table(test_idx);
    daal::data_management::internal::trainTestSplit<int>(origTable, trainTable, testTable, trainIdxTable, testIdxTable);
}

double c_roc_auc_score(data_or_file & y_true, data_or_file & y_test)
{
#if __INTEL_DAAL__ >= 2021 && INTEL_DAAL_VERSION >= 20210200
    const size_t col_true = y_true.table->getNumberOfColumns();
    const size_t row_true = y_true.table->getNumberOfRows();
    const size_t col_test = y_test.table->getNumberOfColumns();
    const size_t row_test = y_test.table->getNumberOfRows();

    if (col_true != 1 || col_test != 1 || row_true != row_test)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown shape data");
        return NULL;
    }

    auto table_true = get_table(y_true);
    auto table_test = get_table(y_test);
    auto type       = (*table_test->getDictionary())[0].indexType;
    if (type == daal::data_management::data_feature_utils::DAAL_FLOAT64 || type == daal::data_management::data_feature_utils::DAAL_INT64_S || type == daal::data_management::data_feature_utils::DAAL_INT64_U || type == daal::data_management::data_feature_utils::DAAL_FLOAT32 || type == daal::data_management::data_feature_utils::DAAL_INT32_S || type == daal::data_management::data_feature_utils::DAAL_INT32_U)
    {
        return daal::data_management::internal::rocAucScore(table_true, table_test);
    }

    PyErr_SetString(PyExc_RuntimeError, "Unknown shape data");
    return 0.0;
#else
    return -1.0;
#endif
}

void c_generate_shuffled_indices(data_or_file & idx, data_or_file & random_state)
{
#if __INTEL_DAAL__ == 2020 && INTEL_DAAL_VERSION >= 20200003 || __INTEL_DAAL__ >= 2021
    auto idxTable         = get_table(idx);
    auto randomStateTable = get_table(random_state);
    daal::data_management::internal::generateShuffledIndices<int>(idxTable, randomStateTable);
#else
#endif
}

void c_tsne_gradient_descent(data_or_file & init, data_or_file & p, data_or_file & size_iter, data_or_file & params, data_or_file & results, char dtype)
{
#if __INTEL_DAAL__ >= 2021 && INTEL_DAAL_VERSION >= 20210600
    auto initTable                                     = get_table(init);
    auto pTable                                        = get_table(p);
    auto sizeIterTable                                 = get_table(size_iter);
    auto paramTable                                    = get_table(params);
    auto resultTable                                   = get_table(results);
    daal::data_management::CSRNumericTablePtr csrTable = daal::services::dynamicPointerCast<daal::data_management::CSRNumericTable, daal::data_management::NumericTable>(pTable);

    if (csrTable)
    {
        switch (dtype)
        {
        case 0: daal::algorithms::internal::tsneGradientDescent<int, double>(initTable, csrTable, sizeIterTable, paramTable, resultTable); break;
        case 1: daal::algorithms::internal::tsneGradientDescent<int, float>(initTable, csrTable, sizeIterTable, paramTable, resultTable); break;
        default: throw std::invalid_argument("Invalid data type specified.");
        }
    }
    else
        PyErr_SetString(PyExc_RuntimeError, "Unexpected table type");
#else
#endif
}
