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

#ifndef _HLAPI_H_INCLUDED_
#define _HLAPI_H_INCLUDED_

#ifdef _WIN32
    #define NOMINMAX
#endif

#include <daal.h>
using daal::step1Local;
using daal::step2Local;
using daal::step3Local;
using daal::step4Local;
using daal::step2Master;
using daal::step3Master;
using daal::step5Master;
using daal::services::LibraryVersionInfo;
#include "daal_compat.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>
#include <string>
#include <unordered_map>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define NTYPE PyObject *

#if PY_VERSION_HEX < 0x03000000
    #define PyUnicode_Check(_x)      PyString_Check(_x)
    #define PyUnicode_AsUTF8(_x)     PyString_AsString(_x)
    #define PyUnicode_FromString(_x) PyString_FromString(_x)
#endif

#include "data_management/data/internal/finiteness_checker.h"
#include "data_management/data/internal/train_test_split.h"

#if __INTEL_DAAL__ >= 2021 && INTEL_DAAL_VERSION >= 20210200
    #include "data_management/data/internal/roc_auc_score.h"
#endif
#if __INTEL_DAAL__ >= 2021 && INTEL_DAAL_VERSION >= 20210600
    #include "algorithms/tsne/tsne_gradient_descent.h"
#endif

extern "C"
{
    void c_daalinit(int nthreads = -1);
    void c_daalfini();
    size_t c_num_threads();
    size_t c_num_procs();
    size_t c_my_procid();
    void c_enable_thread_pinning(bool enabled = true);
}

using daal::data_management::NumericTablePtr;
typedef daal::services::SharedPtr<std::vector<std::vector<daal::byte> > > BytesArray;
typedef std::string std_string;
typedef std::unordered_map<std::string, int64_t> str2i_map_t;
typedef std::unordered_map<int64_t, std::string> i2str_map_t;

template <typename T>
bool use_default(const daal::services::SharedPtr<T> * attr)
{
    return attr == NULL || attr->get() == NULL;
}

template <typename T>
bool use_default(const daal::services::SharedPtr<T> & attr)
{
    return attr.get() == NULL;
}

template <typename T>
bool use_default(const T * attr)
{
    return attr == NULL;
}

inline bool use_default(const std::string & attr)
{
    return attr.length() == 0;
}

inline bool use_default(const int & attr)
{
    return attr == -1;
}

inline bool use_default(const size_t & attr)
{
    return static_cast<long>(attr) == static_cast<long>(-1);
}

#ifndef _WIN32
inline bool use_default(const DAAL_UINT64 & attr)
{
    return static_cast<long>(attr) == static_cast<long>(-1);
}
#endif

inline bool use_default(const double & attr)
{
    return attr != attr;
}

inline bool use_default(const float & attr)
{
    return attr != attr;
}

inline bool string2bool(const std::string & s)
{
    if (s == "True" || s == "true" || s == "1") return true;
    if (s == "False" || s == "false" || s == "0") return false;
    throw std::invalid_argument("Bool must be one of {'True', 'true', '1', 'False', 'false', '0'}");
}

class algo_manager__iface__
{
public:
    inline algo_manager__iface__() {}
    inline virtual ~algo_manager__iface__() {}
    // We don't want any manager to be copied
    algo_manager__iface__(const algo_manager__iface__ &)           = delete;
    algo_manager__iface__ operator=(const algo_manager__iface__ &) = delete;
};

#if 0
static inline NTYPE as_native_shared_ptr(services::SharedPtr< const algo_manager__iface__ > algo)
{
    int gc = 0;
    MK_DAALPTR(ret, new services::SharedPtr< const algo_manager__iface__ >(algo), services::SharedPtr< algo_manager__iface__ >, gc);
    TMGC(gc);
    return ret;
}
#endif

// Our Batch input/Output manager, abstracts from input/output types
// also defines how to get results and finalize
template <typename A, typename O>
struct IOManager
{
    typedef O result_type;

    static result_type getResult(A & algo) { return daal::services::staticPointerCast<typename result_type::ElementType>(algo.getResult()); }
    static bool needsFini() { return true; }
};

struct data_or_file
{
    mutable daal::data_management::NumericTablePtr table;
    std::string file;
    template <typename T>
    inline data_or_file(T * ptr, size_t ncols, size_t nrows, Py_ssize_t layout) : table(), file()
    {
        if (layout > 0) throw std::invalid_argument("Supporting only homogeneous, contiguous arrays.");
        table = daal::data_management::HomogenNumericTable<T>::create(ptr, ncols, nrows);
    }
    inline data_or_file() : table(), file() {}
    data_or_file(PyObject *);
};

// return input as oneDAL numeric table.
extern const daal::data_management::NumericTablePtr get_table(const data_or_file & t);

template <typename T>
struct RAW
{
    typedef T TYPE;
    const TYPE operator()(const T & o) { return o; }
};

template <typename T>
struct RAW<daal::services::SharedPtr<T> >
{
    typedef daal::services::SharedPtr<T> * TYPE;
    TYPE operator()(daal::services::SharedPtr<T> o) { return new daal::services::SharedPtr<T>(o); }
};

template <typename T>
T to_daal(T t)
{
    return t;
}
template <typename T>
daal::services::SharedPtr<T> to_daal(daal::services::SharedPtr<T> * t)
{
    return *t;
}
inline const data_or_file & to_daal(const data_or_file * t)
{
    return *t;
}
inline const data_or_file & to_daal(const data_or_file & t)
{
    return t;
}
inline const data_or_file & to_daal(data_or_file * t)
{
    return *t;
}

template <typename T>
void * get_nt_data_ptr(const daal::data_management::NumericTablePtr * ptr)
{
    auto dptr = dynamic_cast<const daal::data_management::HomogenNumericTable<T> *>((*ptr).get());
    return dptr ? reinterpret_cast<void *>(dptr->getArraySharedPtr().get()) : NULL;
}

extern int64_t string2enum(const std::string & str, str2i_map_t & strmap);

static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

inline static const double get_nan64()
{
    return std::numeric_limits<double>::quiet_NaN();
}

inline static const float get_nan32()
{
    return std::numeric_limits<float>::quiet_NaN();
}

typedef daal::data_management::DataCollectionPtr data_management_DataCollectionPtr;
typedef daal::data_management::NumericTablePtr data_management_NumericTablePtr;
typedef daal::data_management::DataCollectionPtr list_NumericTablePtr;
typedef daal::data_management::KeyValueDataCollectionPtr dict_NumericTablePtr;

extern "C" void to_c_array(const daal::data_management::NumericTablePtr * ptr, void ** data, size_t * dims, char dtype = 0);
extern PyObject * make_nda(daal::data_management::NumericTablePtr * nt_ptr);
extern PyObject * make_nda(daal::data_management::DataCollectionPtr * nt_ptr);
extern PyObject * make_nda(daal::data_management::KeyValueDataCollectionPtr * nt_ptr, const i2str_map_t &);
extern daal::data_management::NumericTablePtr make_nt(PyObject * nda);
extern daal::data_management::DataCollectionPtr make_datacoll(PyObject * nda);
extern daal::data_management::KeyValueDataCollectionPtr make_dnt(PyObject * dict, str2i_map_t &);

extern const daal::data_management::NumericTablePtr readCSV(const std::string & fname);

template <class T, class U>
T * dynamicPointerPtrCast(U * r)
{
    T tmp = daal::services::dynamicPointerCast<typename T::ElementType>(*r);
    return tmp ? new T(*reinterpret_cast<T *>(r)) : NULL;
}

template <typename T>
bool is_valid_ptrptr(T * o)
{
    return o != NULL && (*o).get() != NULL;
}

class ThreadAllow
{
    PyThreadState * _save;

public:
    ThreadAllow() { allow(); }
    ~ThreadAllow() { disallow(); }
    void allow() { _save = PyEval_SaveThread(); }
    void disallow()
    {
        if (_save)
        {
            PyEval_RestoreThread(_save);
            _save = NULL;
        }
    }
};

/* *********************************************************************** */

// An empty virtual base class (used by TVSP) for shared pointer handling
// we use this to have a generic type for all shared pointers
// e.g. used in daalsp_free functions below
class VSP
{
public:
    // we need a virtual destructor
    virtual ~VSP() {};
};
// typed virtual shared pointer, for simplicity we make it a oneDAL shared pointer
template <typename T>
class TVSP : public VSP, public daal::services::SharedPtr<T>
{
public:
    TVSP(const daal::services::SharedPtr<T> & org) : daal::services::SharedPtr<T>(org) {}
    virtual ~TVSP() {};
};

// define our own free functions for wrapping python objects holding our shared pointers
extern void daalsp_free_cap(PyObject *);
extern void rawp_free_cap(PyObject *);

template <typename T>
void set_sp_base(PyArrayObject * ary, daal::services::SharedPtr<T> & sp)
{
    void * tmp_sp  = static_cast<void *>(new TVSP<T>(sp));
    PyObject * cap = PyCapsule_New(tmp_sp, NULL, daalsp_free_cap);
    PyArray_SetBaseObject(ary, cap);
}

template <typename T>
static T * _daal_clone(const T & o)
{
    return new T(o);
}

extern "C"
{
    void set_rawp_base(PyArrayObject *, void *);
}

extern "C"
{
    bool c_assert_all_finite(const data_or_file & t, bool allowNaN, char dtype);
}

extern "C"
{
    void c_train_test_split(data_or_file & orig, data_or_file & train, data_or_file & test, data_or_file & train_idx, data_or_file & test_idx);
}

extern "C"
{
    double c_roc_auc_score(data_or_file & y_true, data_or_file & y_test);
}

extern "C"
{
    void c_generate_shuffled_indices(data_or_file & idx, data_or_file & random_state);
}

extern "C"
{
    void c_tsne_gradient_descent(data_or_file & init, data_or_file & p, data_or_file & size_iter, data_or_file & params, data_or_file & results, char dtype);
}

#endif // _HLAPI_H_INCLUDED_
