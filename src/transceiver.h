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

// Provides communication features on four levels
// 1. Fundamental, low-level communction (transceiver_base_iface)
// 2. Extended low-level communication (transceiver_iface)
// 3. Default implementations of 2. based on 1. (transceiver_impl)
// 4. Typed higher-level API
//
// Transceiver implementations must at least provide transceiver_base_iface.
//
// Initialization is done by calling get_transceiver(). In there the
// actual transceiver type is selected, and the object constructed and initialized.
// Repeated calls of get_transceiver() will not re-initialize.
//
// Transceiver implementations are supposed to be provided in a Python module.
// Such a module must provide "std::shared_pointer<transceiver_iface>*" in its attribute 'transceiver'.
// By putting this into a module we can keep the core of daal4py independent of the communication
// package, such as MPI. It also allows us to select the communication layer at runtime.
// The transceiver implementation can be selected by setting env var 'D4P_TRANSCEIVER' to the module name.
// The current default is 'mpi_transceiver'.

#ifndef _TRANSCEIVER_INCLUDED_
#define _TRANSCEIVER_INCLUDED_

#include <daal.h>
#include <memory>
#include <iostream>
#include <cassert>
#include "daal4py_defines.h"

// Abstract class with minimal functionality needed for communicating between processes.
class transceiver_base_iface
{
public:
    // initialize communication network
    virtual void init() = 0;

    // finalize communication network
    virtual void fini() = 0;

    // @return number of processes in network
    virtual size_t nMembers() = 0;

    // @return identifier of current process
    virtual size_t me() = 0;

    // send message to another process
    // @param[in] buff   bytes to send
    // @param[in] N      number of bytes to send
    // @param[in] recpnt id of recipient
    // @param[in] tag    message tag, to be matched by recipient
    virtual void send(const void * buff, size_t N, size_t recpnt, size_t tag) = 0;

    // receive a message from another process
    // @param[out] buff   buffer to store message in
    // @param[in]  N      size of buffer
    // @param[in]  recpnt id of sender
    // @param[in]  tag    message tag, to be matched with sender tag
    // @return number of received bytes
    virtual size_t recv(void * buff, size_t N, int sender, int tag) = 0;

    // virtual destructor
    virtual ~transceiver_base_iface() {}
};

// Abstract class with all functionality used for communicating between processes.
// Extends transceiver_base_iface with collective operations which can be implemented
// with functions from transceiver_base_iface (see transceiver_impl)
class transceiver_iface : public transceiver_base_iface
{
public:
    // Gather data from all processes on given root process
    // @param[in]  ptr    pointer to data the process contributes to the gather
    // @param[in]  N      number of bytes in ptr
    // @param[in]  root   process id which collects data
    // @param[in]  sizes  number of bytes constributed by each process, relevant on root only
    //                    Can be zero also on root if varying==false
    // @param[in] varying set to false to indicate all members provide same chunksize
    virtual void * gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying = true) = 0;

    // Broadcast data from root to all other processes
    // @param[inout] ptr   on root: pointer to data to be sent
    //                     on all other processes: pointer to buffer to store received data
    // @param[in]    N     number of bytes in ptr
    // @param[in]    root  process id which collects data
    virtual void bcast(void * ptr, size_t N, size_t root) = 0;

    // indicates data types for reductions
    enum type_type
    {
        BOOL,
        INT8,
        UINT8,
        INT32,
        UINT32,
        INT64,
        UINT64,
        FLOAT,
        DOUBLE
    };

    // indicates reduction operation
    enum operation_type
    {
        OP_MAX = 100,
        OP_MIN,
        OP_SUM,
        OP_PROD,
        OP_LAND,
        OP_BAND,
        OP_LOR,
        OP_BOR,
        OP_LXOR,
        OP_BXOR
    };

    // Element-wise reduce given array with given operation and provide result on all processes
    // @param[inout] inout input to reduction and result
    // @param[in]    T     data type of elements in inout
    // @param[in]    N     number of elements in inout
    // @param[in]    op    reduction operation
    virtual void reduce_all(void * inout, type_type T, size_t N, operation_type op) = 0;

    // Element-wise reduce given array partially with given operation
    // Each process will get result of reduction from processes [0:me[ (excluding me!)
    // @param[inout] inout input to reduction and result
    // @param[in]    T     data type of elements in inout
    // @param[in]    N     number of elements in inout
    // @param[in]    op    reduction operation
    virtual void reduce_exscan(void * inout, type_type T, size_t N, operation_type op) = 0;
};

// Default implementation for collective operations.
// tranceiver implementations should derive from here so they can provide
// optimized implementations for only some of the collectives.
// Also provides members m_me and m_nMembers to avoid frequent function calls.
class transceiver_impl : public transceiver_iface
{
public:
    transceiver_impl() : m_me(-1), m_nMembers(0), m_initialized(false) {}

    // implementations/derived classes must call this in their init()
    virtual void init()
    {
        if (!m_initialized)
        {
            m_me          = me();
            m_nMembers    = nMembers();
            m_initialized = true;
        }
    }

    virtual void * gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying) { throw std::logic_error("transceiver_base::gather not yet implemented"); }

    virtual void bcast(void * ptr, size_t N, size_t root) { throw std::logic_error("transceiver_base::bcast not yet implemented"); }

    virtual void reduce_all(void * inout, type_type T, size_t N, operation_type op) { throw std::logic_error("transceiver_base::reduce_all not yet implemented"); }

    virtual void reduce_exscan(void * inout, type_type T, size_t N, operation_type op) { throw std::logic_error("transceiver_base::reduce_exscan not yet implemented"); }

protected:
    bool m_initialized;
    size_t m_me;       // result of me()
    size_t m_nMembers; // result of nMembers()
};

// Higher-level, typ-safe transceiver abstraction.
// Provides features tailored for daal4py specifically
class transceiver
{
public:
    // @param[in] t actual transceiver object
    transceiver(const std::shared_ptr<transceiver_iface> & t) : m_transceiver(t)
    {
        m_transceiver->init();
        m_inited = true;
    }

    ~transceiver() { m_transceiver->fini(); }

    inline size_t nMembers() { return m_transceiver->nMembers(); }

    inline size_t me() { return m_transceiver->me(); }

    // Send object of given type to recpnt.
    // Object is assumed to be a daal::serializable object.
    // @param[in] obj    object to be sent
    // @param[in] recpnt recipient
    // @param[in] tag    message tag to be matched by recipient
    template <typename T>
    void send(const T & obj, size_t recpnt, size_t tag);

    // Receive an object of given type from sender
    // Object is assumed to be a daal::serializable object.
    // @param[in] sender sender
    // @param[in] tag    message tag to be matched with send
    template <typename T>
    T recv(size_t sender, size_t tag);

    // Gather objects stored in a shared pointer on given root process
    // Object is assumed to be a daal::serializable object.
    // @param[in]  sptr    shared pointer with object to be gathered
    // @param[in]  root    process id which collects data
    // @param[in]  varying can be set to false if objects are of identical size on all processes
    template <typename T>
    std::vector<daal::services::SharedPtr<T> > gather(const daal::services::SharedPtr<T> & sptr, size_t root = 0, bool varying = true);

    // Broadcast object from root to all other processes
    // Object is serialized similar to memcpy(buffer, &obj, sizeof(obj)).
    // Object is deserialized similar to memcpy(&obj, buffer, sizeof(obj)).
    // @param[inout] obj   on root: reference of object to be sent
    //                     on all other processes: reference of object to store received data
    // @param[in]    root  process id which collects data
    template <typename T>
    void bcast(T & obj, size_t root = 0);

    // Broadcast shared pointer object from root to all other processes
    // Object is assumed to be a daal::serializable object.
    // @param[inout] obj   on root: reference of shared pointer object to be sent
    //                     on all other processes: reference of shared pointer object to store received data
    // @param[in]    root  process id which collects data
    template <typename T>
    void bcast(daal::services::SharedPtr<T> & obj, size_t root = 0);

    // Element-wise reduce given array with given operation and provide result on all processes
    // Elements are serialized similar to memcpy(buffer, &obj, sizeof(obj)).
    // Elements are deserialized similar to memcpy(&obj, buffer, sizeof(obj)).
    // @param[inout] inout input to reduction and result
    // @param[in]    N     number of elements in inout
    // @param[in]    op    reduction operation
    template <typename T>
    void reduce_all(T * buf, size_t n, transceiver_iface::operation_type op);

    // Element-wise reduce given array partially with given operation
    // Each process will get result of reduction from processes [0:me[ (excluding me!)
    // Elements are serialized similar to memcpy(buffer, &obj, sizeof(obj)).
    // Elements are deserialized similar to memcpy(&obj, buffer, sizeof(obj)).
    // @param[inout] inout input to reduction and result
    // @param[in]    N     number of elements in inout
    // @param[in]    op    reduction operation
    template <typename T>
    void reduce_exscan(T * buf, size_t n, transceiver_iface::operation_type op);

protected:
    std::shared_ptr<transceiver_iface> m_transceiver; // the actual transceiver object
    bool m_inited;                                    // Initialization status
};

// @return the global transceiver object
// Repeated calls will not re-initialize.
extern transceiver * get_transceiver();
extern void del_transceiver();

template <typename T>
struct from_std;
template <>
struct from_std<double>
{
    static const transceiver_iface::type_type typ = transceiver_iface::DOUBLE;
};
template <>
struct from_std<float>
{
    static const transceiver_iface::type_type typ = transceiver_iface::FLOAT;
};
template <>
struct from_std<bool>
{
    static const transceiver_iface::type_type typ = transceiver_iface::BOOL;
};
template <>
struct from_std<int8_t>
{
    static const transceiver_iface::type_type typ = transceiver_iface::INT8;
};
template <>
struct from_std<uint8_t>
{
    static const transceiver_iface::type_type typ = transceiver_iface::UINT8;
};
template <>
struct from_std<int32_t>
{
    static const transceiver_iface::type_type typ = transceiver_iface::INT32;
};
template <>
struct from_std<uint32_t>
{
    static const transceiver_iface::type_type typ = transceiver_iface::UINT32;
};
template <>
struct from_std<int64_t>
{
    static const transceiver_iface::type_type typ = transceiver_iface::INT64;
};
template <>
struct from_std<uint64_t>
{
    static const transceiver_iface::type_type typ = transceiver_iface::UINT64;
};
#ifdef __APPLE__
template <>
struct from_std<long>
{
    static const transceiver_iface::type_type typ = transceiver_iface::INT64;
};
template <>
struct from_std<unsigned long>
{
    static const transceiver_iface::type_type typ = transceiver_iface::UINT64;
};
#endif

template <typename T>
static bool not_empty(const daal::services::SharedPtr<T> & obj)
{
    return obj;
}

template <typename T>
static bool not_empty(const daal::data_management::interface1::NumericTablePtr & obj)
{
    return obj && obj->getNumberOfRows() && obj->getNumberOfColumns();
}

template <typename T>
void transceiver::send(const T & obj, size_t recpnt, size_t tag)
{
    daal::data_management::InputDataArchive in_arch;
    int mysize(0);
    // Serialize the oneDAL object into a data archive
    if (not_empty(obj))
    {
        obj->serialize(in_arch);
        mysize = in_arch.getSizeOfArchive();
    }
    // and send it away to our recipient
    m_transceiver->send(&mysize, sizeof(mysize), recpnt, tag);
    if (mysize > 0)
    {
        m_transceiver->send(in_arch.getArchiveAsArraySharedPtr().get(), mysize, recpnt, tag);
    }
}

template <typename T>
T transceiver::recv(size_t sender, size_t tag)
{
    int sz(0);
    size_t br = m_transceiver->recv(&sz, sizeof(sz), sender, tag);
    assert(br == sizeof(sz));
    T res;
    if (sz > 0)
    {
        daal::byte * buf = static_cast<daal::byte *>(daal::services::daal_malloc(sz * sizeof(daal::byte)));
        DAAL4PY_CHECK_MALLOC(buf);
        br = m_transceiver->recv(buf, sz, sender, tag);
        assert(br == sz);
        // It'd be nice to avoid the additional copy, need a special DatArchive (see older CnC versions of daal4py)
        daal::data_management::OutputDataArchive out_arch(buf, sz);
        res = daal::services::staticPointerCast<typename T::ElementType>(out_arch.getAsSharedPtr());
        daal::services::daal_free(buf);
        buf = NULL;
    }
    return res;
}

template <typename T>
std::vector<daal::services::SharedPtr<T> > transceiver::gather(const daal::services::SharedPtr<T> & obj, size_t root, bool varying)
{
    // we split into 2 gathers: one to send the sizes, a second to send the actual data
    if (varying == false) std::cerr << "Performance warning: no optimization implemented for non-varying gather sizes\n";

    size_t mysize = 0;
    daal::data_management::InputDataArchive in_arch;
    // If we got the data then serialize the partial result into a data archive
    // In other case the size of data to send is equal zero, send nothing
    if (obj)
    {
        obj->serialize(in_arch);
        mysize = in_arch.getSizeOfArchive();
    }

    // gather all partial results
    // First get all sizes, then gather on root
    size_t * sizes = reinterpret_cast<size_t *>(m_transceiver->gather(&mysize, sizeof(mysize), root, NULL, false));
    char * buff    = reinterpret_cast<char *>(m_transceiver->gather(in_arch.getArchiveAsArraySharedPtr().get(), mysize, root, sizes));

    std::vector<daal::services::SharedPtr<T> > all;
    if (m_transceiver->me() == root)
    {
        size_t offset = 0;
        size_t nm     = m_transceiver->nMembers();
        all.resize(nm);
        for (int i = 0; i < nm; ++i)
        {
            if (sizes[i] > 0)
            {
                // This is inefficient, we need to write our own DatArchive to avoid extra copy
                daal::data_management::OutputDataArchive out_arch(reinterpret_cast<daal::byte *>(buff + offset), sizes[i]);
                all[i] = daal::services::staticPointerCast<T>(out_arch.getAsSharedPtr());
                offset += sizes[i];
            }
            else
            {
                all[i] = daal::services::SharedPtr<T>();
            }
        }
        daal::services::daal_free(buff);
        buff = NULL;
    }

    daal::services::daal_free(sizes);
    sizes = NULL;

    return all;
}

template <typename T>
void transceiver::bcast(T & obj, size_t root)
{
    m_transceiver->bcast(&obj, sizeof(obj), root);
}

template <typename T>
void transceiver::bcast(daal::services::SharedPtr<T> & obj, size_t root)
{
    // we split into 2 messages: one to send the size, a second to send the actual data
    if (m_transceiver->me() == root)
    {
        // Serialize the partial result into a data archive
        daal::data_management::InputDataArchive in_arch;
        obj->serialize(in_arch);
        int size = in_arch.getSizeOfArchive();
        m_transceiver->bcast(&size, sizeof(size), root);
        if (size > 0) m_transceiver->bcast(in_arch.getArchiveAsArraySharedPtr().get(), size, root);
    }
    else
    {
        int size = 0;
        m_transceiver->bcast(&size, sizeof(size), root);
        if (size > 0)
        {
            char * buff = static_cast<char *>(daal::services::daal_malloc(size));
            m_transceiver->bcast(buff, size, root);
            daal::data_management::OutputDataArchive out_arch(reinterpret_cast<daal::byte *>(buff), size);
            obj = daal::services::staticPointerCast<T>(out_arch.getAsSharedPtr());
        }
        else
        {
            obj.reset();
        }
    }
}

template <typename T>
void transceiver::reduce_all(T * inout, size_t n, transceiver_iface::operation_type op)
{
    m_transceiver->reduce_all(inout, from_std<T>::typ, n, op);
}

template <typename T>
void transceiver::reduce_exscan(T * inout, size_t n, transceiver_iface::operation_type op)
{
    m_transceiver->reduce_exscan(inout, from_std<T>::typ, n, op);
}

#endif // _TRANSCEIVER_INCLUDED_
