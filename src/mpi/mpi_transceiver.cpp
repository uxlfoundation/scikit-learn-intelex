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

#include "mpi_transceiver.h"
#include "daal4py_defines.h"
#include <mpi.h>
#include <Python.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void mpi_check(int code, const char * operation)
{
    if (code == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, error, &length);
    throw std::runtime_error(std::string(operation) + " failed: " + std::string(error, length));
}

int mpi_count(size_t value, const char * name)
{
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(std::string(name) + " exceeds MPI int count range");
    return static_cast<int>(value);
}
} // namespace

void mpi_transceiver::init()
{
    if (m_initialized) return;

    int finalized = 0;
    mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized) throw std::runtime_error("MPI cannot be reinitialized after MPI_Finalize");

    int initialized = 0;
    mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    int provided = MPI_THREAD_SINGLE;
    if (!initialized)
    {
        mpi_check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided),
                  "MPI_Init_thread");
        m_owns_mpi = true;
    }
    else
    {
        mpi_check(MPI_Query_thread(&provided), "MPI_Query_thread");
    }

    if (provided < MPI_THREAD_MULTIPLE)
    {
        if (m_owns_mpi)
        {
            MPI_Finalize();
            m_owns_mpi = false;
        }
        throw std::runtime_error(
            "daal4py distributed free-threaded execution requires MPI_THREAD_MULTIPLE");
    }

    transceiver_impl::init();
}

void mpi_transceiver::fini()
{
    if (!m_initialized) return;
    if (m_owns_mpi)
    {
        int finalized = 0;
        mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
        if (!finalized) mpi_check(MPI_Finalize(), "MPI_Finalize");
    }
    m_owns_mpi  = false;
    m_initialized = false;
}

size_t mpi_transceiver::nMembers()
{
    int size = 0;
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");
    return static_cast<size_t>(size);
}

size_t mpi_transceiver::me()
{
    int rank = 0;
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    return static_cast<size_t>(rank);
}

void mpi_transceiver::send(const void * buff, size_t N, size_t recpnt, size_t tag)
{
    mpi_check(MPI_Send(buff,
                       mpi_count(N, "send size"),
                       MPI_CHAR,
                       mpi_count(recpnt, "recipient"),
                       mpi_count(tag, "tag"),
                       MPI_COMM_WORLD),
              "MPI_Send");
}

size_t mpi_transceiver::recv(void * buff, size_t N, int sender, int tag)
{
    MPI_Status status;
    mpi_check(MPI_Recv(buff,
                       mpi_count(N, "receive size"),
                       MPI_CHAR,
                       sender,
                       tag,
                       MPI_COMM_WORLD,
                       &status),
              "MPI_Recv");
    int count = 0;
    mpi_check(MPI_Get_count(&status, MPI_CHAR, &count), "MPI_Get_count");
    return static_cast<size_t>(count);
}

void * mpi_transceiver::gather(const void * ptr,
                               size_t N,
                               size_t root,
                               const size_t * sizes,
                               bool varying)
{
    char * buff = nullptr;
    const int count = mpi_count(N, "gather size");
    const int root_rank = mpi_count(root, "root rank");
    if (varying)
    {
        if (m_me == root)
        {
            std::vector<int> offsets(m_nMembers);
            std::vector<int> counts(m_nMembers);
            size_t total = 0;
            for (size_t i = 0; i < m_nMembers; ++i)
            {
                offsets[i] = mpi_count(total, "gather offset");
                counts[i] = mpi_count(sizes[i], "gather member size");
                if (sizes[i] > std::numeric_limits<size_t>::max() - total)
                    throw std::overflow_error("gather total size overflow");
                total += sizes[i];
            }
            buff = static_cast<char *>(daal::services::daal_malloc(total));
            DAAL4PY_CHECK_MALLOC(buff);
            try
            {
                mpi_check(MPI_Gatherv(ptr,
                                      count,
                                      MPI_CHAR,
                                      buff,
                                      counts.data(),
                                      offsets.data(),
                                      MPI_CHAR,
                                      root_rank,
                                      MPI_COMM_WORLD),
                          "MPI_Gatherv");
            }
            catch (...)
            {
                daal::services::daal_free(buff);
                throw;
            }
        }
        else
        {
            mpi_check(MPI_Gatherv(ptr,
                                  count,
                                  MPI_CHAR,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  MPI_CHAR,
                                  root_rank,
                                  MPI_COMM_WORLD),
                      "MPI_Gatherv");
        }
    }
    else
    {
        if (m_me == root)
        {
            if (N && m_nMembers > std::numeric_limits<size_t>::max() / N)
                throw std::overflow_error("gather allocation size overflow");
            buff = static_cast<char *>(daal::services::daal_malloc(m_nMembers * N));
            DAAL4PY_CHECK_MALLOC(buff);
        }
        try
        {
            mpi_check(MPI_Gather(ptr,
                                 count,
                                 MPI_CHAR,
                                 buff,
                                 count,
                                 MPI_CHAR,
                                 root_rank,
                                 MPI_COMM_WORLD),
                      "MPI_Gather");
        }
        catch (...)
        {
            if (buff) daal::services::daal_free(buff);
            throw;
        }
    }
    return buff;
}

static MPI_Datatype to_mpi(transceiver_iface::type_type T)
{
    switch (T)
    {
    case transceiver_iface::BOOL: return MPI_C_BOOL;
    case transceiver_iface::INT8: return MPI_INT8_T;
    case transceiver_iface::UINT8: return MPI_UINT8_T;
    case transceiver_iface::INT32: return MPI_INT32_T;
    case transceiver_iface::UINT32: return MPI_UINT32_T;
    case transceiver_iface::INT64: return MPI_INT64_T;
    case transceiver_iface::UINT64: return MPI_UINT64_T;
    case transceiver_iface::FLOAT: return MPI_FLOAT;
    case transceiver_iface::DOUBLE: return MPI_DOUBLE;
    default: throw std::logic_error("unsupported data type");
    }
}

static MPI_Op to_mpi(transceiver_iface::operation_type operation)
{
    switch (operation)
    {
    case transceiver_iface::OP_MAX: return MPI_MAX;
    case transceiver_iface::OP_MIN: return MPI_MIN;
    case transceiver_iface::OP_SUM: return MPI_SUM;
    case transceiver_iface::OP_PROD: return MPI_PROD;
    case transceiver_iface::OP_LAND: return MPI_LAND;
    case transceiver_iface::OP_BAND: return MPI_BAND;
    case transceiver_iface::OP_LOR: return MPI_LOR;
    case transceiver_iface::OP_BOR: return MPI_BOR;
    case transceiver_iface::OP_LXOR: return MPI_LXOR;
    case transceiver_iface::OP_BXOR: return MPI_BXOR;
    default: throw std::logic_error("unsupported operation type");
    }
}

void mpi_transceiver::bcast(void * ptr, size_t N, size_t root)
{
    mpi_check(MPI_Bcast(ptr,
                        mpi_count(N, "broadcast size"),
                        MPI_CHAR,
                        mpi_count(root, "root rank"),
                        MPI_COMM_WORLD),
              "MPI_Bcast");
}

void mpi_transceiver::reduce_all(void * inout,
                                 transceiver_iface::type_type T,
                                 size_t N,
                                 transceiver_iface::operation_type operation)
{
    mpi_check(MPI_Allreduce(MPI_IN_PLACE,
                            inout,
                            mpi_count(N, "allreduce count"),
                            to_mpi(T),
                            to_mpi(operation),
                            MPI_COMM_WORLD),
              "MPI_Allreduce");
}

void mpi_transceiver::reduce_exscan(void * inout,
                                    transceiver_iface::type_type T,
                                    size_t N,
                                    transceiver_iface::operation_type operation)
{
    mpi_check(MPI_Exscan(MPI_IN_PLACE,
                         inout,
                         mpi_count(N, "exscan count"),
                         to_mpi(T),
                         to_mpi(operation),
                         MPI_COMM_WORLD),
              "MPI_Exscan");
}

extern "C" PyMODINIT_FUNC PyInit_mpi_transceiver(void)
{
    static std::shared_ptr<mpi_transceiver> transceiver_instance;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "daal4py.mpi_transceiver", "No docs", -1, nullptr,
    };
    PyObject * module = PyModule_Create(&moduledef);
    if (!module) return nullptr;
#if PY_VERSION_HEX >= 0x030D0000
    if (PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED) < 0)
    {
        Py_DECREF(module);
        return nullptr;
    }
#endif

    transceiver_instance = std::make_shared<mpi_transceiver>();
    PyObject * pointer = PyLong_FromVoidPtr(static_cast<void *>(&transceiver_instance));
    if (!pointer || PyModule_AddObject(module, "transceiver", pointer) < 0)
    {
        Py_XDECREF(pointer);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
