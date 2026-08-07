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
#include <exception>
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
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::overflow_error(std::string(name) + " exceeds MPI int count range");
    return static_cast<int>(value);
}

void mpi_finalize_noexcept() noexcept
{
    int finalized = 0;
    if (MPI_Finalized(&finalized) != MPI_SUCCESS) return;
    if (!finalized) (void)MPI_Finalize();
}

// Agree across ranks on whether it is safe to enter a collective. This is only
// worth its extra MPI_Allreduce where the validation is *asymmetric* - i.e. the
// gather path, where only the root allocates a receive buffer and so only the
// root can fail before the collective. Without this, a failing root would
// leave every other rank blocked in MPI_Gather until the job is killed.
// Symmetric checks (count range, root rank) fail on all ranks at once and need
// no agreement, so the collectives on the hot path must not use this.
template <typename Validate>
void collective_preflight(Validate && validate)
{
    std::exception_ptr local_error;
    try
    {
        validate();
    }
    catch (...)
    {
        local_error = std::current_exception();
    }

    int local_failed = local_error ? 1 : 0;
    int any_failed   = 0;
    mpi_check(MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), "MPI_Allreduce(collective preflight)");
    if (!any_failed) return;
    if (local_error) std::rethrow_exception(local_error);
    throw std::runtime_error("MPI collective validation or allocation failed on another rank");
}
} // namespace

void mpi_transceiver::init()
{
    std::lock_guard<std::mutex> lock(m_lifecycle_mutex);
    if (m_users)
    {
        ++m_users;
        return;
    }

    try
    {
        int finalized = 0;
        mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
        if (finalized) throw std::runtime_error("MPI cannot be reinitialized after MPI_Finalize");

        int initialized = 0;
        mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
        int provided = MPI_THREAD_SINGLE;
        if (!initialized)
        {
            mpi_check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided), "MPI_Init_thread");
            m_owns_mpi = true;
        }
        else
        {
            mpi_check(MPI_Query_thread(&provided), "MPI_Query_thread");
        }

        // MPI_THREAD_MULTIPLE is what daal4py needs in order to let more than
        // one Python thread drive distributed computations. Not getting it is
        // not fatal - single-threaded distributed use stays correct - so warn
        // rather than fail, which also lets callers escalate it with -W error.
        // The GIL is held here: init() is only reached from a Python call.
        if (provided < MPI_THREAD_MULTIPLE)
        {
            if (PyErr_WarnEx(PyExc_RuntimeWarning,
                             "The MPI library provides a thread support level below MPI_THREAD_MULTIPLE. "
                             "Distributed daal4py computations must then be driven from a single thread.",
                             1)
                < 0)
                throw std::runtime_error("MPI thread support level below MPI_THREAD_MULTIPLE");
        }

        transceiver_impl::init();
        m_users = 1;
    }
    catch (...)
    {
        if (m_owns_mpi) mpi_finalize_noexcept();
        m_owns_mpi    = false;
        m_initialized = false;
        throw;
    }
}

void mpi_transceiver::fini() noexcept
{
    try
    {
        std::lock_guard<std::mutex> lock(m_lifecycle_mutex);
        if (!m_users || --m_users) return;
        const bool owns_mpi = m_owns_mpi;
        m_owns_mpi          = false;
        m_initialized       = false;
        if (owns_mpi) mpi_finalize_noexcept();
    }
    catch (...)
    {}
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
    mpi_check(MPI_Send(buff, mpi_count(N, "send size"), MPI_CHAR, mpi_count(recpnt, "recipient"), mpi_count(tag, "tag"), MPI_COMM_WORLD), "MPI_Send");
}

size_t mpi_transceiver::recv(void * buff, size_t N, int sender, int tag)
{
    MPI_Status status;
    mpi_check(MPI_Recv(buff, mpi_count(N, "receive size"), MPI_CHAR, sender, tag, MPI_COMM_WORLD, &status), "MPI_Recv");
    int count = 0;
    mpi_check(MPI_Get_count(&status, MPI_CHAR, &count), "MPI_Get_count");
    return static_cast<size_t>(count);
}

void * mpi_transceiver::gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying)
{
    char * buff   = nullptr;
    int count     = 0;
    int root_rank = 0;
    std::vector<int> offsets;
    std::vector<int> counts;

    try
    {
        collective_preflight([&] {
            count = mpi_count(N, "gather size");
            if (root >= m_nMembers) throw std::out_of_range("gather root rank is outside MPI_COMM_WORLD");
            root_rank = static_cast<int>(root);

            if (m_me != root) return;
            if (varying)
            {
                if (!sizes) throw std::invalid_argument("gather sizes are null on root");
                offsets.resize(m_nMembers);
                counts.resize(m_nMembers);
                size_t total = 0;
                for (size_t i = 0; i < m_nMembers; ++i)
                {
                    offsets[i] = mpi_count(total, "gather offset");
                    counts[i]  = mpi_count(sizes[i], "gather member size");
                    if (sizes[i] > std::numeric_limits<size_t>::max() - total) throw std::overflow_error("gather total size overflow");
                    total += sizes[i];
                }
                if (total)
                {
                    buff = static_cast<char *>(daal::services::daal_malloc(total));
                    DAAL4PY_CHECK_MALLOC(buff);
                }
            }
            else if (m_nMembers && N)
            {
                if (m_nMembers > std::numeric_limits<size_t>::max() / N) throw std::overflow_error("gather allocation size overflow");
                buff = static_cast<char *>(daal::services::daal_malloc(m_nMembers * N));
                DAAL4PY_CHECK_MALLOC(buff);
            }
        });
    }
    catch (...)
    {
        if (buff) daal::services::daal_free(buff);
        throw;
    }

    try
    {
        if (varying)
            mpi_check(MPI_Gatherv(ptr, count, MPI_CHAR, buff, counts.empty() ? nullptr : counts.data(), offsets.empty() ? nullptr : offsets.data(), MPI_CHAR, root_rank, MPI_COMM_WORLD), "MPI_Gatherv");
        else
            mpi_check(MPI_Gather(ptr, count, MPI_CHAR, buff, count, MPI_CHAR, root_rank, MPI_COMM_WORLD), "MPI_Gather");
    }
    catch (...)
    {
        if (buff) daal::services::daal_free(buff);
        throw;
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

// The validation in the three collectives below is deliberately local: every
// rank passes the same count, root and data type, so any rank that rejects the
// arguments is joined by all the others. See collective_preflight above for the
// asymmetric case.
void mpi_transceiver::bcast(void * ptr, size_t N, size_t root)
{
    const int count = mpi_count(N, "broadcast size");
    if (root >= m_nMembers) throw std::out_of_range("broadcast root rank is outside MPI_COMM_WORLD");
    mpi_check(MPI_Bcast(ptr, count, MPI_CHAR, static_cast<int>(root), MPI_COMM_WORLD), "MPI_Bcast");
}

void mpi_transceiver::reduce_all(void * inout, transceiver_iface::type_type T, size_t N, transceiver_iface::operation_type operation)
{
    const int count = mpi_count(N, "allreduce count");
    mpi_check(MPI_Allreduce(MPI_IN_PLACE, inout, count, to_mpi(T), to_mpi(operation), MPI_COMM_WORLD), "MPI_Allreduce");
}

void mpi_transceiver::reduce_exscan(void * inout, transceiver_iface::type_type T, size_t N, transceiver_iface::operation_type operation)
{
    const int count = mpi_count(N, "exscan count");
    mpi_check(MPI_Exscan(MPI_IN_PLACE, inout, count, to_mpi(T), to_mpi(operation), MPI_COMM_WORLD), "MPI_Exscan");
}

extern "C" PyMODINIT_FUNC PyInit_mpi_transceiver(void)
{
    static std::shared_ptr<mpi_transceiver> transceiver_instance;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "daal4py.mpi_transceiver", "No docs", -1, nullptr,
    };
    PyObject * module = PyModule_Create(&moduledef);
    if (!module) return nullptr;

    transceiver_instance = std::make_shared<mpi_transceiver>();
    PyObject * pointer   = PyLong_FromVoidPtr(static_cast<void *>(&transceiver_instance));
    if (!pointer || PyModule_AddObject(module, "transceiver", pointer) < 0)
    {
        Py_XDECREF(pointer);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
