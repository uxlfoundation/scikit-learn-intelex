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

void mpi_transceiver::init()
{
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    // protect against double-init
    if (!is_mpi_initialized)
    {
        MPI_Init(NULL, NULL);
    }
    transceiver_impl::init();
}

void mpi_transceiver::fini()
{
    MPI_Finalize();
}

size_t mpi_transceiver::nMembers()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

size_t mpi_transceiver::me()
{
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    return me;
}

void mpi_transceiver::send(const void * buff, size_t N, size_t recpnt, size_t tag)
{
    MPI_Send(buff, (int)N, MPI_CHAR, recpnt, tag, MPI_COMM_WORLD);
}

size_t mpi_transceiver::recv(void * buff, size_t N, int sender, int tag)
{
    MPI_Status s;
    MPI_Recv(buff, N, MPI_CHAR, sender, tag, MPI_COMM_WORLD, &s);
    int count;
    MPI_Get_count(&s, MPI_CHAR, &count);
    return count;
}

void * mpi_transceiver::gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying)
{
    char * buff = NULL;
    if (varying)
    {
        // -> gatherv
        if (m_me == root)
        {
            int * offsets = static_cast<int *>(daal::services::daal_malloc(m_nMembers * sizeof(int)));
            DAAL4PY_CHECK_MALLOC(offsets);
            DAAL4PY_CHECK_BAD_CAST(sizes[0] <= std::numeric_limits<int>::max());
            int tot_sz = sizes[0];
            offsets[0] = 0;
            for (int i = 1; i < m_nMembers; ++i)
            {
                DAAL4PY_OVERFLOW_CHECK_BY_ADDING(int, offsets[i - 1], sizes[i - 1]);
                offsets[i] = offsets[i - 1] + sizes[i - 1];
                DAAL4PY_OVERFLOW_CHECK_BY_ADDING(int, tot_sz, sizes[i]);
                tot_sz += sizes[i];
            }
            buff = static_cast<char *>(daal::services::daal_malloc(tot_sz));
            DAAL4PY_CHECK_MALLOC(buff);
            int * szs = static_cast<int *>(daal::services::daal_malloc(m_nMembers * sizeof(int)));
            DAAL4PY_CHECK_MALLOC(szs);
            for (size_t i = 0; i < m_nMembers; ++i)
            {
                szs[i] = static_cast<int>(sizes[i]);
            }
            MPI_Gatherv(ptr, N, MPI_CHAR, buff, szs, offsets, MPI_CHAR, root, MPI_COMM_WORLD);
            daal::services::daal_free(szs);
            szs = NULL;
            daal::services::daal_free(offsets);
            offsets = NULL;
        }
        else
        {
            MPI_Gatherv(ptr, N, MPI_CHAR, NULL, NULL, NULL, MPI_CHAR, root, MPI_COMM_WORLD);
        }
    }
    else
    {
        if (m_me == root)
        {
            buff = static_cast<char *>(daal::services::daal_malloc(m_nMembers * N));
            DAAL4PY_CHECK_MALLOC(buff);
        }
        // -> gather with same size on all procs
        MPI_Gather(ptr, N, MPI_CHAR, buff, N, MPI_CHAR, root, MPI_COMM_WORLD);
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
    case transceiver_iface::UINT32: return MPI_INT32_T;
    case transceiver_iface::INT64: return MPI_INT64_T;
    case transceiver_iface::UINT64: return MPI_INT64_T;
    case transceiver_iface::FLOAT: return MPI_FLOAT;
    case transceiver_iface::DOUBLE: return MPI_DOUBLE;
    default: throw std::logic_error("unsupported data type");
    }
}

static MPI_Op to_mpi(transceiver_iface::operation_type o)
{
    switch (o)
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
    MPI_Bcast(ptr, N, MPI_CHAR, root, MPI_COMM_WORLD);
}

void mpi_transceiver::reduce_all(void * inout, transceiver_iface::type_type T, size_t N, transceiver_iface::operation_type op)
{
    MPI_Allreduce(MPI_IN_PLACE, inout, N, to_mpi(T), to_mpi(op), MPI_COMM_WORLD);
}

void mpi_transceiver::reduce_exscan(void * inout, transceiver_iface::type_type T, size_t N, transceiver_iface::operation_type op)
{
    MPI_Exscan(MPI_IN_PLACE, inout, N, to_mpi(T), to_mpi(op), MPI_COMM_WORLD);
}

// ************************************
// ************************************

extern "C" PyMODINIT_FUNC PyInit_mpi_transceiver(void)
{
    // shared pointer, will GC transceiver when shutting down
    static std::shared_ptr<mpi_transceiver> s_smt;
    PyObject * m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "daal4py.mpi_transceiver", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    s_smt.reset(new mpi_transceiver);
    PyObject_SetAttrString(m, "transceiver", PyLong_FromVoidPtr((void *)(&s_smt)));
    return m;
}
