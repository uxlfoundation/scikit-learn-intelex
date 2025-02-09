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

#ifndef _MPI_TRANSCEIVER_INCLUDED_
#define _MPI_TRANSCEIVER_INCLUDED_

#include "transceiver.h"

// implementation of transceiver_iface using MPI
class mpi_transceiver : public transceiver_impl
{
public:
    mpi_transceiver() : transceiver_impl() {}

    virtual void init();

    virtual void fini();

    virtual size_t nMembers();

    virtual size_t me();

    virtual void send(const void * buff, size_t N, size_t recpnt, size_t tag);

    virtual size_t recv(void * buff, size_t N, int sender, int tag);

    virtual void * gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying = true);

    virtual void bcast(void * ptr, size_t N, size_t root);

    virtual void reduce_all(void * inout, transceiver_iface::type_type T, size_t N, transceiver_iface::operation_type op);

    virtual void reduce_exscan(void * inout, transceiver_iface::type_type T, size_t N, transceiver_iface::operation_type op);
};

#endif // _MPI_TRANSCEIVER_INCLUDED_
