# ==============================================================================
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np

import daal4py


def test_external_mpi_survives_repeated_transceiver_lifecycle():
    """daalfini must release daal4py users, not MPI owned by mpi4py."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    def cycle(_):
        daal4py.daalinit()
        assert daal4py.num_procs() == comm.Get_size()
        assert daal4py.my_procid() == comm.Get_rank()
        daal4py.daalfini()

    for _ in range(20):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(cycle, range(16)))
        assert not MPI.Is_finalized()
        comm.Barrier()

    # The transceiver must still be constructible after previous wrappers die.
    daal4py.daalinit()
    assert daal4py.num_procs() == comm.Get_size()
    daal4py.daalfini()
    assert not MPI.Is_finalized()


def test_lazy_init_does_not_invert_gil_and_lifecycle_mutex():
    """A GIL-holding num_procs waiter must not block distributed lazy init."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    data = np.arange(400, dtype=np.float64).reshape(200, 2) + comm.Get_rank()

    for _ in range(20):
        daal4py.daalfini()
        start = Barrier(2)

        def distributed_compute():
            start.wait()
            result = daal4py.covariance(distributed=True).compute(data)
            assert result.covariance is not None

        def query_topology():
            start.wait()
            assert daal4py.num_procs() == comm.Get_size()

        with ThreadPoolExecutor(max_workers=2) as executor:
            compute = executor.submit(distributed_compute)
            query = executor.submit(query_topology)
            compute.result(timeout=30)
            query.result(timeout=30)
        comm.Barrier()
