# ==============================================================================
# Copyright contributors to the oneDAL project
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

"""Multi-rank check of daal4py-owned MPI initialization and teardown.

This covers the case where ``daal4py`` - not ``mpi4py`` - calls
``MPI_Init_thread``. Nothing here may import ``mpi4py``: importing it initializes
MPI, which would make it the owner and turn this into the externally owned case
that ``tests/test_mpi_lifecycle.py`` covers instead. Everything below therefore
queries ``daal4py`` itself, cross-checked against what the MPI launcher put in
the environment.

For the same reason this is a standalone script rather than a pytest module:
``pytest-mpi`` imports ``mpi4py`` in its ``pytest_runtest_setup`` hook before the
body of any ``mpi``-marked test runs, and ``tests/helper_mpi_tests.py`` imports it
at module scope. MPI also cannot be initialized again after ``MPI_Finalize``, so
the whole lifecycle has to fit in one process.

Run as ``mpiexec -n <ranks> python tests/mpi_lifecycle_smoke.py``; any rank count
above one works, since the launchers differ on how many they permit.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import daal4py

assert daal4py.__has_dist__

# The first distributed call is what initializes MPI, and it warns if the library
# provides a thread support level below MPI_THREAD_MULTIPLE. The threaded sections
# below rely on that level, so promote the warning to an error rather than running
# them on a weaker guarantee.
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    size = daal4py.num_procs()
    rank = daal4py.my_procid()

assert size > 1, "this smoke test needs at least two ranks"
assert 0 <= rank < size

# An oracle independent of daal4py, without initializing MPI a second way: the
# launcher publishes the topology in the environment. Intel MPI and MPICH use the
# PMI names, Open MPI its own; skip the cross-check under a launcher that sets
# neither rather than guessing.
launcher_rank = os.environ.get("PMI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK"))
launcher_size = os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE"))
if launcher_rank is not None:
    assert rank == int(launcher_rank)
if launcher_size is not None:
    assert size == int(launcher_size)


def query_topology(_):
    assert daal4py.num_procs() == size
    assert daal4py.my_procid() == rank
    return daal4py.num_procs(), daal4py.my_procid()


# Keep daal4py-owned MPI alive while native topology calls overlap.
with ThreadPoolExecutor(max_workers=4) as executor:
    topology = list(executor.map(query_topology, range(64)))
assert topology == [(size, rank)] * 64

# A real distributed computation, which serves two purposes: it shows MPI is still
# usable after the threaded section, and its collectives synchronize the ranks, so
# no rank finalizes MPI while another is still communicating. It replaces the
# explicit barrier an mpi4py-based version would use.
data = np.arange(64, dtype=np.float64).reshape(16, 4) + rank
covariance = daal4py.covariance(distributed=True).compute(data)

# Concurrent teardown must be idempotent and result in exactly one process-wide
# MPI_Finalize call. No MPI operation is valid after this point, so the results
# above are only read afterwards - that is local memory.
with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(lambda _: daal4py.daalfini(), range(8)))

assert covariance.covariance.shape == (4, 4)
assert np.all(np.isfinite(covariance.covariance))

# daalinit only configures threads and remains valid. The next lazy distributed
# operation must fail because MPI cannot be initialized again after finalization,
# and that failure is also what proves the teardown above did finalize it.
daal4py.daalinit()
try:
    daal4py.num_procs()
except RuntimeError as error:
    assert "MPI cannot be reinitialized after MPI_Finalize" in str(error)
else:
    raise AssertionError("distributed use unexpectedly reinitialized finalized MPI")
