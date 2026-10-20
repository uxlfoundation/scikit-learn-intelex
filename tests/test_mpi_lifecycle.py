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

"""Lifecycle tests for the case where MPI is owned by *another* library.

Importing ``mpi4py.MPI`` calls ``MPI_Init_thread`` as a side effect, so doing it
before ``daal4py`` is what puts these tests on the not-``m_owns_mpi`` path: the
transceiver must adopt the existing MPI, and ``daalfini`` must never call
``MPI_Finalize`` on something it did not initialize. The import order below is
therefore load-bearing and not merely stylistic.

The mirror case - daal4py itself calling ``MPI_Init_thread`` - cannot be a pytest
module: it requires that nothing has touched MPI beforehand, while ``pytest-mpi``
imports ``mpi4py`` in ``pytest_runtest_setup`` before the body of an ``mpi``-marked
test runs. MPI also cannot be reinitialized after ``MPI_Finalize``, so that whole
lifecycle has to fit in one process. It lives in ``tests/mpi_lifecycle_smoke.py``.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

# Must precede the daal4py import: this is what makes mpi4py the owner of MPI.
MPI = pytest.importorskip("mpi4py.MPI", exc_type=ImportError)

import daal4py

pytest.importorskip("daal4py.mpi_transceiver", exc_type=ImportError)


# Every test here drives MPI, so each one carries the mark explicitly - a
# module-level `pytestmark` would select and skip the same way, but would not be
# greppable per test.
@pytest.mark.mpi
def test_mpi_is_externally_owned():
    """Guard the premise of this module: mpi4py, not daal4py, initialized MPI.

    If the import order above ever gets reordered by a formatter or an editor,
    the other tests here would silently start covering the daal4py-owned path
    instead - and would still pass. This fails loudly instead.
    """
    assert MPI.Is_initialized()
    assert not MPI.Is_finalized()


@pytest.mark.mpi
def test_transceiver_does_not_finalize_mpi_it_does_not_own():
    """daalfini must leave externally owned MPI usable."""
    comm = MPI.COMM_WORLD
    daal4py.daalinit()
    assert daal4py.num_procs() == comm.Get_size()
    assert daal4py.my_procid() == comm.Get_rank()
    daal4py.daalfini()

    assert not MPI.Is_finalized()
    # Not just "not finalized" - still actually usable by its owner.
    assert comm.allreduce(1, op=MPI.SUM) == comm.Get_size()


@pytest.mark.mpi
def test_externally_owned_mpi_can_recreate_transceiver():
    """A zero-user teardown must not prevent a later lazy initialization."""
    comm = MPI.COMM_WORLD

    for _ in range(2):
        assert daal4py.num_procs() == comm.Get_size()
        assert daal4py.my_procid() == comm.Get_rank()
        daal4py.daalfini()
        assert not MPI.Is_finalized()
        assert comm.allreduce(1, op=MPI.SUM) == comm.Get_size()


@pytest.mark.mpi
def test_distributed_compute_works_again_after_daalfini():
    """A second distributed computation must succeed across an intervening daalfini.

    The other tests here reach the transceiver through the topology calls, which
    is enough to cover its lifecycle but not to show that a real distributed
    algorithm can be run twice. This drives ``.compute()`` on both sides of a
    ``daalfini()``: because mpi4py owns MPI, that call releases only daal4py's
    transceiver, so the second ``.compute()`` creates a new one and succeeds. On
    the daal4py-owned path the same sequence must fail instead, which is what
    ``tests/mpi_lifecycle_smoke.py`` asserts.

    Both runs get identical input, so their R factors must agree regardless of how
    many ranks take part. Signs are not unique in a QR factorization, hence the
    comparison on absolute values.
    """
    comm = MPI.COMM_WORLD
    rng = np.random.RandomState(seed=0)
    data = rng.standard_normal(size=(16, 4))

    first = daal4py.qr(distributed=True).compute(data)
    daal4py.daalfini()
    assert not MPI.Is_finalized()

    second = daal4py.qr(distributed=True).compute(data)
    daal4py.daalfini()

    np.testing.assert_allclose(
        np.abs(first.matrixR), np.abs(second.matrixR), rtol=0, atol=1e-10
    )
    assert not MPI.Is_finalized()
    assert comm.allreduce(1, op=MPI.SUM) == comm.Get_size()


@pytest.mark.mpi
def test_external_mpi_survives_repeated_transceiver_lifecycle():
    """daalfini must release daal4py users, not MPI owned by mpi4py."""
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


@pytest.mark.mpi
def test_lazy_init_does_not_invert_gil_and_lifecycle_mutex():
    """A GIL-holding num_procs waiter must not block distributed lazy init."""
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
