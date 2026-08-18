# ==============================================================================
# Copyright 2025 Intel Corporation
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

"""Thread-local manager for SPMD communicators.

This mirrors :mod:`onedal.utils._sycl_queue_manager`, but for the MPI (and,
later, oneCCL) communicators produced by the standalone ``_onedal_spmd_mpi``
module. The communicator is created lazily on first use and cached so that the
transport backend is not re-initialized on every call.

The cached communicator's identity is the underlying MPI communicator, i.e. the
optional external comm handle. When no handle is supplied it is always
``MPI_COMM_WORLD``, so there is exactly one communicator; a distinct one is only
created when an external communicator is explicitly provided (a later addition
for dask-mpi). The SYCL queue is passed through to construction solely to select
the host vs. device flavor of the communicator and is deliberately *not* part of
the cache identity: communicator creation is a collective operation across ranks
(especially once oneCCL is added) that must run the same number of times and in
the same order on every rank, and queue objects have churning identity. A
process/thread is assumed to use a single target device, matching the dask-mpi
worker model.
"""

from threading import local

from .. import _spmd_mpi_backend


class ThreadLocalGlobals(local):

    def __init__(self):
        # Lazily created communicator, valid for the current comm_handle.
        self.communicator = None
        # Optional external MPI communicator handle (a Fortran integer, e.g.
        # from mpi4py's ``MPI.Comm.py2f()``). When None the default
        # MPI_COMM_WORLD is used. The delivery mechanism (dask-mpi integration)
        # is a later addition; this is the seam.
        self.comm_handle = None


__globals = ThreadLocalGlobals()


def _check_backend():
    """Raise if the SPMD MPI communicator module is unavailable."""
    if _spmd_mpi_backend is None:
        raise RuntimeError(
            "SPMD communicator support is not available: the _onedal_spmd_mpi "
            "module failed to load. Distributed (SPMD) execution requires a "
            "build with MPI support."
        )


def get_global_communicator(queue=None):
    """Get the SPMD communicator, creating it on first use.

    Parameters
    ----------
    queue : SyclQueue or None, default=None
        SYCL queue selecting a device communicator. If ``None``, a host
        (CPU-distributed) communicator is created. Used only to select the
        communicator flavor at creation; it is not part of the cache identity.

    Returns
    -------
    communicator : communicator_host or communicator_device
        Communicator object from the ``_onedal_spmd_mpi`` module, suitable for
        passing to an SPMD policy constructor.
    """
    _check_backend()
    if __globals.communicator is None:
        __globals.communicator = _spmd_mpi_backend.create_communicator(
            queue, comm_handle=__globals.comm_handle
        )
    return __globals.communicator


def update_global_communicator(comm_handle):
    """Set the external MPI communicator handle used for communicator creation.

    Any previously cached communicator is discarded so that the next access
    recreates it on the new communicator.

    Parameters
    ----------
    comm_handle : int or None
        MPI_Comm Fortran integer handle (e.g. from ``MPI.Comm.py2f()``). ``None``
        restores the default of ``MPI_COMM_WORLD``.
    """
    __globals.comm_handle = comm_handle
    __globals.communicator = None


def remove_global_communicator():
    """Discard the cached communicator and clear any external comm handle."""
    __globals.comm_handle = None
    __globals.communicator = None
