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

# Import daal4py before mpi4py to exercise daal4py-owned MPI initialization.
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

is_free_threaded = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
if is_free_threaded:
    assert not sys._is_gil_enabled()

import daal4py

assert daal4py.__has_dist__
assert daal4py.num_procs() == 2
rank = daal4py.my_procid()
assert rank in (0, 1)
if is_free_threaded:
    assert not sys._is_gil_enabled()

from mpi4py import MPI

assert MPI.Is_initialized()
assert not MPI.Is_finalized()
assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
assert MPI.COMM_WORLD.Get_size() == 2
assert MPI.COMM_WORLD.Get_rank() == rank
assert MPI.COMM_WORLD.allreduce(rank, op=MPI.SUM) == 1
if is_free_threaded:
    assert not sys._is_gil_enabled()


def cycle(_):
    daal4py.daalinit()
    assert daal4py.num_procs() == 2
    assert daal4py.my_procid() == rank
    daal4py.daalfini()


with ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(cycle, range(8)))

assert not MPI.Is_finalized()
MPI.COMM_WORLD.Barrier()
