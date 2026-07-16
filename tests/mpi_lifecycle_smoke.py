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
import daal4py

assert daal4py.__has_dist__
assert daal4py.num_procs() == 2
rank = daal4py.my_procid()
assert rank in (0, 1)

from mpi4py import MPI

assert MPI.Is_initialized()
assert not MPI.Is_finalized()
assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
assert MPI.COMM_WORLD.Get_size() == 2
assert MPI.COMM_WORLD.Get_rank() == rank
assert MPI.COMM_WORLD.allreduce(rank, op=MPI.SUM) == 1
