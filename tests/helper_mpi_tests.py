# ===============================================================================
# Copyright 2024 Intel Corporation
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
# ===============================================================================

# The purpose of this helper script is to be able to execute
# different shell commands according to whether the process
# is the first MPI rank or not. It works by passing it the
# arguments to a pytest call with the json report arguments
# from an mpirun/mpiexec invocation - e.g.
#     mpirun -n 2 helper_mpi_tests.py pytest <args to pytest>

import sys

import pytest

print("Info from 'helper_mpi_tests.py'")
print("sys.argv:", sys.argv)

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    is_rank_zero = rank == 0
except ImportError:
    is_rank_zero = True

args_base = [
    arg for arg in sys.argv[1:] if not ("--json-report" in arg) and (arg != "pytest")
]
args_json_report = [
    arg for arg in sys.argv[1:] if ("--json-report" in arg) and (arg != "pytest")
]

# Propagate pytest's status: without this the script always exits zero, so an MPI
# rank that fails its tests reports success to the launcher and to the caller.
if is_rank_zero:
    sys.exit(pytest.main(args_base + args_json_report))
else:
    sys.exit(pytest.main(args_base))
