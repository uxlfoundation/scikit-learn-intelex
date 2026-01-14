# ==============================================================================
# Copyright 2023 Intel Corporation
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

# sklearnex DBSCAN example for distributed systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./dbscan_spmd.py

from warnings import warn

import dpnp
import numpy as np
from dpctl import SyclQueue
from mpi4py import MPI
from sklearn.datasets import load_digits

from sklearnex.spmd.cluster import DBSCAN


def get_data_slice(chunk, count):
    assert chunk < count
    X, y = load_digits(return_X_y=True)
    n_samples, _ = X.shape
    size = n_samples // count
    first = chunk * size
    last = first + size
    return (X[first:last, :], y[first:last])


def get_train_data(rank, size):
    return get_data_slice(rank, size + 1)


def get_test_data(size):
    return get_data_slice(size, size + 1)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X, _ = get_train_data(rank, size)

queue = SyclQueue("gpu")

dpnp_X = dpnp.asarray(X, usm_type="device", sycl_queue=queue)

model = DBSCAN(eps=3, min_samples=2).fit(dpnp_X)

print(f"Labels on rank {rank} (slice of 2):\n", model.labels_[:2])
