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

# sklearnex HDBSCAN example for distributed systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./hdbscan_spmd.py

import dpnp
from dpctl import SyclQueue
from mpi4py import MPI
from sklearn.datasets import load_digits

from sklearnex import config_context
from sklearnex.spmd.cluster import HDBSCAN


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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X, _ = get_train_data(rank, size)

queue = SyclQueue("gpu")

dpnp_X = dpnp.asarray(X, usm_type="device", sycl_queue=queue)

# Array API dispatch keeps dpnp data on device throughout the computation.
# The SCIPY_ARRAY_API environment variable must also be set to enable this.
with config_context(array_api_dispatch=True):
    model = HDBSCAN(min_cluster_size=10).fit(dpnp_X)

    print(f"Labels on rank {rank} (slice of 2):\n", model.labels_[:2])
