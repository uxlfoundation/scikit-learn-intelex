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

from warnings import warn

import dpctl
import dpctl.tensor as dpt
import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose
from sklearn.metrics import mean_squared_error

from sklearnex.spmd.neighbors import KNeighborsRegressor


def generate_X_y(par, coef_seed, data_seed):
    ns, nf = par["ns"], par["nf"]

    crng = np.random.default_rng(coef_seed)
    coef = crng.uniform(-10, 10, size=(nf,))

    drng = np.random.default_rng(data_seed)
    data = drng.uniform(-100, 100, size=(ns, nf))
    resp = data @ coef

    return data, resp, coef


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if dpctl.has_gpu_devices():
    q = dpctl.SyclQueue("gpu")
else:
    raise RuntimeError(
        "GPU devices unavailable. Currently, "
        "SPMD execution mode is implemented only for this device type."
    )

params_train = {"ns": 1000000, "nf": 3}
params_test = {"ns": 100, "nf": 3}

X_train, y_train, coef_train = generate_X_y(params_train, 10, rank)
X_test, y_test, coef_test = generate_X_y(params_test, 10, rank + 99)

dpt_X_train = dpt.asarray(X_train, usm_type="device", sycl_queue=q)
dpt_y_train = dpt.asarray(y_train, usm_type="device", sycl_queue=q)
dpt_X_test = dpt.asarray(X_test, usm_type="device", sycl_queue=q)
# dpt_y_test = dpt.asarray(y_test, usm_type="device", sycl_queue=q)

assert_allclose(coef_train, coef_test)

model_spmd = KNeighborsRegressor(
    algorithm="brute", n_neighbors=5, weights="uniform", p=2, metric="minkowski"
)
model_spmd.fit(dpt_X_train, dpt_y_train)

y_predict = model_spmd.predict(dpt_X_test)

print("Brute Force Distributed kNN regression results:")
print("Ground truth (first 5 observations on rank {}):\n{}".format(rank, y_test[:5]))
print(
    "Regression results (first 5 observations on rank {}):\n{}".format(
        rank, dpt.to_numpy(y_predict)[:5]
    )
)
print(
    "MSE for entire rank {}: {}\n".format(
        rank,
        mean_squared_error(y_test, dpt.to_numpy(y_predict)),
    )
)
