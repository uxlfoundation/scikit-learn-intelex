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
from sklearn.metrics import accuracy_score

from sklearnex.spmd.neighbors import KNeighborsClassifier


def generate_X_y(par, seed):
    ns, nf = par["ns"], par["nf"]

    drng = np.random.default_rng(seed)
    data = drng.uniform(-1, 1, size=(ns, nf))
    resp = (data > 0) @ (2 ** np.arange(nf))

    return data, resp


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

params_train = {"ns": 100000, "nf": 3}
params_test = {"ns": 100, "nf": 3}

X_train, y_train = generate_X_y(params_train, rank)
X_test, y_test = generate_X_y(params_test, rank + 99)

dpt_X_train = dpt.asarray(X_train, usm_type="device", sycl_queue=q)
dpt_y_train = dpt.asarray(y_train, usm_type="device", sycl_queue=q)
dpt_X_test = dpt.asarray(X_test, usm_type="device", sycl_queue=q)
dpt_y_test = dpt.asarray(y_test, usm_type="device", sycl_queue=q)

model_spmd = KNeighborsClassifier(
    algorithm="brute", n_neighbors=20, weights="uniform", p=2, metric="minkowski"
)
model_spmd.fit(dpt_X_train, dpt_y_train)

y_predict = model_spmd.predict(dpt_X_test)

print("Brute Force Distributed kNN classification results:")
print("Ground truth (first 5 observations on rank {}):\n{}".format(rank, y_test[:5]))
print(
    "Classification results (first 5 observations on rank {}):\n{}".format(
        rank, dpt.to_numpy(y_predict)[:5]
    )
)
print(
    "Accuracy for entire rank {} (256 classes): {}\n".format(
        rank, accuracy_score(y_test, dpt.to_numpy(y_predict))
    )
)
