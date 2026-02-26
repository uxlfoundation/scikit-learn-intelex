# ==============================================================================
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
# ==============================================================================

from warnings import warn

import dpctl
import dpnp
import numpy as np
from mpi4py import MPI
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearnex.spmd.linear_model import LogisticRegression


def generate_X_y(par, seed):
    np.random.seed()
    ns, nf = par["ns"], par["nf"]

    assert nf > 2
    # 2 last features will be redundant, weights are same for all ranks
    np.random.seed(42)
    intercept = np.random.normal(0, 5)
    weights = np.hstack([np.random.normal(0, 4, nf - 2), np.zeros(2)])

    np.random.seed(seed)
    X = np.random.normal(0, 3, (ns, nf))
    noise = np.random.normal(0, 4, ns)
    y = expit(X @ weights + noise + intercept) >= 0.5
    y = y.astype(np.int32)
    return X, y


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

params = {"ns": 100000, "nf": 8}

X, y = generate_X_y(params, rank)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rank
)

dpnp_X_train = dpnp.asarray(X_train, usm_type="device", sycl_queue=q)
dpnp_y_train = dpnp.asarray(y_train, usm_type="device", sycl_queue=q)
dpnp_X_test = dpnp.asarray(X_test, usm_type="device", sycl_queue=q)

model_spmd = LogisticRegression(solver='newton-cg')
model_spmd.fit(dpnp_X_train, dpnp_y_train)

y_predict = model_spmd.predict(dpnp_X_test)

print("Distributed LogisticRegression results:")
print("Coefficients on rank {}:\n{}:".format(rank, model_spmd.coef_))
print("Intercept on rank {}:\n{}:".format(rank, model_spmd.intercept_))
print("Ground truth (first 5 observations on rank {}):\n{}".format(rank, y_test[:5]))
print(
    "Classification results (first 5 observations on rank {}):\n{}".format(
        rank, y_predict[:5]
    )
)
print(
    "Accuracy for entire rank {} (2 classes): {}\n".format(
        rank, accuracy_score(y_test, dpnp.asnumpy(y_predict))
    )
)
