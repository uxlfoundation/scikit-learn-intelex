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

# sklearnex RF example for distributed systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./random_forest_classifier_spmd.py

import dpctl
import dpnp
import numpy as np
from mpi4py import MPI

from sklearnex.spmd.ensemble import RandomForestClassifier


def generate_X_y(par, seed):
    ns, nf = par["ns"], par["nf"]

    drng = np.random.default_rng(seed)
    data = drng.uniform(-1, 1, size=(ns, nf))
    resp = (data > 0) @ (2 ** np.arange(nf))

    return data, resp


params_train = {"ns": 1000000, "nf": 3}
params_test = {"ns": 100, "nf": 3}

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

X_train, y_train = generate_X_y(params_train, mpi_rank)
X_test, y_test = generate_X_y(params_test, mpi_rank + 777)

q = dpctl.SyclQueue("gpu")  # GPU

dpnp_X_train = dpnp.asarray(X_train, usm_type="device", sycl_queue=q)
dpnp_y_train = dpnp.asarray(y_train, usm_type="device", sycl_queue=q)
dpnp_X_test = dpnp.asarray(X_test, usm_type="device", sycl_queue=q)

rf = RandomForestClassifier(max_depth=2, random_state=0).fit(dpnp_X_train, dpnp_y_train)

pred = rf.predict(dpnp_X_test)

print("Random Forest classification results:")
print("Ground truth (first 5 observations on rank {}):\n{}".format(mpi_rank, y_test[:5]))
print(
    "Classification results (first 5 observations on rank {}):\n{}".format(
        mpi_rank, dpnp.asnumpy(pred)[:5]
    )
)
