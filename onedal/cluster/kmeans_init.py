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

import numpy as np
from sklearn.utils import check_random_state

from .. import _default_backend
from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..datatypes import from_table, to_table
from ..utils import _sycl_queue_manager as QM


class KMeansInit:
    """KMeansInit oneDAL implementation.

    Parameters
    ----------
    cluster_count : int
        Number of centroids to compute.

    seed : int, default=777
        Seed of the random number generator used by the sampling methods.

    local_trials_count : int, default=None
        Number of trials performed on each sampling step. If None,
        ``2 + floor(log(cluster_count))`` trials are used.

    algorithm : str, default="plus_plus_dense"
        Centroid initialization method. Available methods: "dense",
        "by_default", "plus_plus_dense", "random_dense", "plus_plus_csr",
        "random_csr".

    is_csr : bool, default=False
        If True, the input data is expected in CSR format and the
        computation is forced onto the host.
    """

    def __init__(
        self,
        cluster_count,
        seed=777,
        local_trials_count=None,
        algorithm="plus_plus_dense",
        is_csr=False,
    ):
        self.cluster_count = cluster_count
        self.seed = seed
        self.local_trials_count = local_trials_count
        self.algorithm = algorithm
        self.is_csr = is_csr

        if local_trials_count is None:
            self.local_trials_count = 2 + int(np.log(cluster_count))
        else:
            self.local_trials_count = local_trials_count

    # force use of `no_policy` as it must be directly managed for csr data
    @bind_default_backend("kmeans_init.init", lookup_name="compute", no_policy=True)
    def _backend_compute(self, policy, params, X_table): ...

    # it checks for csr data and forces computation on host
    def backend_compute(self, params, X_table):
        policy = _default_backend.get_policy(
            None if self.is_csr else QM.get_global_queue()
        )
        return self._backend_compute(policy, params, X_table)

    def _get_onedal_params(self, dtype=np.float32):
        return {
            "fptype": dtype,
            "local_trials_count": self.local_trials_count,
            "method": self.algorithm,
            "seed": self.seed,
            "cluster_count": self.cluster_count,
        }

    def _compute_raw(self, X_table, dtype=np.float32, queue=None):
        params = self._get_onedal_params(dtype)
        result = self.backend_compute(params, X_table)
        return result.centroids

    def _compute(self, X):
        X_table = to_table(X, queue=QM.get_global_queue())
        centroids = self._compute_raw(X_table, X_table.dtype)
        return from_table(centroids)

    def compute_raw(self, X_table, dtype=np.float32, queue=None):
        # no @supports_queue decorator here, because we only accept X_table that has no queue information
        return self._compute_raw(X_table, dtype, queue)

    @supports_queue
    def compute(self, X, queue=None):
        return self._compute(X)


def kmeans_plusplus(
    X,
    n_clusters,
    *,
    x_squared_norms=None,
    random_state=None,
    n_local_trials=None,
    queue=None,
):
    random_seed = check_random_state(random_state).tomaxint()
    return (
        KMeansInit(
            n_clusters, seed=random_seed, local_trials_count=n_local_trials
        ).compute(X, queue=queue),
        np.full(n_clusters, -1),
    )
