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
from scipy.sparse import issparse
from sklearn.utils import check_random_state

from daal4py.sklearn._utils import daal_check_version, get_dtype

from ..common._base import BaseEstimator as onedal_BaseEstimator
from ..datatypes import from_table, to_table
from ..utils.validation import _check_array

if daal_check_version((2023, "P", 200)):

    class KMeansInit(onedal_BaseEstimator):
        """
        KMeansInit oneDAL implementation.
        """

        def __init__(
            self,
            cluster_count,
            seed=777,
            local_trials_count=None,
            algorithm="plus_plus_dense",
        ):
            self.cluster_count = cluster_count
            self.seed = seed
            self.local_trials_count = local_trials_count
            self.algorithm = algorithm

            if local_trials_count is None:
                self.local_trials_count = 2 + int(np.log(cluster_count))
            else:
                self.local_trials_count = local_trials_count

        def _get_onedal_params(self, dtype=np.float32):
            return {
                "fptype": dtype,
                "local_trials_count": self.local_trials_count,
                "method": self.algorithm,
                "seed": self.seed,
                "cluster_count": self.cluster_count,
            }

        def _get_params_and_input(self, X, queue):
            X = _check_array(
                X,
                dtype=[np.float64, np.float32],
                accept_sparse="csr",
                force_all_finite=False,
            )
            X = to_table(X, queue=queue)
            params = self._get_onedal_params(X.dtype)
            return (params, X, X.dtype)

        def _compute_raw(self, X_table, module, policy, dtype=np.float32):
            params = self._get_onedal_params(dtype)

            result = module.compute(policy, params, X_table)

            return result.centroids

        def _compute(self, X, module, queue):
            policy = self._get_policy(queue, X)
            # oneDAL KMeans Init for sparse data does not have GPU support
            if issparse(X):
                policy = self._get_policy(None, None)
            _, X_table, dtype = self._get_params_and_input(X, queue)

            centroids = self._compute_raw(X_table, module, policy, dtype)

            return from_table(centroids)

        def compute_raw(self, X_table, policy, dtype=np.float32):
            return self._compute_raw(
                X_table, self._get_backend("kmeans_init", "init", None), policy, dtype
            )

        def compute(self, X, queue=None):
            return self._compute(X, self._get_backend("kmeans_init", "init", None), queue)

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
            ).compute(X, queue),
            np.full(n_clusters, -1),
        )
