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

from daal4py.sklearn._utils import daal_check_version
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM
from sklearnex._config import config_context, get_config

from ..datatypes import from_table, to_table
from ..utils.validation import _check_array

if daal_check_version((2023, "P", 200)):

    def force_host_if_csr(func):
        """
        Decorator to force computation on the host if the input data is in CSR format.

        CSR (Compressed Sparse Row) format is not supported on GPU. This decorator ensures that
        the computation is performed on the host (CPU) when the input data is in CSR format.
        It is necessary that `self.is_csr` is to the correct value (True or False) before calling.
        It temporarily modifies the configuration to target the CPU and removes the global SYCL queue
        to ensure the computation is executed on the host. The original configuration is restored upon exit.

        Parameters:
        func (function): The function to be decorated.

        Returns:
        function: The wrapped function with the modified configuration if the input data is in CSR format.
        """
        config = get_config()
        # `auto` rather than `cpu` is necessary in order to force host when a SYCL cpu target is unavailable
        config["target_offload"] = "auto"

        def wrapper(self, *args, **kwargs):
            if self.is_csr:
                with (
                    config_context(**config),
                    QM.manage_global_queue(None, None),
                ):
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    class KMeansInit:
        """
        KMeansInit oneDAL implementation.
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

        @bind_default_backend("kmeans_init.init", lookup_name="compute")
        def backend_compute(self, params, X_table): ...

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

        def _compute_raw(self, X_table, dtype=np.float32):
            params = self._get_onedal_params(dtype)
            result = self.backend_compute(params, X_table)
            return result.centroids

        def _compute(self, X):
            _, X_table, dtype = self._get_params_and_input(X, queue=QM.get_global_queue())
            centroids = self._compute_raw(X_table, dtype)
            return from_table(centroids)

        @force_host_if_csr
        def compute_raw(self, X_table, dtype=np.float32, queue=None):
            # no @supports_queue decorator here, because we only accept X_table that has no queue information
            return self._compute_raw(X_table, dtype)

        @force_host_if_csr
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
