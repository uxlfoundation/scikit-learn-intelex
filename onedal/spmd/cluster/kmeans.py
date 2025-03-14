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

from ..._device_offload import support_input_format, supports_queue
from ...cluster import KMeans as KMeans_Batch
from ...cluster import KMeansInit as KMeansInit_Batch
from ...common._backend import bind_spmd_backend
from ...spmd.basic_statistics import BasicStatistics


class KMeansInit(KMeansInit_Batch):
    """
    KMeansInit oneDAL implementation for SPMD iface.

    Unlike the batch version, the SPMD library provides GPU support.
    """

    def compute_raw(self, X_table, dtype=np.float32, queue=None):
        # no @supports_queue decorator here, because we only accept X_table that has no queue information
        return self._compute_raw(X_table, dtype)

    @supports_queue
    def compute(self, X, queue=None):
        return self._compute(X)

    @bind_spmd_backend("kmeans_init.init", lookup_name="compute")
    def backend_compute(self, params, data): ...


class KMeans(KMeans_Batch):
    def _get_basic_statistics_backend(self, result_options):
        return BasicStatistics(result_options)

    def _get_kmeans_init(self, cluster_count, seed, algorithm, is_csr):
        return KMeansInit(
            cluster_count=cluster_count,
            seed=seed,
            algorithm=algorithm,
            is_csr=is_csr,
        )

    @bind_spmd_backend("kmeans.clustering")
    def train(self, params, X_table, centroids_table): ...

    @bind_spmd_backend("kmeans.clustering")
    def infer(self, params, model, centroids_table): ...

    @support_input_format
    def fit(self, X, y=None, queue=None):
        return super().fit(X, y, queue=queue)

    @support_input_format
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

    @support_input_format
    def fit_predict(self, X, y=None, queue=None):
        return super().fit_predict(X, queue=queue)
