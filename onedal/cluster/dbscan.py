# ===============================================================================
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
# ===============================================================================

import numpy as np

from daal4py.sklearn._utils import get_dtype, make2d
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend

from ..common._mixin import ClusterMixin
from ..datatypes import from_table, to_table
from ..utils.validation import _check_array


class DBSCAN(ClusterMixin):
    def __init__(
        self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    @bind_default_backend("dbscan.clustering")
    def compute(self, params, data_table, weights_table): ...

    def _get_onedal_params(self, dtype=np.float32):
        return {
            "fptype": dtype,
            "method": "by_default",
            "min_observations": int(self.min_samples),
            "epsilon": float(self.eps),
            "mem_save_mode": False,
            "result_options": "core_observation_indices|responses",
        }

    @supports_queue
    def fit(self, X, y=None, sample_weight=None, queue=None):
        X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        sample_weight = make2d(sample_weight) if sample_weight is not None else None
        X_table, sample_weight_table = to_table(X, sample_weight, queue=queue)

        params = self._get_onedal_params(X_table.dtype)
        result = self.compute(params, X_table, sample_weight_table)

        self.labels_ = from_table(result.responses).ravel()
        if result.core_observation_indices is not None:
            self.core_sample_indices_ = from_table(
                result.core_observation_indices
            ).ravel()
        else:
            self.core_sample_indices_ = np.array([], dtype=np.intc)
        self.components_ = np.take(X, self.core_sample_indices_, axis=0)
        self.n_features_in_ = X.shape[1]
        return self
