# ===============================================================================
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
# ===============================================================================

import warnings

import numpy as np

from daal4py.sklearn._utils import get_dtype

from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import from_table, to_table
from ..utils.validation import _check_array, _check_X_y, _is_csr


class Louvain(BaseEstimator, ClusterMixin):

    def __init__(self, resolution=1.0, *, tol=0.0001, max_iter=10):
        self.resolution = resolution
        self.tol = tol
        self.max_iter = max_iter

    def _get_onedal_params(self, dtype=np.float64):
        return {
            "fptype": dtype,
            "method": "by_default",
            "resolution": float(self.resolution),
            "accuracy_threshold": float(self.tol),
            "max_iteration_count": int(self.max_iter),
        }

    def fit(self, X, y=None, queue=None):
        # queue is only included to match convention for all onedal estimators
        if queue is not None:
            warnings.warn("Louvain is implemented only for CPU")
        assert _is_csr(X), "input must be CSR sparse"
        # limitations in oneDAL's shared object force the topology to double type
        if y is None:
            X = _check_array(X, accept_sparse="csr", dtype=np.float64)
        else:
            X, y = _check_X_y(X, y, accept_sparse="csr", dtype=np.float64)
            y = y.astype(np.int64)  # restriction by oneDAL initial partition

        module = self._get_backend("louvain", "vertex_partitioning", None)

        # conversion of a scipy csr to dal csr_table will have incorrect dtypes and indices
        # must be done via three tables with types double, int32, int64 for oneDAL graph type
        data = to_table(X.data, X.indices, X.indptr) if y is None else to_table(X.data, X.indices, X.indptr, y)
        params = self._get_onedal_params(data[0].dtype)
        result = module.vertex_partitioning(params, *data)
        self.labels_ = from_table(result.labels).ravel()
        self.modularity_ = float(result.modularity)
        self.community_count_ = int(result.community_count)
        self.n_features_in_ = X.shape[1]
        return self
