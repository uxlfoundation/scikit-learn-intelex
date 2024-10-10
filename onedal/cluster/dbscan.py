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
from sklearn import set_config
from sklearn.utils import check_array

from onedal.datatypes._data_conversion import get_dtype, make2d

from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils._array_api import (
    _asarray,
    _convert_to_numpy,
    get_dtype,
    get_namespace,
    make2d,
)


class BaseDBSCAN(BaseEstimator, ClusterMixin):
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

    def _get_onedal_params(self, xp, dtype):
        return {
            "fptype": "float" if dtype == xp.float32 else "double",
            "method": "by_default",
            "min_observations": int(self.min_samples),
            "epsilon": float(self.eps),
            "mem_save_mode": False,
            "result_options": "core_observation_indices|responses",
        }

    def _fit(self, X, xp, is_array_api_compliant, y, sample_weight, queue):
        policy = self._get_policy(queue, X)
        # just for debug.
        set_config(array_api_dispatch=True)
        # TODO:
        # check on dispatching and warn.
        # using scikit-learn primitives will require array_api_dispatch=True
        X = check_array(X, accept_sparse="csr", dtype=[xp.float64, xp.float32])
        sample_weight = make2d(sample_weight) if sample_weight is not None else None
        X = make2d(X)
        # X_device = X.device if xp else None

        types = [xp.float32, xp.float64]
        if get_dtype(X) not in types:
            X = X.astype(xp.float64)
        # TODO:
        # update iface
        # X = _convert_to_supported(policy, X, xp)
        # TODO:
        # remove if not required.
        sample_weight = (
            _convert_to_supported(policy, sample_weight, xp)
            if sample_weight is not None
            else None
        )
        dtype = get_dtype(X)
        params = self._get_onedal_params(xp, dtype)

        # Since `to_table` data management enabled only for numpy host inputs,
        # copy data into numpy host for to_table conversion.
        result = self._get_backend("dbscan", "clustering", None).compute(
            policy, params, to_table(_convert_to_numpy(X, xp=xp)), to_table(None)
        )

        # Since `from_table` data management enabled only for numpy host,
        # copy data from numpy host output to xp namespace array.
        self.labels_ = _asarray(
            from_table(result.responses).reshape(-1), xp=xp, sycl_queue=queue
        )
        if result.core_observation_indices is not None:
            # self.core_sample_indices_ = _asarray(from_table(result.core_observation_indices).reshape(-1), xp=xp, sycl_queue=queue)
            self.core_sample_indices_ = xp.asarray(
                from_table(result.core_observation_indices).reshape(-1),
                usm_type="device",
                sycl_queue=queue,
            )
        else:
            self.core_sample_indices_ = xp.array([], dtype=xp.int32)
        self.components_ = xp.take(X, self.core_sample_indices_, axis=0)
        self.n_features_in_ = X.shape[1]
        return self


class DBSCAN(BaseDBSCAN):
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

    def fit(self, X, y=None, sample_weight=None, queue=None):
        xp, is_array_api_compliant = get_namespace(X)
        # TODO:
        # update for queue getting.
        queue = X.sycl_queue
        return super()._fit(X, xp, is_array_api_compliant, y, sample_weight, queue)
