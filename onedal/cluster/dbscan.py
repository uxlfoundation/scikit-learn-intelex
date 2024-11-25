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
from sklearn.utils import check_array

from onedal.utils._array_api import get_dtype, make2d

from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils._array_api import (
    _asarray,
    _convert_to_numpy,
    _ravel,
    get_dtype,
    get_namespace,
    make2d,
    sklearn_array_api_dispatch,
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

    @sklearn_array_api_dispatch()
    def _fit(self, X, sua_iface, xp, is_array_api_compliant, y, sample_weight, queue):
        policy = self._get_policy(queue, X)
        # TODO:
        # check on dispatching and warn.
        # using scikit-learn primitives will require array_api_dispatch=True
        X = check_array(X, accept_sparse="csr", dtype=[xp.float64, xp.float32])

        sample_weight = make2d(sample_weight) if sample_weight is not None else None
        X = make2d(X)
        # X_device = X.device if xp else None

        # TODO:
        # move to _convert_to_supported to do astype conversion
        # at once.
        types = [xp.float32, xp.float64]

        # TODO:
        # could be impossible, if device doesn't support fp65
        # make sense update _convert_to_supported for it.
        if get_dtype(X) not in types:
            X = X.astype(xp.float64)
        X = _convert_to_supported(policy, X, xp=xp)
        # TODO:
        # remove if not required.
        sample_weight = (
            _convert_to_supported(policy, sample_weight, xp=xp)
            if sample_weight is not None
            else None
        )
        dtype = get_dtype(X)
        params = self._get_onedal_params(xp, dtype)
        X_table = to_table(X)
        sample_weight_table = to_table(sample_weight)

        result = self._get_backend("dbscan", "clustering", None).compute(
            policy, params, X_table, sample_weight_table
        )
        self.labels_ = _ravel(
            from_table(result.responses, sua_iface=sua_iface, sycl_queue=queue, xp=xp), xp
        )
        if (
            result.core_observation_indices is not None
            and not result.core_observation_indices.kind == "empty"
        ):
            self.core_sample_indices_ = _ravel(
                from_table(
                    result.core_observation_indices,
                    sycl_queue=queue,
                    sua_iface=sua_iface,
                    xp=xp,
                ),
                xp,
            )
        else:
            # TODO:
            # self.core_sample_indices_ = _asarray([], xp, sycl_queue=queue, dtype=xp.int32)
            if sua_iface:
                self.core_sample_indices_ = xp.asarray(
                    [], sycl_queue=queue, dtype=xp.int32
                )
            else:
                self.core_sample_indices_ = xp.asarray([], dtype=xp.int32)
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
        sua_iface, xp, is_array_api_compliant = get_namespace(X)
        # TODO:
        # update for queue getting.
        if sua_iface:
            queue = X.sycl_queue
        return super()._fit(
            X, sua_iface, xp, is_array_api_compliant, y, sample_weight, queue
        )
