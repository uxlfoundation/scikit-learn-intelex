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

import warnings
from abc import ABC

import numpy as np

# from ..common._mixin import ClusterMixin
from sklearn.base import BaseEstimator, ClusterMixin

# from daal4py.sklearn._utils import daal_check_version, get_dtype
from daal4py.sklearn._utils import get_dtype
from onedal import _backend

# from sklearn.base import BaseEstimator
from ..common._policy import _get_policy
from ..datatypes import _convert_to_supported, from_table, to_table

# class BaseDBSCAN(ClusterMixin, BaseEstimator)


class BaseDBSCAN(ClusterMixin):
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

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_onedal_params(self, dtype=np.float32):
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "by_default",
            "min_observations": int(self.min_samples),
            "epsilon": float(self.eps),
            "mem_save_mode": False,
            "result_options": "core_observation_indices|responses",
        }

    def _fit(self, X, y, sample_weight, module, queue):
        policy = self._get_policy(queue, X)
        # X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        # ww = make2d(sample_weight) if sample_weight is not None else None
        # XX = make2d(X)
        # TODO:
        # sample_weight check.
        # if sample_weight is not None:
        #     sample_weight = _check_sample_weight(sample_weight, X)
        X_loc = np.asarray(X)
        types = [np.float32, np.float64]
        if get_dtype(X_loc) not in types:
            X_loc = X_loc.astype(np.float64)
        X_loc = _convert_to_supported(policy, X_loc)
        dtype = get_dtype(X_loc)
        params = self._get_onedal_params(dtype)
        result = module.compute(policy, params, to_table(X_loc), to_table(sample_weight))

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
        return super()._fit(X, y, sample_weight, _backend.dbscan.clustering, queue)
