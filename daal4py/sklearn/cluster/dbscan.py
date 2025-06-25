# ==============================================================================
# Copyright 2014 Intel Corporation
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

import numbers

import numpy as np
from scipy import sparse as sp
from sklearn.cluster import DBSCAN as DBSCAN_original
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

import daal4py

from .._n_jobs_support import control_n_jobs
from .._utils import PatchingConditionsChain, getFPType, make2d, sklearn_check_version
from ..utils.validation import check_feature_names

if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
    from sklearn.utils import check_scalar


def _daal_dbscan(X, eps=0.5, min_samples=5, sample_weight=None):
    ww = make2d(sample_weight) if sample_weight is not None else None
    XX = make2d(X)

    fpt = getFPType(XX)  # codespell:ignore fpt
    alg = daal4py.dbscan(
        method="defaultDense",
        fptype=fpt,  # codespell:ignore fpt
        epsilon=float(eps),
        minObservations=int(min_samples),
        memorySavingMode=False,
        resultsToCompute="computeCoreIndices",
    )

    daal_res = alg.compute(XX, ww)
    assignments = daal_res.assignments.ravel()
    if daal_res.coreIndices is not None:
        core_ind = daal_res.coreIndices.ravel()
    else:
        core_ind = np.array([], dtype=np.intc)

    return (core_ind, assignments)


@control_n_jobs(decorated_methods=["fit"])
class DBSCAN(DBSCAN_original):
    __doc__ = DBSCAN_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**DBSCAN_original._parameter_constraints}

    def __init__(
        self,
        eps=0.5,
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

    def fit(self, X, y=None, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        elif sklearn_check_version("1.1"):
            check_scalar(
                self.eps,
                "eps",
                target_type=numbers.Real,
                min_val=0.0,
                include_boundaries="neither",
            )
            check_scalar(
                self.min_samples,
                "min_samples",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )
            check_scalar(
                self.leaf_size,
                "leaf_size",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )
            if self.p is not None:
                check_scalar(
                    self.p,
                    "p",
                    target_type=numbers.Real,
                    min_val=0.0,
                    include_boundaries="left",
                )
            if self.n_jobs is not None:
                check_scalar(self.n_jobs, "n_jobs", target_type=numbers.Integral)
        else:
            if self.eps <= 0.0:
                raise ValueError(f"eps == {self.eps}, must be > 0.0.")

        check_feature_names(self, X, reset=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        _patching_status = PatchingConditionsChain("sklearn.cluster.DBSCAN.fit")
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    self.algorithm in ["auto", "brute"],
                    f"'{self.algorithm}' algorithm is not supported. "
                    "Only 'auto' and 'brute' algorithms are supported",
                ),
                (
                    self.metric == "euclidean"
                    or (self.metric == "minkowski" and self.p == 2),
                    f"'{self.metric}' (p={self.p}) metric is not supported. "
                    "Only 'euclidean' or 'minkowski' with p=2 metrics are supported.",
                ),
                (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
            ]
        )

        _patching_status.write_log()
        if _dal_ready:
            X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
            core_ind, assignments = _daal_dbscan(
                X, self.eps, self.min_samples, sample_weight=sample_weight
            )
            self.core_sample_indices_ = core_ind
            self.labels_ = assignments
            self.components_ = np.take(X, core_ind, axis=0)
            self.n_features_in_ = X.shape[1]
            return self
        return super().fit(X, y, sample_weight=sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        return super().fit_predict(X, y, sample_weight)

    fit.__doc__ = DBSCAN_original.fit.__doc__
    fit_predict.__doc__ = DBSCAN_original.fit_predict.__doc__
