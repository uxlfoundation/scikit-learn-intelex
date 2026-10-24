# ===============================================================================
# Copyright 2021 Intel Corporation
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

import numbers

from sklearn.cluster import DBSCAN as _sklearn_DBSCAN
from sklearn.utils._array_api import get_namespace

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, is_sparse
from onedal.cluster import DBSCAN as onedal_DBSCAN
from onedal.utils._array_api import _is_numpy_namespace

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import enable_array_api
from ..utils.validation import _check_sample_weight, validate_data

# scikit-learn's names of the distances oneDAL computes, as oneDAL names them
_METRIC_ALIASES = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "l1": "manhattan",
    "cityblock": "manhattan",
    "chebyshev": "chebyshev",
    "minkowski": "minkowski",
    "cosine": "cosine",
}

# the degrees for which the Minkowski distance is one of the dedicated metrics
_MINKOWSKI_DEGREES = {1: "manhattan", 2: "euclidean"}

# oneDAL computes only the euclidean distance before 2026.3
_ONEDAL_HAS_METRICS = daal_check_version((2026, "P", 300))
_SUPPORTED_METRICS = (
    "the euclidean, manhattan, chebyshev, minkowski (p >= 1) and cosine distances"
    if _ONEDAL_HAS_METRICS
    else "'euclidean' or 'minkowski' with p=2 metrics"
)


def _onedal_metric(metric, p, metric_params):
    """Map scikit-learn's distance onto the one oneDAL computes identically.

    Parameters
    ----------
    metric : str or callable
        The ``metric`` parameter of the estimator.
    p : float or None
        The ``p`` parameter of the estimator.
    metric_params : dict or None
        The ``metric_params`` parameter of the estimator.

    Returns
    -------
    metric : tuple of (str, float) or None
        oneDAL's metric and Minkowski degree, or None when scikit-learn's result
        or error cannot be reproduced.
    """
    if metric_params:
        return None
    if not _ONEDAL_HAS_METRICS:
        if metric == "euclidean" or (metric == "minkowski" and p == 2):
            return "euclidean", 2.0
        return None
    if not isinstance(metric, str) or metric not in _METRIC_ALIASES:
        return None

    name = _METRIC_ALIASES[metric]
    if name != "minkowski":
        return name, 2.0
    # scikit-learn raises for 'p=None' and warns for 'p < 1'
    if not isinstance(p, numbers.Real) or not p >= 1:
        return None
    if p in _MINKOWSKI_DEGREES:
        return _MINKOWSKI_DEGREES[p], 2.0
    return "minkowski", float(p)


@enable_array_api
@control_n_jobs(decorated_methods=["fit"])
class DBSCAN(oneDALEstimator, _sklearn_DBSCAN):
    __doc__ = _sklearn_DBSCAN.__doc__

    _parameter_constraints: dict = {**_sklearn_DBSCAN._parameter_constraints}

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
        super(DBSCAN, self).__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    _onedal_dbscan = staticmethod(onedal_DBSCAN)

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        xp, _ = get_namespace(X, y, sample_weight)
        X = validate_data(self, X, accept_sparse="csr", dtype=[xp.float64, xp.float32])
        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=[xp.float64, xp.float32]
            )

        metric, degree = _onedal_metric(self.metric, self.p, self.metric_params)
        onedal_params = {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": metric,
            "metric_params": self.metric_params,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": degree,
            "n_jobs": self.n_jobs,
        }
        self._onedal_estimator = self._onedal_dbscan(**onedal_params)

        self._onedal_estimator.fit(X, y=y, sample_weight=sample_weight, queue=queue)
        if self._onedal_estimator.core_sample_indices_ is None:
            kwargs = {"dtype": xp.int32}  # always the same
            if not _is_numpy_namespace(xp):
                kwargs["device"] = X.device
            self.core_sample_indices_ = xp.empty((0,), **kwargs)
        else:
            self.core_sample_indices_ = self._onedal_estimator.core_sample_indices_

        self.components_ = xp.take(X, self.core_sample_indices_, axis=0)
        self.labels_ = self._onedal_estimator.labels_
        self.n_features_in_ = X.shape[1]

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.cluster.{class_name}.{method_name}"
        )
        if method_name == "fit":
            X = data[0]
            patching_status.and_conditions(
                [
                    (
                        self.algorithm in ["auto", "brute"],
                        f"'{self.algorithm}' algorithm is not supported. "
                        "Only 'auto' and 'brute' algorithms are supported",
                    ),
                    (
                        _onedal_metric(self.metric, self.p, self.metric_params)
                        is not None,
                        f"'{self.metric}' metric with p={self.p} and metric_params="
                        f"{self.metric_params} is not supported. Only "
                        f"{_SUPPORTED_METRICS} without metric_params are supported.",
                    ),
                    (not is_sparse(X), "X is sparse. Sparse input is not supported."),
                ]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def fit(self, X, y=None, sample_weight=None):
        self._validate_params()
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_DBSCAN.fit,
            },
            X,
            y,
            sample_weight,
        )

        return self

    fit.__doc__ = _sklearn_DBSCAN.fit.__doc__
