# ===============================================================================
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
# ===============================================================================

import warnings
from functools import partial

import numpy as np
from joblib import effective_n_jobs
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import pairwise_distances as pairwise_distances_original
from sklearn.metrics.pairwise import (
    _VALID_METRICS,
    PAIRWISE_BOOLEAN_FUNCTIONS,
    PAIRWISE_DISTANCE_FUNCTIONS,
    _pairwise_callable,
    _parallel_pairwise,
    check_pairwise_arrays,
)
from sklearn.utils.validation import check_non_negative

try:
    from sklearn.metrics.pairwise import _precompute_metric_params
except ImportError:

    def _precompute_metric_params(*args, **kwrds):
        return dict()


from scipy.sparse import issparse
from scipy.spatial import distance

import daal4py
from daal4py.sklearn.utils.validation import _daal_check_array

from .._utils import PatchingConditionsChain, getFPType, sklearn_check_version

if sklearn_check_version("1.3"):
    from sklearn.utils._param_validation import (
        Hidden,
        Integral,
        StrOptions,
        validate_params,
    )


def _daal4py_cosine_distance_dense(X):
    X_fptype = getFPType(X)
    alg = daal4py.cosine_distance(fptype=X_fptype, method="defaultDense")
    res = alg.compute(X)
    return res.cosineDistance


def _daal4py_correlation_distance_dense(X):
    X_fptype = getFPType(X)
    alg = daal4py.correlation_distance(fptype=X_fptype, method="defaultDense")
    res = alg.compute(X)
    return res.correlationDistance


def _pairwise_distances(
    X, Y=None, metric="euclidean", *, n_jobs=None, force_all_finite=True, **kwds
):
    if metric not in _VALID_METRICS and not callable(metric) and metric != "precomputed":
        raise ValueError(
            "Unknown metric %s. Valid metrics are %s, or 'precomputed', "
            "or a callable" % (metric, _VALID_METRICS)
        )

    X = _daal_check_array(
        X, accept_sparse=["csr", "csc", "coo"], force_all_finite=force_all_finite
    )

    _patching_status = PatchingConditionsChain("sklearn.metrics.pairwise_distances")
    _dal_ready = _patching_status.and_conditions(
        [
            (
                metric == "cosine" or metric == "correlation",
                f"'{metric}' metric is not supported. "
                "Only 'cosine' and 'correlation' metrics are supported.",
            ),
            (Y is None, "Second feature array is not supported."),
            (not issparse(X), "X is sparse. Sparse input is not supported."),
            (
                X.dtype == np.float64,
                f"{X.dtype} X data type is not supported. Only np.float64 is supported.",
            ),
        ]
    )
    _patching_status.write_log()
    if _dal_ready:
        if metric == "cosine":
            return _daal4py_cosine_distance_dense(X)
        if metric == "correlation":
            return _daal4py_correlation_distance_dense(X)
        raise ValueError(f"'{metric}' distance is wrong for daal4py.")
    if metric == "precomputed":
        if sklearn_check_version("1.6"):
            X, _ = check_pairwise_arrays(
                X, Y, precomputed=True, ensure_all_finite=force_all_finite
            )
        else:
            X, _ = check_pairwise_arrays(
                X, Y, precomputed=True, force_all_finite=force_all_finite
            )
        whom = (
            "`pairwise_distances`. Precomputed distance "
            " need to have non-negative values."
        )
        check_non_negative(X, whom=whom)
        return X
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        if sklearn_check_version("1.6"):
            func = partial(
                _pairwise_callable,
                metric=metric,
                ensure_all_finite=force_all_finite,
                **kwds,
            )
        else:
            func = partial(
                _pairwise_callable,
                metric=metric,
                force_all_finite=force_all_finite,
                **kwds,
            )
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not" " support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else None

        if dtype == bool and (X.dtype != bool or (Y is not None and Y.dtype != bool)):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        if sklearn_check_version("1.6"):
            X, Y = check_pairwise_arrays(
                X, Y, dtype=dtype, ensure_all_finite=force_all_finite
            )
        else:
            X, Y = check_pairwise_arrays(
                X, Y, dtype=dtype, force_all_finite=force_all_finite
            )

        # precompute data-derived metric params
        params = _precompute_metric_params(X, Y, metric=metric, **kwds)
        kwds.update(**params)

        if effective_n_jobs(n_jobs) == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric, **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)


# logic to deprecate `force_all_finite` from sklearn:
# it was renamed to `ensure_all_finite` since 1.6 and will be removed in 1.8
if sklearn_check_version("1.3"):
    pairwise_distances_parameters = {
        "X": ["array-like", "sparse matrix"],
        "Y": ["array-like", "sparse matrix", None],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "n_jobs": [Integral, None],
        "force_all_finite": [
            "boolean",
            StrOptions({"allow-nan"}),
            Hidden(StrOptions({"deprecated"})),
        ],
        "ensure_all_finite": [
            "boolean",
            StrOptions({"allow-nan"}),
            Hidden(None),
        ],
    }
    if sklearn_check_version("1.6"):
        if sklearn_check_version("1.8"):
            del pairwise_distances_parameters["force_all_finite"]

            def pairwise_distances(
                X,
                Y=None,
                metric="euclidean",
                *,
                n_jobs=None,
                ensure_all_finite=None,
                **kwds,
            ):
                return _pairwise_distances(
                    X,
                    Y,
                    metric,
                    n_jobs=n_jobs,
                    force_all_finite=ensure_all_finite,
                    **kwds,
                )

        else:
            from sklearn.utils.deprecation import _deprecate_force_all_finite

            def pairwise_distances(
                X,
                Y=None,
                metric="euclidean",
                *,
                n_jobs=None,
                force_all_finite="deprecated",
                ensure_all_finite=None,
                **kwds,
            ):
                force_all_finite = _deprecate_force_all_finite(
                    force_all_finite, ensure_all_finite
                )
                return _pairwise_distances(
                    X, Y, metric, n_jobs=n_jobs, force_all_finite=force_all_finite, **kwds
                )

    else:
        del pairwise_distances_parameters["ensure_all_finite"]

        def pairwise_distances(
            X,
            Y=None,
            metric="euclidean",
            *,
            n_jobs=None,
            force_all_finite=True,
            **kwds,
        ):
            return _pairwise_distances(
                X,
                Y,
                metric,
                n_jobs=n_jobs,
                force_all_finite=force_all_finite,
                **kwds,
            )

    pairwise_distances = validate_params(
        pairwise_distances_parameters,
        prefer_skip_nested_validation=True,
    )(pairwise_distances)
else:
    pairwise_distances = _pairwise_distances
pairwise_distances.__doc__ = pairwise_distances_original.__doc__
