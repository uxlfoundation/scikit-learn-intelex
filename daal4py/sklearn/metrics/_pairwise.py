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

import numpy as np
from sklearn.metrics import pairwise_distances as pairwise_distances_original
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.utils._param_validation import (
    Hidden,
    Integral,
    StrOptions,
    validate_params,
)

import daal4py
from daal4py.sklearn.utils.validation import _daal_check_array

from .._utils import (
    PatchingConditionsChain,
    check_is_array_api,
    getFPType,
    is_sparse,
    sklearn_check_version,
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


# Comment 2026-03-05: this function originally was a copy-paste from an older
# version of scikit-learn with some parts replaced with oneDAL calls. Later
# on, it was changed to offload directly to scikit-learn in cases where there
# would be no oneDAL involvement, so many code branches here might be unreachable.
def _pairwise_distances(
    X, Y=None, metric="euclidean", *, n_jobs=None, force_all_finite=True, **kwds
):
    _patching_status = PatchingConditionsChain("sklearn.metrics.pairwise_distances")
    _dal_ready = _patching_status.and_conditions(
        [
            (
                metric == "cosine" or metric == "correlation",
                f"'{metric}' metric is not supported. "
                "Only 'cosine' and 'correlation' metrics are supported.",
            ),
            (Y is None, "Second feature array is not supported."),
            (not is_sparse(X), "X is sparse. Sparse input is not supported."),
            (not check_is_array_api(X), "Array API inputs are not supported."),
        ]
    )
    if not _dal_ready:
        _patching_status.write_log()
        return _sklearn_pairwise_distances(
            X,
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            force_all_finite=force_all_finite,
            **kwds,
        )

    X = _daal_check_array(
        X, accept_sparse=["csr", "csc", "coo"], force_all_finite=force_all_finite
    )
    _dal_ready = _patching_status.and_conditions(
        [
            (
                X.dtype == np.float64,
                f"{X.dtype} X data type is not supported. Only np.float64 is supported.",
            ),
        ]
    )
    _patching_status.write_log()
    if not _dal_ready:
        return _sklearn_pairwise_distances(
            X,
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            force_all_finite=force_all_finite,
            **kwds,
        )
    # Reaching this point guarantees metric is "cosine" or "correlation" and
    # X is float64, so oneDAL always handles the computation from here.
    if metric == "cosine":
        return _daal4py_cosine_distance_dense(X)
    if metric == "correlation":
        return _daal4py_correlation_distance_dense(X)
    raise ValueError(f"'{metric}' distance is wrong for daal4py.")


# logic to deprecate `force_all_finite` from sklearn:
# it was renamed to `ensure_all_finite` since 1.6 and will be removed in 1.8
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
if sklearn_check_version("1.8"):
    del pairwise_distances_parameters["force_all_finite"]

    # Note: these functions '_sklearn_pairwise_distances' are designed
    # to have the same signature as the internal '_pairwise_distances'
    # defined here, so that it could be called from that function when
    # falling back to scikit-learn with the same arguments regardless
    # of version.
    def _sklearn_pairwise_distances(
        X,
        Y=None,
        metric="euclidean",
        *,
        n_jobs=None,
        force_all_finite=True,
        **kwds,
    ):
        return pairwise_distances_original(
            X,
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            ensure_all_finite=force_all_finite,
            **kwds,
        )

    def pairwise_distances(
        X,
        Y=None,
        metric="euclidean",
        *,
        n_jobs=None,
        ensure_all_finite=True,
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

    def _sklearn_pairwise_distances(
        X,
        Y=None,
        metric="euclidean",
        *,
        n_jobs=None,
        force_all_finite=True,
        **kwds,
    ):
        return pairwise_distances_original(
            X,
            Y=Y,
            metric=metric,
            n_jobs=n_jobs,
            force_all_finite="deprecated",
            ensure_all_finite=force_all_finite,
            **kwds,
        )

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


pairwise_distances = validate_params(
    pairwise_distances_parameters,
    prefer_skip_nested_validation=True,
)(pairwise_distances)

pairwise_distances.__doc__ = pairwise_distances_original.__doc__
