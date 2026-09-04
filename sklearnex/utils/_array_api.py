# ==============================================================================
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
# ==============================================================================

"""Tools to support array API."""

import math
from collections.abc import Callable
from typing import Union

import scipy.linalg as linalg
from sklearn.covariance import log_likelihood as _sklearn_log_likelihood
from sklearn.utils._array_api import get_namespace

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils._array_api import _is_numpy_namespace

from ..base import Tags, oneDALEstimator


def _enable_array_api(original_class: type[oneDALEstimator]) -> type[oneDALEstimator]:
    def __sklearn_tags__(self) -> Tags:
        sktags = super(original_class, self).__sklearn_tags__()
        sktags.onedal_array_api = True
        return sktags

    original_class.__sklearn_tags__ = __sklearn_tags__

    return original_class


def enable_array_api(
    class_or_str: Union[type[oneDALEstimator], str],
) -> Union[type[oneDALEstimator], Callable]:
    """Enable sklearnex to use dpnp or array API inputs in oneDAL offloading.

    This wrapper sets the proper flags/tags for the sklearnex infrastructure
    to maintain the data framework, as the estimator can use it natively.

    Parameters
    ----------
    class_or_str : oneDALEstimator subclass or str
        Class which should enable data zero-copy support in sklearnex. By
        default it will enable for sklearn versions >1.3. If the wrapper is
        decorated with an argument, it must be a string defining the oldest
        sklearn version where array API support begins.

    Returns
    -------
    cls or wrapper : modified oneDALEstimator subclass or wrapper
        Estimator class or wrapper.

    Examples
    --------
    @enable_array_api  # default array API support
    class PCA():
        ...

    @enable_array_api("1.5")  # array API support for sklearn > 1.5
    class Ridge():
        ...
    """
    if isinstance(class_or_str, str):
        # enable array_api for the estimator for a given sklearn version str
        if sklearn_check_version(class_or_str):
            return _enable_array_api
        else:
            # do not apply the wrapper as it is not supported
            return lambda x: x
    else:
        # default setting (apply array_api enablement for sklearn >=1.3)
        return _enable_array_api(class_or_str)


def _pinvh(a, atol=None, rtol=None, lower=True, return_rank=False, check_finite=True):
    # array API enabled pinvh implementation, via direct translation of scipy.linalg.pinhv
    # this should be considered a temporary stopgap until implemented in oneDAL
    xp, _ = get_namespace(a)
    # fall back to scipy if the namespace is of a numpy origin
    if _is_numpy_namespace(xp):
        return linalg.pinvh(
            a,
            atol=atol,
            rtol=rtol,
            lower=lower,
            return_rank=return_rank,
            check_finite=check_finite,
        )

    if check_finite:
        raise NotImplementedError("finite checking does not occur in sklearnex's pinvh")

    s, u = xp.linalg.eigh(a)
    maxS = xp.max(xp.abs(s))

    atol = 0.0 if atol is None else atol
    rtol = max(a.shape) * xp.finfo(u.dtype).eps if (rtol is None) else rtol

    if (atol < 0.0) or (rtol < 0.0):
        raise ValueError("atol and rtol values must be positive.")

    val = atol + maxS * rtol
    above_cutoff = xp.nonzero(abs(s) > val)[0]

    psigma_diag = 1.0 / xp.take(s, above_cutoff)
    u = xp.take(u, above_cutoff, axis=1)

    uconj = xp.conj(u) if xp.isdtype(u.dtype, kind="complex floating") else u

    B = (u * psigma_diag) @ uconj.T

    if return_rank:
        return B, len(psigma_diag)
    else:
        return B


def log_likelihood(emp_cov, precision):
    # this is to compensate for a lack of array API support in sklearn
    # even though it exists for ``fast_logdet``
    xp, _ = get_namespace(emp_cov, precision)
    p = precision.shape[0]
    # extract sklearn.utils.extmath.fast_logdet for dpnp support
    sign, ld = xp.linalg.slogdet(precision)
    if not sign > 0:
        ld = -xp.inf
    log_likelihood_ = -xp.sum(emp_cov * precision) + ld
    log_likelihood_ -= p * math.log(2 * math.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_


log_likelihood.__doc__ = _sklearn_log_likelihood.__doc__
