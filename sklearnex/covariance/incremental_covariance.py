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

import numbers
import warnings
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.covariance import EmpiricalCovariance as _sklearn_EmpiricalCovariance
from sklearn.utils import gen_batches
from sklearn.utils.validation import _num_features, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from daal4py.sklearn.metrics import pairwise_distances
from onedal._device_offload import support_input_format, support_sycl_format
from onedal.covariance import (
    IncrementalEmpiricalCovariance as onedal_IncrementalEmpiricalCovariance,
)
from onedal.utils._array_api import _is_numpy_namespace

from .._config import config_context, get_config
from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain, _add_inc_serialization_note
from ..base import oneDALEstimator
from ..utils._array_api import _pinvh, enable_array_api, get_namespace, log_likelihood
from ..utils.validation import validate_data

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval

# This is a temporary workaround for issues with sklearnex._device_offload._get_host_inputs
# passing kwargs with sycl queues with other host data will cause failures
_mahalanobis = support_input_format(partial(pairwise_distances, metric="mahalanobis"))


@enable_array_api
@control_n_jobs(decorated_methods=["partial_fit", "fit", "_onedal_finalize_fit"])
class IncrementalEmpiricalCovariance(oneDALEstimator, BaseEstimator):
    """
    Incremental maximum likelihood covariance estimator.

    Estimator that allows for the estimation when the data are split into
    batches. The user can use the ``partial_fit`` method to provide a
    single batch of data or use the ``fit`` method to provide the entire
    dataset.

    Parameters
    ----------
    store_precision : bool, default=False
        Specifies if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide
        a balance between approximation accuracy and memory consumption.

    copy : bool, default=True
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e. the estimated mean.

    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to ``fit``, but increments across ``partial_fit`` calls.

    batch_size_ : int
        Inferred batch size from ``batch_size``.

    n_features_in_ : int
        Number of features seen during ``fit`` or ``partial_fit``.

    Notes
    -----
    Sparse data formats are not supported. Input dtype must be ``float32`` or ``float64``.

    %incremental_serialization_note%

    Examples
    --------
    >>> import numpy as np
    >>> from sklearnex.covariance import IncrementalEmpiricalCovariance
    >>> inccov = IncrementalEmpiricalCovariance(batch_size=1)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> inccov.partial_fit(X[:1])
    >>> inccov.partial_fit(X[1:])
    >>> inccov.covariance_
    np.array([[1., 1.],[1., 1.]])
    >>> inccov.location_
    np.array([2., 3.])
    >>> inccov.fit(X)
    >>> inccov.covariance_
    np.array([[1., 1.],[1., 1.]])
    >>> inccov.location_
    np.array([2., 3.])
    """

    __doc__ = _add_inc_serialization_note(__doc__)

    _onedal_incremental_covariance = staticmethod(onedal_IncrementalEmpiricalCovariance)

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            "store_precision": ["boolean"],
            "assume_centered": ["boolean"],
            "batch_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
            "copy": ["boolean"],
        }

    def __init__(
        self, *, store_precision=False, assume_centered=False, batch_size=None, copy=True
    ):
        self.assume_centered = assume_centered
        self.store_precision = store_precision
        self.batch_size = batch_size
        self.copy = copy

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.covariance.{self.__class__.__name__}.{method_name}"
        )
        return patching_status

    def _onedal_finalize_fit(self):
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()
        self._need_to_finalize = False

        if not daal_check_version((2024, "P", 400)) and self.assume_centered:
            xp, _ = get_namespace(self._onedal_estimator.location_)
            location = self._onedal_estimator.location_[None, :]
            self._onedal_estimator.covariance_ += xp.dot(location.T, location)
            self._onedal_estimator.location_ = xp.zeros_like(xp.squeeze(location))
        if self.store_precision:
            self.precision_ = _pinvh(
                self._onedal_estimator.covariance_, check_finite=False
            )
        else:
            self.precision_ = None

    @property
    def covariance_(self):
        if hasattr(self, "_onedal_estimator"):
            if self._need_to_finalize:
                self._onedal_finalize_fit()
            return self._onedal_estimator.covariance_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'covariance_'"
            )

    @property
    def location_(self):
        if hasattr(self, "_onedal_estimator"):
            if self._need_to_finalize:
                self._onedal_finalize_fit()
            return self._onedal_estimator.location_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'location_'"
            )

    def _onedal_partial_fit(self, X, queue=None, check_input=True):
        first_pass = not hasattr(self, "n_samples_seen_") or self.n_samples_seen_ == 0

        if check_input and not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                reset=first_pass,
                copy=self.copy,
            )

        onedal_params = {
            "method": "dense",
            "bias": True,
            "assume_centered": self.assume_centered,
        }
        if not hasattr(self, "_onedal_estimator"):
            self._onedal_estimator = self._onedal_incremental_covariance(**onedal_params)
        try:
            if first_pass:
                self.n_samples_seen_ = X.shape[0]
                self.n_features_in_ = X.shape[1]
            else:
                self.n_samples_seen_ += X.shape[0]

            self._onedal_estimator.partial_fit(X, queue=queue)
        finally:
            self._need_to_finalize = True

        return self

    def get_precision(self):
        if self.store_precision:
            precision = self.precision_
        else:
            precision = _pinvh(self.covariance_, check_finite=False)
        return precision

    def partial_fit(self, X, y=None, check_input=True):
        """
        Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        check_input : bool, default=True
            Run check_array on X.

        Returns
        -------
        self : IncrementalEmpiricalCovariance
            Returns the instance itself.
        """
        if sklearn_check_version("1.2") and check_input:
            self._validate_params()
        return dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": None,
            },
            X,
            check_input=check_input,
        )

    def fit(self, X, y=None):
        """
        Fit the model with X, using minibatches of size ``batch_size``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : IncrementalEmpiricalCovariance
            Returns the instance itself.
        """
        if sklearn_check_version("1.2"):
            self._validate_params()
        return dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": None,
            },
            X,
        )

    def _onedal_fit(self, X, queue=None):
        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        if not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(self, X, dtype=[xp.float64, xp.float32], copy=self.copy)

        self.batch_size_ = self.batch_size if self.batch_size else 5 * self.n_features_in_

        if X.shape[0] == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        for batch in gen_batches(X.shape[0], self.batch_size_):
            X_batch = X[batch, ...]
            self._onedal_partial_fit(X_batch, queue=queue, check_input=False)

        self._onedal_finalize_fit()

        return self

    @wrap_output_data
    @support_sycl_format
    def score(self, X_test, y=None):

        check_is_fitted(self)
        # Only covariance evaluated for get_namespace due to dpnp/dpctl
        # support without array_api_dispatch
        xp, _ = get_namespace(X_test, self.covariance_)

        X = validate_data(
            self,
            X_test,
            dtype=[xp.float64, xp.float32],
            reset=False,
        )

        location = self.location_
        precision = self.get_precision()

        est = clone(self)
        est.set_params(assume_centered=True)

        # test_cov is a numpy array, but calculated on device
        test_cov = est.fit(X - self.location_).covariance_
        if not _is_numpy_namespace(xp):
            test_cov = xp.asarray(test_cov, device=X_test.device)
        res = log_likelihood(test_cov, self.get_precision())

        return res

    @wrap_output_data
    @support_sycl_format
    def error_norm(self, comp_cov, norm="frobenius", scaling=True, squared=True):
        # equivalent to the sklearn implementation but written for array API
        # in the case of numpy-like inputs it will use sklearn's version instead.
        # This can be deprecated if/when sklearn makes the equivalent array API enabled.
        # This includes a validate_data call and an unusual call to get_namespace in
        # order to also support dpnp/dpctl without array_api_dispatch.
        check_is_fitted(self)
        # Only covariance evaluated for get_namespace due to dpnp/dpctl
        # support without array_api_dispatch
        xp, _ = get_namespace(self.covariance_)
        c_cov = validate_data(
            self,
            comp_cov,
            dtype=[xp.float64, xp.float32],
            reset=False,
        )

        if _is_numpy_namespace(xp):
            # must be done this way is it does not inherit from sklearn
            return _sklearn_EmpiricalCovariance.error_norm(
                self, c_cov, norm=norm, scaling=scaling, squared=squared
            )

        # compute the error
        error = c_cov - self.covariance_
        # compute the error norm
        if norm == "frobenius":
            # variance from sklearn version to leverage BLAS GEMM
            # squared_norm = xp.sum(error**2)
            squared_norm = xp.matmul(xp.reshape(error, (-1)), xp.reshape(error, (-1)))
        elif norm == "spectral":
            squared_norm = xp.max(xp.linalg.svdvals(xp.matmul(error.T, error)))
        else:
            raise NotImplementedError("Only spectral and frobenius norms are implemented")
        # optionally scale the error norm
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        # finally get either the squared norm or the norm
        if squared:
            result = squared_norm
        else:
            result = xp.sqrt(squared_norm)

        return result

    # expose sklearnex pairwise_distances if mahalanobis distance eventually supported
    @support_sycl_format
    def mahalanobis(self, X):
        # This must be done as ```support_input_format``` is insufficient for array API
        # support when attributes are non-numpy.
        check_is_fitted(self)
        precision = self.get_precision()
        loc = self.location_[None, :]
        xp, _ = get_namespace(X, precision, loc)
        # do not check dtype, done in pairwise_distances
        X_in = validate_data(self, X, reset=False)

        if not _is_numpy_namespace(xp) and isinstance(X_in, np.ndarray):
            # corrects issues with respect to dpnp/dpctl support without array_api_dispatch
            X_in = X
            loc = xp.asarray(loc, device=X.device)
            precision = xp.asarray(precision, device=X.device)

        with config_context(assume_finite=True):
            try:
                dist = _mahalanobis(X_in, loc, VI=precision)

            except ValueError as e:
                # Throw the expected sklearn error in an n_feature length violation
                if "Incompatible dimension for X and Y matrices: X.shape[1] ==" in str(e):
                    raise ValueError(
                        f"X has {_num_features(X)} features, but {self.__class__.__name__} "
                        f"is expecting {self.n_features_in_} features as input."
                    ) from None
                else:
                    raise e

        if not _is_numpy_namespace(xp):
            dist = xp.asarray(dist, device=X.device)

        return (xp.reshape(dist, (-1,))) ** 2

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    mahalanobis.__doc__ = _sklearn_EmpiricalCovariance.mahalanobis.__doc__
    error_norm.__doc__ = _sklearn_EmpiricalCovariance.error_norm.__doc__
    score.__doc__ = _sklearn_EmpiricalCovariance.score.__doc__
    get_precision.__doc__ = _sklearn_EmpiricalCovariance.get_precision.__doc__
