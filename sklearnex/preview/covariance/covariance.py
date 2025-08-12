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
from functools import partial

import scipy.sparse as sp
from sklearn.base import clone
from sklearn.covariance import EmpiricalCovariance as _sklearn_EmpiricalCovariance
from sklearn.utils.validation import check_array, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from daal4py.sklearn.metrics import pairwise_distances
from onedal._device_offload import support_input_format, support_sycl_format
from onedal.covariance import EmpiricalCovariance as onedal_EmpiricalCovariance
from onedal.utils._array_api import _is_numpy_namespace
from sklearnex import config_context

from ..._config import get_config
from ..._device_offload import dispatch, wrap_output_data
from ..._utils import PatchingConditionsChain, register_hyperparameters
from ...base import oneDALEstimator
from ...utils._array_api import _pinvh, enable_array_api, get_namespace, log_likelihood
from ...utils.validation import assert_all_finite, validate_data

# This is a temporary workaround for issues with sklearnex._device_offload._get_host_inputs
# passing kwargs with sycl queues with other host data will cause failures
_mahalanobis = support_input_format(partial(pairwise_distances, metric="mahalanobis"))


@enable_array_api
@register_hyperparameters({"fit": ("covariance", "compute")})
@control_n_jobs(decorated_methods=["fit", "mahalanobis"])
class EmpiricalCovariance(oneDALEstimator, _sklearn_EmpiricalCovariance):
    __doc__ = _sklearn_EmpiricalCovariance.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_EmpiricalCovariance._parameter_constraints,
        }

    def _set_covariance(self, covariance):
        if not get_config()["use_raw_input"]:
            if sklearn_check_version("1.6"):
                covariance = check_array(covariance, ensure_all_finite=False)
            else:
                covariance = check_array(covariance, force_all_finite=False)
            assert_all_finite(covariance)
        # set covariance
        self.covariance_ = covariance
        # set precision
        if self.store_precision:
            self.precision_ = _pinvh(covariance, check_finite=False)
        else:
            self.precision_ = None

    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")
        if not daal_check_version((2024, "P", 400)) and self.assume_centered:
            xp, _ = get_namespace(self._onedal_estimator.location_)
            location = self._onedal_estimator.location_[None, :]
            self._onedal_estimator.covariance_ += xp.dot(location.T, location)
            self._onedal_estimator.location_ = xp.zeros_like(
                self._onedal_estimator.location_
            )
        self._set_covariance(self._onedal_estimator.covariance_)
        self.location_ = self._onedal_estimator.location_

    _onedal_covariance = staticmethod(onedal_EmpiricalCovariance)

    def _onedal_fit(self, X, queue=None):
        if not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(self, X, dtype=[xp.float64, xp.float32])

        if X.shape[0] == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        self._onedal_estimator = self._onedal_covariance(
            method="dense", bias=True, assume_centered=self.assume_centered
        )
        self._onedal_estimator.fit(X, queue=queue)
        self._save_attributes()

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.covariance.{class_name}.{method_name}"
        )
        if method_name in ["fit", "mahalanobis"]:
            (X,) = data
            patching_status.and_conditions(
                [
                    (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                ]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def get_precision(self):
        # use array API-enabled version
        if self.store_precision:
            precision = self.precision_
        else:
            precision = _pinvh(self.covariance_, check_finite=False)
        return precision

    def fit(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_EmpiricalCovariance.fit,
            },
            X,
        )

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
            test_cov = xp.asarray(test_cov, device=X.device)
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
        xp, _ = get_namespace(comp_cov, self.covariance_)
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
            squared_norm = xp.matmul(
                xp.reshape(error, (1, -1)), xp.reshape(error, (-1, 1))
            )
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

    fit.__doc__ = _sklearn_EmpiricalCovariance.fit.__doc__
    mahalanobis.__doc__ = _sklearn_EmpiricalCovariance.mahalanobis.__doc__
    error_norm.__doc__ = _sklearn_EmpiricalCovariance.error_norm.__doc__
    score.__doc__ = _sklearn_EmpiricalCovariance.score.__doc__
    get_precision.__doc__ = _sklearn_EmpiricalCovariance.get_precision.__doc__
    _set_covariance.__doc__ = _sklearn_EmpiricalCovariance._set_covariance.__doc__
