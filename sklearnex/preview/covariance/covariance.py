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

import scipy.sparse as sp
from sklearn.covariance import EmpiricalCovariance as _sklearn_EmpiricalCovariance

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal._device_offload import support_input_format
from onedal.common.hyperparameters import get_hyperparameters
from onedal.covariance import EmpiricalCovariance as onedal_EmpiricalCovariance
from onedal.utils._array_api import _is_numpy_namespace
from sklearnex import config_context
from sklearnex.metrics import pairwise_distances

from ..._device_offload import dispatch
from ..._utils import PatchingConditionsChain, register_hyperparameters
from ...base import oneDALEstimator
from ...utils._array_api import get_namespace, pinvh
from ...utils.validation import validate_data


@register_hyperparameters({"fit": get_hyperparameters("covariance", "compute")})
@control_n_jobs(decorated_methods=["fit", "mahalanobis"])
class EmpiricalCovariance(oneDALEstimator, _sklearn_EmpiricalCovariance):
    __doc__ = _sklearn_EmpiricalCovariance.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_EmpiricalCovariance._parameter_constraints,
        }

    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")
        lp, _ = get_namespace(self._onedal_estimator.location_)
        if not daal_check_version((2024, "P", 400)) and self.assume_centered:
            location = self._onedal_estimator.location_[None, :]
            self._onedal_estimator.covariance_ += lp.dot(location.T, location)
            self._onedal_estimator.location_ = lp.zeros_like(lp.squeeze(location))
        self._set_covariance(self._onedal_estimator.covariance_)
        self.location_ = lp.squeeze(self._onedal_estimator.location_)

    _onedal_covariance = staticmethod(onedal_EmpiricalCovariance)

    def _onedal_fit(self, X, queue=None):
        xp, _ = get_namespace(X)
        if sklearn_check_version("1.2"):
            self._validate_params()

        X = validate_data(self, X, dtype=[xp.float64, xp.float32])

        if X.shape[0] == 1:
            warnings.warn(
                "Only one sample available. You may want to reshape your data array"
            )

        onedal_params = {
            "method": "dense",
            "bias": True,
            "assume_centered": self.assume_centered,
        }

        self._onedal_estimator = self._onedal_covariance(**onedal_params)
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

    def get_precision():
        # use array API-enabled version
        if self.store_precision:
            precision = self.precision_
        else:
            precision = pinvh(self.covariance_, check_finite=False)
        return precision

    def fit(self, X, y=None):
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
    def score(self, X_test, y=None):
        xp, _ = get_namespace(X_test)

        check_is_fitted(self)

        X = validate_data(
            self,
            X_test,
            dtype=[xp.float64, xp.float32],
            reset=False,
        )

        location = self.location_
        precision = self.get_precision()

        if not _is_numpy_namespace(xp):
            # depending on the sklearn version, check_array
            # and validate_data will return only numpy arrays
            # which will break dpnp/dpctl support. If the
            # array namespace isn't from numpy and the data
            # is now a numpy array, it has been validated and
            # the original can be used.
            if isinstance(X, np.ndarray):
                X = X_test
            location = xp.asarray(location, device=X.device)
            precision = xp.asarray(precision, device=X.device)

        est = clone(self)
        est.set_params(**{"assume_centered": True})

        # test_cov is a numpy array, but calculated on device
        test_cov = est.fit(X - location).covariance_
        if not _is_numpy_namespace(xp):
            test_cov = xp.asarray(test_cov, device=X.device)
        res = log_likelihood(test_cov, precision)

        return res

    def error_norm(self, comp_cov, norm="frobenius", scaling=True, squared=True):
        # simple branched version for array API support
        xp, _ = get_namespace(comp_cov)
        if _is_numpy_namespace(xp):
            return super().error_norm(
                comp_cov, norm=norm, scaling=scaling, squared=squared
            )

        # translated copy of sklearn's error_norm.  Can be deprecated when
        # implemented in sklearn
        # compute the error
        error = comp_cov - self.covariance_
        # compute the error norm
        if norm == "frobenius":
            squared_norm = xp.sum(error**2)
        elif norm == "spectral":
            squared_norm = xp.max(xp.linalg.svdvals(xp.dot(error.T, error)))
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
    def mahalanobis(self, X):
        xp, _ = get_namespace(X)
        X = validate_data(self, X, reset=False, dtype=[xp.float64, xp.float32])

        precision = self.get_precision()
        # compute mahalanobis distances
        # pairwise_distances will check n_features (via n_feature matching with
        # self.location_) , and will check for finiteness via check array
        # check_feature_names will match _validate_data functionally
        location = self.location_[None, :]

        if not _is_numpy_namespace(xp):
            # Guarantee that inputs to pairwise_distances match in type and location
            location = xp.asarray(location, device=X.device)
            precision = xp.asarray(precision, device=X.device)

        with config_context(assume_finite=True):

            try:
                dist = pairwise_distances(X, location, metric="mahalanobis", VI=precision)

            except ValueError as e:
                # Throw the expected sklearn error in an n_feature length violation
                if "Incompatible dimension for X and Y matrices: X.shape[1] ==" in str(e):
                    raise ValueError(
                        f"X has {_num_features(X)} features, but {self.__class__.__name__} "
                        f"is expecting {self.n_features_in_} features as input."
                    )
                else:
                    raise e

        if not _is_numpy_namespace(xp):
            dist = xp.asarray(dist, device=X.device)

        return (xp.reshape(dist, (-1,))) ** 2

    fit.__doc__ = _sklearn_EmpiricalCovariance.fit.__doc__
    mahalanobis.__doc__ = _sklearn_EmpiricalCovariance.mahalanobis
    error_norm.__doc__ = _sklearn_EmpiricalCovariance.error_norm.__doc__
    score.__doc__ = _sklearn_EmpiricalCovariance.score.__doc__
    get_precision.__doc__ = _sklearn_EmpiricalCovariance.get_precision.__doc__
