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

import numpy as np
from scipy import sparse as sp
from sklearn.covariance import EmpiricalCovariance as _sklearn_EmpiricalCovariance
from sklearn.utils import check_array

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.common.hyperparameters import get_hyperparameters
from onedal.covariance import EmpiricalCovariance as onedal_EmpiricalCovariance
from sklearnex import config_context
from sklearnex.metrics import pairwise_distances

from ..._device_offload import dispatch, wrap_output_data
from ..._utils import PatchingConditionsChain, register_hyperparameters
from ...base import oneDALEstimator
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
        if not daal_check_version((2024, "P", 400)) and self.assume_centered:
            location = self._onedal_estimator.location_[None, :]
            self._onedal_estimator.covariance_ += np.dot(location.T, location)
            self._onedal_estimator.location_ = np.zeros_like(np.squeeze(location))
        self._set_covariance(self._onedal_estimator.covariance_)
        self.location_ = self._onedal_estimator.location_

    _onedal_covariance = staticmethod(onedal_EmpiricalCovariance)

    def _onedal_fit(self, X, queue=None):
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

    def fit(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        X = validate_data(self, X, ensure_all_finite=False)

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

    # expose sklearnex pairwise_distances if mahalanobis distance eventually supported
    @wrap_output_data
    def mahalanobis(self, X):
        X = validate_data(self, X, reset=False)

        precision = self.get_precision()
        with config_context(assume_finite=True):
            # compute mahalanobis distances
            dist = pairwise_distances(
                X, self.location_[np.newaxis, :], metric="mahalanobis", VI=precision
            )

        return np.reshape(dist, (len(X),)) ** 2

    error_norm = wrap_output_data(_sklearn_EmpiricalCovariance.error_norm)
    score = wrap_output_data(_sklearn_EmpiricalCovariance.score)

    fit.__doc__ = _sklearn_EmpiricalCovariance.fit.__doc__
    mahalanobis.__doc__ = _sklearn_EmpiricalCovariance.mahalanobis
    error_norm.__doc__ = _sklearn_EmpiricalCovariance.error_norm.__doc__
    score.__doc__ = _sklearn_EmpiricalCovariance.score.__doc__
