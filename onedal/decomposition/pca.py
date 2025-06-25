# ==============================================================================
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
# ==============================================================================

import numbers
from abc import ABCMeta

import numpy as np
from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils.extmath import stable_cumsum

from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend

from .._config import _get_config
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace


class BasePCA(metaclass=ABCMeta):

    def __init__(
        self,
        n_components=None,
        is_deterministic=True,
        method="cov",
        whiten=False,
    ):
        self.n_components = n_components
        self.method = method
        self.is_deterministic = is_deterministic
        self.whiten = whiten

    # provides direct access to the backend model constructor
    @bind_default_backend("decomposition.dim_reduction")
    def model(self): ...

    @bind_default_backend("decomposition.dim_reduction")
    def train(self, params, X): ...

    @bind_default_backend("decomposition.dim_reduction")
    def infer(self, params, X, model): ...

    def _get_onedal_params(self, data, stage=None):
        if stage is None:
            n_components = self._resolve_n_components_for_training(data.shape)
        elif stage == "predict":
            n_components = self.n_components_
        return {
            "fptype": data.dtype,
            "method": self.method,
            "n_components": n_components,
            "is_deterministic": self.is_deterministic,
            "whiten": self.whiten,
        }

    def _validate_n_components(self, n_components, n_samples, n_features):
        if n_components is None:
            n_components = min(n_samples, n_features)
        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(n_samples, n_features))
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

    def _resolve_n_components_for_training(self, shape_tuple):
        if self.n_components is None or self.n_components == "mle":
            return min(shape_tuple)
        elif (
            isinstance(self.n_components, float)
            and self.n_components > 0.0
            and self.n_components <= 1.0
        ):
            return min(shape_tuple)
        else:
            return self.n_components

    def _resolve_n_components_for_result(self, shape_tuple):
        if self.n_components is None:
            return min(shape_tuple)
        elif self.n_components == "mle":
            return _infer_dimension(self.explained_variance_, shape_tuple[0])
        elif 0.0 < self.n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            return np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
        elif isinstance(self.n_components, float) and self.n_components == 1.0:
            return min(shape_tuple)
        else:
            return self.n_components

    def _compute_noise_variance(self, n_components, n_sf_min):
        if n_components < n_sf_min:
            if len(self.explained_variance_) == n_sf_min:
                return self.explained_variance_[n_components:].mean()
            elif len(self.explained_variance_) < n_sf_min:
                # TODO Rename variances_ to var_ to align with sklearn/sklearnex IncrementalPCA
                if hasattr(self, "variances_"):
                    resid_var = self.variances_.sum()
                elif hasattr(self, "var_"):
                    resid_var = self.var_.sum()

                resid_var -= self.explained_variance_.sum()
                return resid_var / (n_sf_min - n_components)
        else:
            return 0.0

    def _create_model(self):
        m = self.model()
        m.eigenvectors = to_table(self.components_)
        m.means = to_table(self.mean_)
        if self.whiten:
            m.eigenvalues = to_table(self.explained_variance_)
        self._onedal_model = m
        return m

    @supports_queue
    def predict(self, X, queue=None):
        model = self._create_model()
        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(X_table, stage="predict")
        result = self.infer(params, model, X_table)
        return from_table(result.transformed_data)

    transform = predict


class PCA(BasePCA):

    @supports_queue
    def fit(self, X, y=None, queue=None):
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, xp, _ = _get_sycl_namespace(X)
        if use_raw_input and sua_iface:
            queue = X.sycl_queue

        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)
        self._validate_n_components(self.n_components, n_samples, n_features)

        # TODO: investigate why np.ndarray with OWNDATA=FALSE flag
        # fails to be converted to oneDAL table
        if isinstance(X, np.ndarray) and not X.flags["OWNDATA"]:
            X = X.copy()

        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X)
        hparams = get_hyperparameters("pca", "train")
        if hparams is not None and not hparams.is_default:
            result = self.train(params, hparams.backend, X)
        else:
            result = self.train(params, X)

        self.mean_ = from_table(result.means).ravel()
        self.variances_ = from_table(result.variances)
        self.components_ = from_table(result.eigenvectors)
        self.singular_values_ = from_table(result.singular_values).ravel()
        self.explained_variance_ = np.maximum(from_table(result.eigenvalues).ravel(), 0)
        self.explained_variance_ratio_ = from_table(
            result.explained_variances_ratio
        ).ravel()
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        n_components = self._resolve_n_components_for_result(X.shape)
        self.n_components_ = n_components
        self.noise_variance_ = self._compute_noise_variance(n_components, n_sf_min)

        if n_components < params["n_components"]:
            self.explained_variance_ = self.explained_variance_[:n_components]
            self.components_ = self.components_[:n_components]
            self.singular_values_ = self.singular_values_[:n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]

        return self
