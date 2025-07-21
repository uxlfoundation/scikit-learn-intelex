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

from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend

from .._config import _get_config
from ..common.hyperparameters import get_hyperparameters
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace


class PCA(metaclass=ABCMeta):

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

    def _get_onedal_params(self, data):
        return {
            "fptype": data.dtype,
            "method": self.method,
            "n_components": self.n_components_,
            "is_deterministic": self.is_deterministic,
            "whiten": self.whiten,
        }

    def _create_model(self, queue=None):
        m = self.model()
        m.eigenvectors = to_table(self.components_, queue=queue)
        m.means = to_table(self.mean_, queue=queue)
        if self.whiten:
            m.eigenvalues = to_table(self.explained_variance_, queue=queue)
        self._onedal_model = m

    @supports_queue
    def predict(self, X, queue=None):
        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(X_table)
        if not hasattr(self, "_onedal_model"):
            self._create_model(queue)
        result = self.infer(params, self._onedal_model, X_table)
        return from_table(result.transformed_data, like=X)

    @supports_queue
    def fit(self, X, y=None, queue=None):

        X_table = to_table(X, queue=queue)
        # define n_components_ to allow for external modification of
        # transforms
        self.n_components_ = self.n_components

        params = self._get_onedal_params(X_table)
        hparams = get_hyperparameters("pca", "train")
        if hparams is not None and not hparams.is_default:
            result = self.train(params, hparams.backend, X_table)
        else:
            result = self.train(params, X_table)

        (
            mean_,
            var_,
            self.components_,
            sing_vals_,
            eigenvalues_,
            var_ratio,
        ) = from_table(
            result.means,
            result.variances,
            result.eigenvectors,
            result.singular_values,
            result.eigenvalues,
            result.explained_variances_ratio,
            like=X,
        )

        # tables are 2d, but outputs are inherently 1d, reduce dimensions
        self.mean_ = mean_[0, ...]
        self.var_ = var_[0, ...]
        self.singular_values_ = sing_vals_[0, ...]
        self.explained_variance_ = eigenvalues_[0, ...]
        self.explained_variance_ratio_ = var_ratio[0, ...]

        return self
