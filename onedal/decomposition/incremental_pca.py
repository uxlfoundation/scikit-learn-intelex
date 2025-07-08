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

import numpy as np

from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM

from .._config import _get_config
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import _check_array
from .pca import BasePCA


class IncrementalPCA(BasePCA):
    """Incremental oneDAL PCA estimator.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If ``n_components`` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    is_deterministic : bool, default=True
        When True the ``components_`` vectors are chosen in deterministic
        way, otherwise some of them can be oppositely directed.

    method : str, default='cov'
        Method used in oneDAL computation.

    whiten : bool, default=False
        When True the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    Attributes
    ----------
        components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. Equivalently, the right singular
        vectors of the centered input data, parallel to its eigenvectors.
        The components are sorted by decreasing ``explained_variance_``.

        explained_variance_ : ndarray of shape (n_components,)
            Variance explained by each of the selected components.

        explained_variance_ratio_ : ndarray of shape (n_components,)
            Percentage of variance explained by each of the selected components.
            If all components are stored, the sum of explained variances is equal
            to 1.0.

        singular_values_ : ndarray of shape (n_components,)
            The singular values corresponding to each of the selected components.
            The singular values are equal to the 2-norms of the ``n_components``
            variables in the lower-dimensional space.

        mean_ : ndarray of shape (n_features,)
            Per-feature empirical mean, aggregate over calls to ``partial_fit``.

        var_ : ndarray of shape (n_features,)
            Per-feature empirical variance, aggregate over calls to
            ``partial_fit``.

        noise_variance_ : float
            Equal to the average of (min(n_features, n_samples) - n_components)
            smallest eigenvalues of the covariance matrix of X.
    """

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
        self._queue = None
        self._reset()

    @bind_default_backend("decomposition.dim_reduction")
    def finalize_train(self, params, partial_result): ...

    @bind_default_backend("decomposition.dim_reduction")
    def partial_train(self, params, partial_result, X_table): ...

    @bind_default_backend("decomposition.dim_reduction")
    def partial_train_result(self): ...

    def _reset(self):
        self._need_to_finalize = False
        self._queue = None
        self._partial_result = self.partial_train_result()
        if hasattr(self, "components_"):
            del self.components_

    def __getstate__(self):
        # Since finalize_fit can't be dispatched without directly provided queue
        # and the dispatching policy can't be serialized, the computation is finalized
        # here and the policy is not saved in serialized data.

        self.finalize_fit()
        data = self.__dict__.copy()
        data.pop("_queue", None)
        return data

    @supports_queue
    def partial_fit(self, X, queue=None):
        """Generate partial PCA from batch data in `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        queue : SyclQueue or None, default=None
            SYCL Queue object for device code execution. Default
            value None causes computation on host.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, _, _ = _get_sycl_namespace(X)

        # All data should use the same sycl queue
        if use_raw_input and sua_iface:
            queue = X.sycl_queue
        if not use_raw_input:
            X = _check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)

        n_samples, n_features = X.shape
        first_pass = not hasattr(self, "components_")
        if first_pass:
            self.components_ = None
            self.n_samples_seen_ = n_samples
            self.n_features_in_ = n_features
        else:
            self.n_samples_seen_ += n_samples

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        else:
            self.n_components_ = self.n_components

        self._queue = queue

        X_table = to_table(X, queue=queue)

        if not hasattr(self, "_dtype"):
            self._dtype = X_table.dtype
            self._params = self._get_onedal_params(X_table)

        self._partial_result = self.partial_train(
            self._params, self._partial_result, X_table
        )
        self._need_to_finalize = True
        self._queue = queue
        return self

    def finalize_fit(self):
        """Finalize statistics from the current `_partial_result`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self._need_to_finalize:
            with QM.manage_global_queue(self._queue):
                result = self.finalize_train(self._params, self._partial_result)
            self.mean_ = from_table(result.means).ravel()
            self.var_ = from_table(result.variances).ravel()
            self.components_ = from_table(result.eigenvectors)
            self.singular_values_ = np.nan_to_num(
                from_table(result.singular_values).ravel()
            )
            self.explained_variance_ = np.maximum(
                from_table(result.eigenvalues).ravel(), 0
            )
            self.explained_variance_ratio_ = from_table(
                result.explained_variances_ratio
            ).ravel()
            self.noise_variance_ = self._compute_noise_variance(
                self.n_components_, min(self.n_samples_seen_, self.n_features_in_)
            )
            self._need_to_finalize = False
            self._queue = None

        return self
