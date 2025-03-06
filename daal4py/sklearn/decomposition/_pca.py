# ==============================================================================
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
# ==============================================================================

import numbers
from math import sqrt

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted

import daal4py

from .._n_jobs_support import control_n_jobs
from .._utils import PatchingConditionsChain, getFPType, sklearn_check_version
from ..utils.validation import check_feature_names, validate_data

if sklearn_check_version("1.4"):
    from sklearn.utils._array_api import get_namespace

if sklearn_check_version("1.3"):
    from sklearn.base import _fit_context

if sklearn_check_version("1.1"):
    from sklearn.utils import check_scalar

from sklearn.decomposition._pca import PCA as PCA_original
from sklearn.decomposition._pca import _infer_dimension


@control_n_jobs(decorated_methods=["fit", "transform"])
class PCA(PCA_original):
    __doc__ = PCA_original.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**PCA_original._parameter_constraints}

    if sklearn_check_version("1.1"):

        def __init__(
            self,
            n_components=None,
            copy=True,
            whiten=False,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=None,
        ):
            self.n_components = n_components
            self.copy = copy
            self.whiten = whiten
            self.svd_solver = svd_solver
            self.tol = tol
            self.iterated_power = iterated_power
            self.n_oversamples = n_oversamples
            self.power_iteration_normalizer = power_iteration_normalizer
            self.random_state = random_state

    else:

        def __init__(
            self,
            n_components=None,
            copy=True,
            whiten=False,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            random_state=None,
        ):
            self.n_components = n_components
            self.copy = copy
            self.whiten = whiten
            self.svd_solver = svd_solver
            self.tol = tol
            self.iterated_power = iterated_power
            self.random_state = random_state

    def _validate_n_components(self, n_components, n_samples, n_features):
        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported " "if n_samples >= n_features"
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

    def _fit_full_daal4py(self, X, n_components):
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        if n_components == "mle":
            daal_n_components = n_features
        elif n_components < 1:
            daal_n_components = n_sf_min
        else:
            daal_n_components = n_components

        fpType = getFPType(X)

        covariance_algo = daal4py.covariance(
            fptype=fpType, outputMatrixType="covarianceMatrix"
        )
        covariance_res = covariance_algo.compute(X)

        self.mean_ = covariance_res.mean.ravel()
        covariance = covariance_res.covariance
        variances_ = np.array([covariance[i, i] for i in range(n_features)])

        pca_alg = daal4py.pca(
            fptype=fpType,
            method="correlationDense",
            resultsToCompute="eigenvalue",
            isDeterministic=True,
            nComponents=daal_n_components,
        )
        pca_res = pca_alg.compute(X, covariance)

        components_ = pca_res.eigenvectors
        explained_variance_ = np.maximum(pca_res.eigenvalues.ravel(), 0)
        tot_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / tot_var

        if n_components == "mle":
            n_components = _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1

        if n_components < n_sf_min:
            if explained_variance_.shape[0] == n_sf_min:
                self.noise_variance_ = explained_variance_[n_components:].mean()
            else:
                resid_var_ = variances_.sum()
                resid_var_ -= explained_variance_[:n_components].sum()
                self.noise_variance_ = resid_var_ / (n_sf_min - n_components)
        else:
            self.noise_variance_ = 0.0

        if sklearn_check_version("1.2"):
            self.n_samples_, self.n_features_in_ = n_samples, n_features
        else:
            self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = np.sqrt((n_samples - 1) * self.explained_variance_)

    def _fit_full(self, X, n_components):
        n_samples, n_features = X.shape
        self._validate_n_components(n_components, n_samples, n_features)

        self._fit_full_daal4py(X, min(X.shape))

        U = None
        V = self.components_
        S = self.singular_values_

        if n_components == "mle":
            n_components = _infer_dimension(self.explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1

        if n_components < min(n_features, n_samples):
            self.noise_variance_ = self.explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        if sklearn_check_version("1.2"):
            self.n_samples_, self.n_features_in_ = n_samples, n_features
        else:
            self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = self.components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
        self.singular_values_ = self.singular_values_[:n_components]

        return U, S, V

    def _fit(self, X):
        if sklearn_check_version("1.4"):
            xp, is_array_api_compliant = get_namespace(X)

            if issparse(X) and self.svd_solver != "arpack":
                raise TypeError(
                    'PCA only support sparse inputs with the "arpack" solver, while '
                    f'"{self.svd_solver}" was passed. See TruncatedSVD for a possible'
                    " alternative."
                )
            # Raise an error for non-Numpy input and arpack solver.
            if self.svd_solver == "arpack" and is_array_api_compliant:
                raise ValueError(
                    "PCA with svd_solver='arpack' is not supported for Array API inputs."
                )

            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse=("csr", "csc"),
                ensure_2d=True,
                copy=self.copy,
            )

        else:
            if issparse(X):
                raise TypeError(
                    "PCA does not support sparse input. See "
                    "TruncatedSVD for a possible alternative."
                )
            X = validate_data(
                self, X, dtype=[np.float64, np.float32], ensure_2d=True, copy=False
            )

        if self.n_components is None:
            if self.svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        self._fit_svd_solver = self.svd_solver
        shape_good_for_daal = X.shape[1] / X.shape[0] < 2

        if self._fit_svd_solver == "auto":
            if sklearn_check_version("1.1"):
                # Small problem or n_components == 'mle', just call full PCA
                if max(X.shape) <= 500 or n_components == "mle":
                    self._fit_svd_solver = "full"
                elif 1 <= n_components < 0.8 * min(X.shape):
                    self._fit_svd_solver = "randomized"
                # This is also the case of n_components in (0,1)
                else:
                    self._fit_svd_solver = "full"
            else:
                if n_components == "mle":
                    self._fit_svd_solver = "full"
                else:
                    n, p, k = X.shape[0], X.shape[1], n_components
                    # These coefficients are result of training of Logistic Regression
                    # (max_iter=10000, solver="liblinear", fit_intercept=False)
                    # on different datasets and number of components.
                    # X is a dataset with npk, np^2, and n^2 columns.
                    # And y is speedup of patched scikit-learn's
                    # full PCA against stock scikit-learn's randomized PCA.
                    regression_coefs = np.array(
                        [
                            [9.779873e-11, n * p * k],
                            [-1.122062e-11, n * p * p],
                            [1.127905e-09, n**2],
                        ]
                    )

                    if (
                        n_components >= 1
                        and np.dot(regression_coefs[:, 0], regression_coefs[:, 1]) <= 0
                    ):
                        self._fit_svd_solver = "randomized"
                    else:
                        self._fit_svd_solver = "full"

        if not shape_good_for_daal or self._fit_svd_solver != "full":
            if sklearn_check_version("1.4"):
                X = validate_data(self, X, copy=self.copy, accept_sparse=("csr", "csc"))
            else:
                X = validate_data(self, X, copy=self.copy)

        _patching_status = PatchingConditionsChain("sklearn.decomposition.PCA.fit")
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    self._fit_svd_solver == "full",
                    f"'{self._fit_svd_solver}' SVD solver is not supported. "
                    "Only 'full' solver is supported.",
                )
            ]
        )

        if _dal_ready:
            _dal_ready = _patching_status.and_conditions(
                [
                    (
                        shape_good_for_daal,
                        "The shape of X does not satisfy oneDAL requirements: "
                        "number of features / number of samples >= 2",
                    ),
                ]
            )
            if _dal_ready:
                result = self._fit_full(X, n_components)
            else:
                result = PCA_original._fit_full(self, X, n_components)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            result = self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'" "".format(self._fit_svd_solver)
            )

        _patching_status.write_log()
        return result

    def _transform_daal4py(self, X, whiten=False, scale_eigenvalues=True, check_X=True):
        check_is_fitted(self)

        check_feature_names(self, X, reset=False)
        if sklearn_check_version("1.6"):
            X = check_array(X, dtype=[np.float64, np.float32], ensure_all_finite=check_X)
        else:
            X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=check_X)
        fpType = getFPType(X)

        tr_data = dict()
        if self.mean_ is not None:
            tr_data["mean"] = self.mean_.reshape((1, -1))
        if whiten:
            if scale_eigenvalues:
                tr_data["eigenvalue"] = (
                    self.n_samples_ - 1
                ) * self.explained_variance_.reshape((1, -1))
            else:
                tr_data["eigenvalue"] = self.explained_variance_.reshape((1, -1))
        elif scale_eigenvalues:
            tr_data["eigenvalue"] = np.full(
                (1, self.explained_variance_.shape[0]),
                self.n_samples_ - 1.0,
                dtype=X.dtype,
            )

        if sklearn_check_version("1.2"):
            expected_n_features = self.n_features_in_
        else:
            expected_n_features = self.n_features_
        if X.shape[1] != expected_n_features:
            raise ValueError(
                (
                    f"X has {X.shape[1]} features, "
                    f"but PCA is expecting {expected_n_features} features as input"
                )
            )

        tr_res = daal4py.pca_transform(fptype=fpType).compute(
            X, self.components_, tr_data
        )

        return tr_res.transformedData

    if sklearn_check_version("1.3"):

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            """Fit the model with X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data, where `n_samples` is the number of samples
                and `n_features` is the number of features.

            y : Ignored
                Ignored.

            Returns
            -------
            self : object
                Returns the instance itself.
            """
            self._fit(X)
            return self

    else:

        def fit(self, X, y=None):
            """Fit the model with X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data, where `n_samples` is the number of samples
                and `n_features` is the number of features.

            y : Ignored
                Ignored.

            Returns
            -------
            self : object
                Returns the instance itself.
            """
            if sklearn_check_version("1.2"):
                self._validate_params()
            elif sklearn_check_version("1.1"):
                check_scalar(
                    self.n_oversamples,
                    "n_oversamples",
                    min_val=1,
                    target_type=numbers.Integral,
                )

            self._fit(X)
            return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        _patching_status = PatchingConditionsChain("sklearn.decomposition.PCA.transform")
        _dal_ready = _patching_status.and_conditions(
            [
                (self.n_components_ > 0, "Number of components <= 0."),
                (not issparse(X), "oneDAL PCA does not support sparse input"),
            ]
        )

        _patching_status.write_log()
        if _dal_ready:
            return self._transform_daal4py(
                X, whiten=self.whiten, check_X=True, scale_eigenvalues=False
            )
        return PCA_original.transform(self, X)

    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.

        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """

        if sklearn_check_version("1.2"):
            self._validate_params()

        U, S, Vt = self._fit(X)

        _patching_status = PatchingConditionsChain(
            "sklearn.decomposition.PCA.fit_transform"
        )
        _dal_ready = _patching_status.and_conditions(
            [(U is None, "Stock fitting was used.")]
        )
        if _dal_ready:
            _dal_ready = _patching_status.and_conditions(
                [
                    (self.n_components_ > 0, "Number of components <= 0."),
                    (not issparse(X), "oneDAL PCA does not support sparse input"),
                ]
            )
            if _dal_ready:
                result = self._transform_daal4py(
                    X, whiten=self.whiten, check_X=False, scale_eigenvalues=False
                )
            else:
                result = np.empty((self.n_samples_, 0), dtype=X.dtype)
        else:
            U = U[:, : self.n_components_]

            if self.whiten:
                U *= sqrt(X.shape[0] - 1)
            else:
                U *= S[: self.n_components_]

            result = U

        _patching_status.write_log()
        return result
